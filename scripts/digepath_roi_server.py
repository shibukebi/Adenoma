#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
DIGEPATH_APP_ROOT = REPO_ROOT / "models" / "DigeApplication"
if str(DIGEPATH_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(DIGEPATH_APP_ROOT))


def _current_env_cuda_libs():
    prefix = Path(sys.prefix)
    version_dir = "python{0}.{1}".format(sys.version_info.major, sys.version_info.minor)
    candidates = [
        prefix / "lib" / version_dir / "site-packages" / "nvidia" / "nvjitlink" / "lib",
        prefix / "lib" / version_dir / "site-packages" / "nvidia" / "cusparse" / "lib",
        prefix / "lib",
    ]
    return [str(path) for path in candidates if path.exists()]


def ensure_torch_runtime():
    if os.environ.get("_DIGEPATH_SERVER_LD_READY") == "1":
        return
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    new_parts = list(_current_env_cuda_libs())
    for part in current:
        if part not in new_parts:
            new_parts.append(part)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    env["_DIGEPATH_SERVER_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_torch_runtime()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torchvision import transforms  # noqa: E402

from adenoma_agent.tissue_context import (  # noqa: E402
    normalized_entropy,
    tissue_context_from_probabilities,
    tissue_context_label,
)

from Model_Factory.digepath_model_factory import DigepathModelFactory  # noqa: E402


def _log_json(event_type, payload):
    row = {"timestamp": int(time.time()), "event": event_type}
    row.update(payload)
    print(json.dumps(row, ensure_ascii=False), flush=True)


class PredictRequest(BaseModel):
    image_path: Optional[str] = None
    image_paths: List[str] = Field(default_factory=list)
    patch_id: Optional[List[int]] = None
    patch_ids: List[List[int]] = Field(default_factory=list)
    task: Optional[str] = None
    class_names: List[str] = Field(default_factory=list)


class Prediction(BaseModel):
    patch_id: Optional[List[int]] = None
    image_path: str
    label: str
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    logits: Dict[str, float]
    tissue_context: Dict[str, float]
    tissue_context_label: str
    uncertainty: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model: str
    task: str
    latency_ms: int


def _device_arg(value):
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return value


class DigePathRoi9Predictor(object):
    def __init__(self, digepath_model_dir, roi_model_dir, device="auto"):
        self.digepath_model_dir = Path(digepath_model_dir)
        self.roi_model_dir = Path(roi_model_dir)
        self.device = torch.device(_device_arg(device))
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.feature_model = self._load_feature_model()
        self.roi_factory = DigepathModelFactory(str(self.roi_model_dir), device=str(self.device))
        if self.roi_factory.data_type != "ROI":
            raise RuntimeError("Expected an ROI model, got {0}".format(self.roi_factory.data_type))
        self.roi_model = self.roi_factory.get_model()
        self.class_names = list(self.roi_factory.get_class_names())

    def _load_feature_model(self):
        try:
            import timm
        except Exception as exc:
            raise RuntimeError("timm is required to serve Digepath image features: {0}".format(exc))
        config_path = self.digepath_model_dir / "config.json"
        weights_path = self.digepath_model_dir / "model.safetensors"
        if not config_path.exists():
            raise FileNotFoundError(str(config_path))
        if not weights_path.exists():
            raise FileNotFoundError(str(weights_path))
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        model = timm.create_model(
            cfg.get("architecture", "vit_large_patch16_224"),
            pretrained=False,
            num_classes=0,
            global_pool=cfg.get("global_pool", "token"),
            init_values=float(cfg.get("init_values", 1.0)),
            dynamic_img_size=bool(cfg.get("dynamic_img_size", True)),
        )
        state_dict = load_file(str(weights_path), device="cpu")
        load_result = model.load_state_dict(state_dict, strict=False)
        _log_json(
            "digepath_feature_model_loaded",
            {
                "model_dir": str(self.digepath_model_dir),
                "missing_keys": len(load_result.missing_keys),
                "unexpected_keys": len(load_result.unexpected_keys),
            },
        )
        model = model.to(self.device)
        model.eval()
        return model

    def _load_image(self, path):
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB"))

    def predict(self, image_paths, patch_ids=None):
        if not image_paths:
            raise ValueError("image_path or image_paths is required")
        tensors = []
        for image_path in image_paths:
            candidate = Path(image_path)
            if not candidate.exists():
                raise FileNotFoundError(str(candidate))
            tensors.append(self._load_image(candidate))
        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            features = self.feature_model(batch)
            if isinstance(features, dict):
                for feature_key in ("x_norm_clstoken", "features", "logits"):
                    if features.get(feature_key) is not None:
                        features = features.get(feature_key)
                        break
                if isinstance(features, dict):
                    raise RuntimeError("Unsupported Digepath feature output keys: {0}".format(sorted(features.keys())))
            if isinstance(features, (list, tuple)):
                features = features[0]
            if features.dim() == 3:
                features = features[:, 0, :]
            logits = self.roi_model(features)["logits"]
            probs = F.softmax(logits, dim=-1)
        logits_cpu = logits.detach().cpu()
        probs_cpu = probs.detach().cpu()
        predictions = []
        patch_ids = patch_ids or []
        for index, image_path in enumerate(image_paths):
            prob_row = probs_cpu[index]
            logit_row = logits_cpu[index]
            pred_index = int(torch.argmax(prob_row).item())
            class_name = self.class_names[pred_index]
            patch_id = patch_ids[index] if index < len(patch_ids) else None
            probability_map = {
                self.class_names[class_index]: float(prob_row[class_index].item())
                for class_index in range(len(self.class_names))
            }
            context = tissue_context_from_probabilities(probability_map, model_name="digepath")
            predictions.append(
                {
                    "patch_id": patch_id,
                    "image_path": str(image_path),
                    "label": class_name,
                    "class_name": class_name,
                    "confidence": float(prob_row[pred_index].item()),
                    "probabilities": probability_map,
                    "logits": {
                        self.class_names[class_index]: float(logit_row[class_index].item())
                        for class_index in range(len(self.class_names))
                    },
                    "tissue_context": context,
                    "tissue_context_label": tissue_context_label(context),
                    "uncertainty": normalized_entropy(context),
                }
            )
        return predictions


def create_app(args):
    predictor = DigePathRoi9Predictor(
        digepath_model_dir=args.digepath_model_dir,
        roi_model_dir=args.roi_model_dir,
        device=args.device,
    )
    app = FastAPI(title="digepath_roi9_server")

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        started = time.time()
        image_paths = list(request.image_paths or [])
        if request.image_path:
            image_paths = [request.image_path] + [path for path in image_paths if path != request.image_path]
        patch_ids = list(request.patch_ids or [])
        if request.patch_id is not None and len(image_paths) == 1:
            patch_ids = [request.patch_id]
        try:
            predictions = predictor.predict(image_paths, patch_ids=patch_ids)
        except Exception as exc:
            _log_json("digepath_predict_error", {"error": str(exc)})
            raise HTTPException(status_code=500, detail=str(exc))
        latency_ms = int(round((time.time() - started) * 1000.0))
        _log_json("digepath_predict_ok", {"count": len(predictions), "latency_ms": latency_ms})
        return {
            "predictions": predictions,
            "model": "Digepath+BOW_CRC100K_9CLS_ROI",
            "task": request.task or "digepath_roi9_patch_classification",
            "latency_ms": latency_ms,
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI server for DIgePath CRC100K 9-class ROI screening.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--digepath-model-dir", default=str(REPO_ROOT / "models" / "Digepath"))
    parser.add_argument(
        "--roi-model-dir",
        default=str(REPO_ROOT / "models" / "DigeApplication" / "Model_Zoo" / "BOW_CRC100K_9CLS_ROI"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _log_json(
        "digepath_server_starting",
        {
            "host": args.host,
            "port": args.port,
            "device": args.device,
            "digepath_model_dir": args.digepath_model_dir,
            "roi_model_dir": args.roi_model_dir,
        },
    )
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
