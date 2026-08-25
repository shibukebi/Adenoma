#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


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
    if os.environ.get("_UNI_PRISMNET_SERVER_LD_READY") == "1":
        return
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    new_parts = list(_current_env_cuda_libs())
    for part in current:
        if part not in new_parts:
            new_parts.append(part)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    env["_UNI_PRISMNET_SERVER_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_torch_runtime()

import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from adenoma_agent.tissue_context import (  # noqa: E402
    normalized_entropy,
    tissue_context_from_probabilities,
    tissue_context_label,
)


PATHPRISM_CRC100K_LABELS = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
PATHPRISM_CLASS_NAMES = {
    "ADI": "adipose tissue",
    "BACK": "background",
    "DEB": "debris",
    "LYM": "lymphocytes",
    "MUC": "mucus",
    "MUS": "smooth muscle",
    "NORM": "normal colon mucosa",
    "STR": "cancer-associated stroma",
    "TUM": "colorectal adenocarcinoma epithelium",
}
FIVE_CLASS_MAP = {
    "ADI": "background_or_artifact",
    "BACK": "background_or_artifact",
    "DEB": "background_or_artifact",
    "MUS": "background_or_artifact",
    "LYM": "inflammatory_or_stromal_context",
    "STR": "inflammatory_or_stromal_context",
    "MUC": "mucus_rich_or_pale_context",
    "NORM": "reviewable_normal_mucosa",
    "TUM": "epithelial_neoplasia_suspicious",
}


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


class Prediction(BaseModel):
    patch_id: Optional[List[int]] = None
    image_path: str
    label: str
    crc_label: str
    class_name: str
    five_class: str
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
    if value != "auto":
        return value
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.empty(1, device="cuda:0")
    except Exception as exc:
        _log_json("uni_prismnet_cuda_unavailable", {"error": str(exc)})
        return "cpu"
    return "cuda:0"


class UniPrismNetPredictor(object):
    def __init__(self, uni_weights_path, prismnet_path, device="auto"):
        self.uni_weights_path = Path(uni_weights_path)
        self.prismnet_path = Path(prismnet_path)
        if not self.uni_weights_path.exists():
            raise FileNotFoundError(str(self.uni_weights_path))
        if not self.prismnet_path.exists():
            raise FileNotFoundError(str(self.prismnet_path))
        self.device = torch.device(_device_arg(device))
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.encoder = self._load_uni_encoder()
        self.pca_mu, self.pca_w, self.classifier = self._load_prismnet_probe()
        _log_json(
            "uni_prismnet_model_loaded",
            {
                "uni_weights_path": str(self.uni_weights_path),
                "prismnet_path": str(self.prismnet_path),
                "device": str(self.device),
                "labels": PATHPRISM_CRC100K_LABELS,
            },
        )

    def _load_uni_encoder(self):
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        state_dict = torch.load(str(self.uni_weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _load_prismnet_probe(self):
        checkpoint = torch.load(str(self.prismnet_path), map_location="cpu")
        pca_mu = checkpoint["pca_mu"].to(device=self.device, dtype=torch.float32)
        pca_w = checkpoint["pca_W"].to(device=self.device, dtype=torch.float32)
        num_classes = int(checkpoint.get("num_classes", len(PATHPRISM_CRC100K_LABELS)))
        classifier = torch.nn.Linear(int(pca_w.shape[1]), num_classes, bias=True)
        classifier.load_state_dict(checkpoint["state_dict"])
        classifier.to(self.device)
        classifier.eval()
        return pca_mu, pca_w, classifier

    def _load_image(self, path):
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

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
        with torch.inference_mode():
            features = self.encoder(batch).to(dtype=torch.float32)
            projected = (features - self.pca_mu) @ self.pca_w
            logits = self.classifier(projected)
            probs = F.softmax(logits, dim=-1)
        logits_cpu = logits.detach().cpu()
        probs_cpu = probs.detach().cpu()
        patch_ids = patch_ids or []
        predictions = []
        for index, image_path in enumerate(image_paths):
            prob_row = probs_cpu[index]
            logit_row = logits_cpu[index]
            pred_index = int(torch.argmax(prob_row).item())
            label = PATHPRISM_CRC100K_LABELS[pred_index]
            patch_id = patch_ids[index] if index < len(patch_ids) else None
            probability_map = {
                PATHPRISM_CRC100K_LABELS[class_index]: float(prob_row[class_index].item())
                for class_index in range(len(PATHPRISM_CRC100K_LABELS))
            }
            context = tissue_context_from_probabilities(probability_map, model_name="uni_prismnet")
            predictions.append(
                {
                    "patch_id": patch_id,
                    "image_path": str(image_path),
                    "label": label,
                    "crc_label": label,
                    "class_name": PATHPRISM_CLASS_NAMES[label],
                    "five_class": FIVE_CLASS_MAP[label],
                    "confidence": float(prob_row[pred_index].item()),
                    "probabilities": probability_map,
                    "logits": {
                        PATHPRISM_CRC100K_LABELS[class_index]: float(logit_row[class_index].item())
                        for class_index in range(len(PATHPRISM_CRC100K_LABELS))
                    },
                    "tissue_context": context,
                    "tissue_context_label": tissue_context_label(context),
                    "uncertainty": normalized_entropy(context),
                }
            )
        return predictions


def create_app(args):
    predictor = UniPrismNetPredictor(
        uni_weights_path=args.uni_weights_path,
        prismnet_path=args.prismnet_path,
        device=args.device,
    )
    app = FastAPI(title="uni_prismnet_roi9_server")

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
            _log_json("uni_prismnet_predict_error", {"error": str(exc)})
            raise HTTPException(status_code=500, detail=str(exc))
        latency_ms = int(round((time.time() - started) * 1000.0))
        _log_json("uni_prismnet_predict_ok", {"count": len(predictions), "latency_ms": latency_ms})
        return {
            "predictions": predictions,
            "model": "UNI+PathPrism_PrismNet_CRC100K_9CLS_ROI",
            "task": request.task or "uni_prismnet_roi9_patch_classification",
            "latency_ms": latency_ms,
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI server for UNI + PathPrism PrismNet CRC100K ROI classification.")
    parser.add_argument("--uni-weights-path", default=str(REPO_ROOT / "models" / "UNI" / "weights" / "pytorch_model.bin"))
    parser.add_argument("--prismnet-path", default=str(REPO_ROOT / "models" / "PathPrism" / "prismnet_linprobe.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8400)
    return parser.parse_args()


def main():
    args = parse_args()
    _log_json(
        "uni_prismnet_server_starting",
        {
            "uni_weights_path": args.uni_weights_path,
            "prismnet_path": args.prismnet_path,
            "device": args.device,
            "host": args.host,
            "port": args.port,
        },
    )
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
