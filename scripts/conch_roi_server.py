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
    if os.environ.get("_CONCH_SERVER_LD_READY") == "1":
        return
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    new_parts = list(_current_env_cuda_libs())
    for part in current:
        if part not in new_parts:
            new_parts.append(part)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    env["_CONCH_SERVER_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_torch_runtime()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import uvicorn  # noqa: E402
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from PIL import Image  # noqa: E402

from adenoma_agent.tissue_context import (  # noqa: E402
    normalized_entropy,
    tissue_context_from_probabilities,
    tissue_context_label,
)


CRC100K_CLASS_NAMES = [
    {
        "label": "ADI",
        "names": ["adipose tissue", "fat tissue"],
    },
    {
        "label": "BACK",
        "names": ["background", "empty slide background", "blank glass background"],
    },
    {
        "label": "DEB",
        "names": ["debris", "necrotic debris", "cellular debris"],
    },
    {
        "label": "LYM",
        "names": ["lymphocytes", "lymphoid inflammatory cells", "lymphocytic infiltrate"],
    },
    {
        "label": "MUC",
        "names": ["mucus", "mucin", "mucin pool"],
    },
    {
        "label": "MUS",
        "names": ["smooth muscle", "muscularis propria", "muscle tissue"],
    },
    {
        "label": "NORM",
        "names": ["normal colon mucosa", "benign colorectal mucosa", "normal colonic epithelium"],
    },
    {
        "label": "STR",
        "names": ["stroma", "fibrous stroma", "desmoplastic stroma"],
    },
    {
        "label": "TUM",
        "names": ["tumor epithelium", "neoplastic epithelium", "colorectal adenocarcinoma epithelium"],
    },
]

ZEROSHOT_TEMPLATES = [
    "a histopathology image of CLASSNAME.",
    "a hematoxylin and eosin stained colorectal tissue tile showing CLASSNAME.",
    "a pathology patch containing CLASSNAME.",
    "a cropped whole slide image region with CLASSNAME.",
]


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
    labels: List[str] = Field(default_factory=list)
    crc100k_labels: List[str] = Field(default_factory=list)


class Prediction(BaseModel):
    patch_id: Optional[List[int]] = None
    image_path: str
    label: str
    crc_label: str
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
    if value != "auto":
        return value
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.empty(1, device="cuda:0")
    except Exception as exc:
        _log_json("conch_cuda_unavailable", {"error": str(exc)})
        return "cpu"
    return "cuda:0"


class ConchCrc100kPredictor(object):
    def __init__(self, model_path, device="auto"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))
        self.device = torch.device(_device_arg(device))
        self.labels = [item["label"] for item in CRC100K_CLASS_NAMES]
        self.classname_lookup = {
            item["label"]: item["names"][0]
            for item in CRC100K_CLASS_NAMES
        }
        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            str(self.model_path),
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = get_tokenizer()
        self.classifier = self._build_classifier()
        _log_json(
            "conch_model_loaded",
            {
                "model_path": str(self.model_path),
                "device": str(self.device),
                "labels": self.labels,
                "template_count": len(ZEROSHOT_TEMPLATES),
            },
        )

    def _build_classifier(self):
        classnames = [item["names"] for item in CRC100K_CLASS_NAMES]
        zeroshot_weights = []
        with torch.inference_mode():
            for classnames_for_class in classnames:
                embeddings_for_class = []
                for classname in classnames_for_class:
                    texts = [template.replace("CLASSNAME", classname) for template in ZEROSHOT_TEMPLATES]
                    token_ids = self._tokenize_texts(texts).to(self.device)
                    classname_embeddings = self.model.encode_text(token_ids)
                    embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))
                class_embedding = torch.stack(embeddings_for_class, dim=0).mean(dim=(0, 1))
                class_embedding = F.normalize(class_embedding, dim=-1)
                zeroshot_weights.append(class_embedding)
        classifier = torch.stack(zeroshot_weights, dim=1)
        return classifier.to(self.device)

    def _tokenize_texts(self, texts):
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and hasattr(self.tokenizer, "token_to_id"):
            pad_token_id = self.tokenizer.token_to_id("<pad>")
        if pad_token_id is None:
            pad_token_id = 0
        if callable(self.tokenizer):
            try:
                encoded = self.tokenizer(
                    texts,
                    max_length=127,
                    add_special_tokens=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                return F.pad(encoded["input_ids"], (0, 1), value=pad_token_id)
            except Exception:
                pass
        if hasattr(self.tokenizer, "batch_encode_plus"):
            encoded = self.tokenizer.batch_encode_plus(
                texts,
                max_length=127,
                add_special_tokens=True,
                return_token_type_ids=False,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return F.pad(encoded["input_ids"], (0, 1), value=pad_token_id)
        if hasattr(self.tokenizer, "encode_batch"):
            encodings = self.tokenizer.encode_batch(texts)
            rows = []
            for encoding in encodings:
                ids = list(getattr(encoding, "ids", encoding))
                ids = ids[:127]
                ids = ids + [pad_token_id] * max(0, 127 - len(ids))
                ids.append(pad_token_id)
                rows.append(ids)
            return torch.tensor(rows, dtype=torch.long)
        raise RuntimeError("Unsupported CONCH tokenizer type: {0}".format(type(self.tokenizer).__name__))

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
        with torch.inference_mode():
            image_features = self.model.encode_image(batch, proj_contrast=True, normalize=True)
            logits = image_features @ self.classifier
            scale = float(self.model.logit_scale.exp().detach().cpu().item()) if hasattr(self.model, "logit_scale") else 1.0
            scaled_logits = logits * scale
            probs = F.softmax(scaled_logits, dim=-1)
        logits_cpu = scaled_logits.detach().cpu()
        probs_cpu = probs.detach().cpu()
        patch_ids = patch_ids or []
        predictions = []
        for index, image_path in enumerate(image_paths):
            prob_row = probs_cpu[index]
            logit_row = logits_cpu[index]
            pred_index = int(torch.argmax(prob_row).item())
            label = self.labels[pred_index]
            patch_id = patch_ids[index] if index < len(patch_ids) else None
            probability_map = {
                self.labels[class_index]: float(prob_row[class_index].item())
                for class_index in range(len(self.labels))
            }
            context = tissue_context_from_probabilities(probability_map, model_name="conch_zeroshot")
            predictions.append(
                {
                    "patch_id": patch_id,
                    "image_path": str(image_path),
                    "label": label,
                    "crc_label": label,
                    "class_name": self.classname_lookup[label],
                    "confidence": float(prob_row[pred_index].item()),
                    "probabilities": probability_map,
                    "logits": {
                        self.labels[class_index]: float(logit_row[class_index].item())
                        for class_index in range(len(self.labels))
                    },
                    "tissue_context": context,
                    "tissue_context_label": tissue_context_label(context),
                    "uncertainty": normalized_entropy(context),
                }
            )
        return predictions


def create_app(args):
    predictor = ConchCrc100kPredictor(
        model_path=args.model_path,
        device=args.device,
    )
    app = FastAPI(title="conch_crc100k_roi_server")

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
            _log_json("conch_predict_error", {"error": str(exc)})
            raise HTTPException(status_code=500, detail=str(exc))
        latency_ms = int(round((time.time() - started) * 1000.0))
        _log_json("conch_predict_ok", {"count": len(predictions), "latency_ms": latency_ms})
        return {
            "predictions": predictions,
            "model": "CONCH_ViT-B-16_zero_shot_CRC100K_ROI9",
            "task": request.task or "conch_crc100k_patch_classification",
            "latency_ms": latency_ms,
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI server for CONCH zero-shot CRC100K 9-class ROI screening.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "models" / "CONCH" / "pytorch_model.bin"))
    return parser.parse_args()


def main():
    args = parse_args()
    _log_json(
        "conch_server_starting",
        {
            "host": args.host,
            "port": args.port,
            "device": args.device,
            "model_path": args.model_path,
        },
    )
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
