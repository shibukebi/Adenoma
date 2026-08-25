#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
    if os.environ.get("_QWEN_SERVER_LD_READY") == "1":
        return
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    new_parts = list(_current_env_cuda_libs())
    for part in current:
        if part not in new_parts:
            new_parts.append(part)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    env["_QWEN_SERVER_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_torch_runtime()

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from adenoma_agent.qwen_inference import QwenInference  # noqa: E402


def _bytes_to_gb(value: int) -> float:
    return round(float(value) / float(1024**3), 3)


def _gpu_snapshot():
    available = bool(torch.cuda.is_available())
    payload = {
        "cuda_available": available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": int(torch.cuda.device_count()) if available else 0,
    }
    if not available:
        return payload
    current_index = 0
    payload["device_name"] = torch.cuda.get_device_name(current_index)
    payload["device_id"] = current_index
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(current_index)
        payload["vram_free_gb"] = _bytes_to_gb(free_bytes)
        payload["vram_total_gb"] = _bytes_to_gb(total_bytes)
        payload["vram_used_gb"] = _bytes_to_gb(total_bytes - free_bytes)
    except Exception as exc:
        payload["vram_error"] = str(exc)
    return payload


def _log_json(event_type: str, payload: dict):
    row = {"timestamp": int(time.time()), "event": event_type, **payload}
    print(json.dumps(row, ensure_ascii=False), flush=True)


class PredictRequest(BaseModel):
    image_path: Optional[str] = None
    image_paths: List[str] = Field(default_factory=list)
    prompt: str
    max_new_tokens: int = 512
    stage: Optional[str] = None


class PredictResponse(BaseModel):
    text: str
    request_id: str
    gpu_device_id: Optional[int] = None
    cuda_visible_devices: Optional[str] = None
    model_id: str
    adapter_path: Optional[str] = None
    round_trip_ms: int


def build_messages(image_paths: List[str], prompt: str):
    content = []
    for image_path in image_paths:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def parse_args():
    parser = argparse.ArgumentParser(description="Resident FastAPI model server for local_cpathagent_qwen.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--vision-encoder-id", default="built_in_from_checkpoint")
    parser.add_argument("--projector-type", default="checkpoint_native")
    parser.add_argument("--max-batch-size", type=int, default=1)
    return parser.parse_args()


def create_app(args):
    before = _gpu_snapshot()
    _log_json("startup_before_load", before)

    engine = QwenInference(
        model_id=args.model_id,
        adapter_path=args.adapter_path or None,
        device_map="auto",
        max_batch_size=int(args.max_batch_size),
        use_hf=True,
        download_model=False,
        vision_encoder_id=args.vision_encoder_id,
        projector_type=args.projector_type,
    )

    after = _gpu_snapshot()
    _log_json(
        "startup_after_load",
        {
            **after,
            "model_id": args.model_id,
            "adapter_path": args.adapter_path or None,
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
        },
    )

    app = FastAPI(title="adenoma_agent model server")

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        image_paths = list(request.image_paths or [])
        if request.image_path:
            image_paths = [request.image_path]
        stage = str(request.stage or "").strip()
        if not image_paths and stage in {"trace", "navigate", "observe_step"}:
            raise HTTPException(status_code=400, detail="image_path or image_paths is required")
        for image_path in image_paths:
            if not Path(image_path).exists():
                raise HTTPException(status_code=400, detail="image not found: {0}".format(image_path))

        request_id = str(uuid.uuid4())
        started = time.time()
        try:
            messages = build_messages(image_paths, request.prompt)
            text = engine.generate(messages, max_new_tokens=int(request.max_new_tokens), temperature=0.0)
        except Exception as exc:
            _log_json(
                "predict_error",
                {"request_id": request_id, "stage": request.stage, "error": str(exc)},
            )
            raise HTTPException(status_code=500, detail=str(exc))

        elapsed_ms = int(round((time.time() - started) * 1000.0))
        gpu = _gpu_snapshot()
        response = PredictResponse(
            text=text,
            request_id=request_id,
            gpu_device_id=gpu.get("device_id"),
            cuda_visible_devices=gpu.get("cuda_visible_devices"),
            model_id=args.model_id,
            adapter_path=args.adapter_path or None,
            round_trip_ms=elapsed_ms,
        )
        _log_json(
            "predict_ok",
            {
                "request_id": request_id,
                "stage": request.stage,
                "round_trip_ms": elapsed_ms,
                "gpu_device_id": response.gpu_device_id,
                "cuda_visible_devices": response.cuda_visible_devices,
            },
        )
        return response

    return app


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
