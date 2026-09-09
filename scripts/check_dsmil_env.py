#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DS-MIL runtime dependencies.")
    parser.add_argument("--require-gpu", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {}
    try:
        import torch

        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["torch_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        payload["torch_error"] = str(exc)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_gpu and not payload.get("torch_cuda_available", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
