import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


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
    if os.environ.get("_CPATHAGENT_QWEN_LD_READY") == "1":
        return
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    new_parts = list(_current_env_cuda_libs())
    for part in current:
        if part not in new_parts:
            new_parts.append(part)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    env["_CPATHAGENT_QWEN_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_torch_runtime()

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from swift import InferRequest, RequestConfig, TransformersEngine  # noqa: E402


@dataclass
class CPathAgentQwenBackboneSpec:
    llm_backbone_model_id: str = "Qwen/Qwen3-14B"
    multimodal_model_id: Optional[str] = None
    vision_encoder_id: str = "CPath-CLIP-compatible"
    adapter_path: Optional[str] = None
    projector_type: str = "two_layer_mlp"
    projector_hidden_act: str = "gelu"
    cache_dir: Optional[str] = None
    device_map: str = "auto"
    max_batch_size: int = 1
    use_hf: bool = True
    download_model: bool = False
    torch_dtype: torch.dtype = field(default=torch.bfloat16)
    model_kwargs: Dict = field(default_factory=dict)

    def resolved_model_id(self) -> str:
        if self.multimodal_model_id:
            return self.multimodal_model_id
        return self.llm_backbone_model_id

    def resolved_adapter_path(self) -> Optional[str]:
        if self.adapter_path and Path(self.adapter_path).exists():
            return self.adapter_path
        return None

    def to_metadata(self) -> Dict:
        return {
            "llm_backbone_model_id": self.llm_backbone_model_id,
            "multimodal_model_id": self.multimodal_model_id,
            "vision_encoder_id": self.vision_encoder_id,
            "adapter_path": self.adapter_path,
            "projector_type": self.projector_type,
            "projector_hidden_act": self.projector_hidden_act,
            "cache_dir": self.cache_dir,
        }


class CPathAgentQwenBackbone:
    def __init__(
        self,
        spec: CPathAgentQwenBackboneSpec,
    ):
        self.spec = spec
        self.model_id = spec.resolved_model_id()
        self.adapter_path = spec.resolved_adapter_path()
        if Path(self.model_id).exists():
            self.spec.use_hf = False
            self.spec.download_model = False
        if self.adapter_path and Path(self.adapter_path).exists():
            self.spec.use_hf = False
        self.model_config = self._load_model_config()
        self.engine = TransformersEngine(
            self.model_id,
            adapters=[self.adapter_path] if self.adapter_path else None,
            max_batch_size=spec.max_batch_size,
            torch_dtype=spec.torch_dtype,
            device_map=spec.device_map,
            use_hf=self.spec.use_hf,
            download_model=self.spec.download_model,
            model_kwargs=dict(spec.model_kwargs),
        )

    def _load_model_config(self):
        try:
            return AutoConfig.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                local_files_only=not self.spec.download_model,
            )
        except Exception:
            return None

    def is_multimodal_ready(self) -> bool:
        model_id_lower = str(self.model_id or "").lower()
        if "pathreasoner-r1" in model_id_lower or "patho-r1" in model_id_lower:
            return True
        config = self.model_config
        if config is None:
            return False
        if getattr(config, "vision_config", None) is not None:
            return True
        model_type = str(getattr(config, "model_type", "")).lower()
        architectures = [str(name).lower() for name in getattr(config, "architectures", [])]
        if "vl" in model_type or any("vl" in item for item in architectures):
            return True
        if "vision" in model_type or any("vision" in item for item in architectures):
            return True
        return False

    def ensure_inference_ready(self, expect_vision: bool):
        if not expect_vision:
            return
        if self.is_multimodal_ready():
            return
        raise RuntimeError(
            "The configured serving model '{0}' is not a ready multimodal checkpoint. "
            "To mirror the paper architecture, provide either a merged multimodal checkpoint in "
            "`cpathagent_qwen_multimodal_model_id` or a loadable checkpoint that already includes "
            "Qwen backbone + pathology vision encoder ({1}) + {2} projector.".format(
                self.model_id,
                self.spec.vision_encoder_id,
                self.spec.projector_type,
            )
        )

    def describe_architecture(self) -> Dict:
        return {
            **self.spec.to_metadata(),
            "serving_model_id": self.model_id,
            "multimodal_ready": self.is_multimodal_ready(),
        }

    def generate(self, messages: List[dict], max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        expect_vision = any(
            isinstance(message.get("content"), list)
            and any(isinstance(item, dict) and item.get("type") == "image" for item in message.get("content", []))
            for message in messages
        )
        self.ensure_inference_ready(expect_vision=expect_vision)
        infer_request = InferRequest(messages=messages)
        request_config = RequestConfig(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=False,
        )
        response = self.engine.infer([infer_request], request_config=request_config, use_tqdm=False)[0]
        return response.choices[0].message.content or ""


class QwenInference(CPathAgentQwenBackbone):
    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        max_batch_size: int = 1,
        use_hf: bool = True,
        download_model: bool = False,
        model_kwargs: Optional[Dict] = None,
        vision_encoder_id: str = "CPath-CLIP-compatible",
        projector_type: str = "two_layer_mlp",
    ):
        spec = CPathAgentQwenBackboneSpec(
            llm_backbone_model_id=model_id,
            multimodal_model_id=model_id,
            vision_encoder_id=vision_encoder_id,
            adapter_path=adapter_path,
            projector_type=projector_type,
            device_map=device_map,
            max_batch_size=max_batch_size,
            use_hf=use_hf,
            download_model=download_model,
            model_kwargs=dict(model_kwargs or {}),
        )
        super().__init__(spec)
