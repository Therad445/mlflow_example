from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("EMBED_MODEL_ID", "sergeyzh/rubert-mini-frida")
    default_prompt: str = os.getenv("EMBED_DEFAULT_PROMPT", "categorize: ")
    max_length: int = int(os.getenv("EMBED_MAX_LENGTH", "512"))
    normalize_default: bool = os.getenv("EMBED_NORMALIZE_DEFAULT", "true").lower() == "true"
    torch_num_threads: int = int(os.getenv("TORCH_NUM_THREADS", "4"))
    onnx_num_threads: int = int(os.getenv("ONNX_NUM_THREADS", "4"))
    onnx_dir: Path = Path(os.getenv("ONNX_MODEL_DIR", "artifacts/onnx/rubert-mini-frida"))
    batch_max_size: int = int(os.getenv("DYN_BATCH_MAX_SIZE", "32"))
    batch_timeout_ms: int = int(os.getenv("DYN_BATCH_TIMEOUT_MS", "8"))


settings = Settings()
