from __future__ import annotations

from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from inference.config import settings


def export_model(output_dir: str | Path | None = None) -> Path:
    out_dir = Path(output_dir or settings.onnx_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
    model = ORTModelForFeatureExtraction.from_pretrained(settings.model_id, export=True)

    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    exported_dir = export_model()
    print(f"Exported ONNX model to: {exported_dir}")
