from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from inference.config import settings
from inference.pooling import l2_normalize_numpy, mean_pool_numpy


class OnnxEmbedder:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir or settings.onnx_dir)
        self.model_id = settings.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = settings.onnx_num_threads
        session_options.inter_op_num_threads = 1

        model_path = self.model_dir / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. Export it first with export_to_onnx.py"
            )

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.backend_name = "onnxruntime-cpu"

    def _run_prepared(self, prepared_texts: list[str]) -> np.ndarray:
        tokenized = self.tokenizer(
            prepared_texts,
            padding=True,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="np",
        )

        ort_inputs: dict[str, np.ndarray] = {}
        for key, value in tokenized.items():
            if key in self.input_names:
                ort_inputs[key] = value.astype(np.int64)

        outputs = self.session.run(None, ort_inputs)
        hidden_state = outputs[0]
        attention_mask = ort_inputs["attention_mask"]
        embeddings = mean_pool_numpy(hidden_state, attention_mask)
        return embeddings.astype(np.float32)

    def encode_prepared(self, prepared_texts: list[str], normalize: bool | None = None) -> list[list[float]]:
        do_normalize = settings.normalize_default if normalize is None else normalize
        embeddings = self._run_prepared(prepared_texts)
        if do_normalize:
            embeddings = l2_normalize_numpy(embeddings)
        return embeddings.astype(np.float32).tolist()

    def encode(
        self,
        texts: list[str],
        prompt: str | None = None,
        normalize: bool | None = None,
    ) -> list[list[float]]:
        prefix = settings.default_prompt if prompt is None else prompt
        prepared_texts = [f"{prefix}{text}" if prefix else text for text in texts]
        return self.encode_prepared(prepared_texts, normalize=normalize)
