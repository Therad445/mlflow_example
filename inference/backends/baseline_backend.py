from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from inference.config import settings
from inference.pooling import mean_pool_torch


class BaselineEmbedder:
    def __init__(self) -> None:
        torch.set_num_threads(settings.torch_num_threads)
        self.model_id = settings.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()
        self.backend_name = "transformers-cpu"

    def encode(
        self,
        texts: list[str],
        prompt: str | None = None,
        normalize: bool | None = None,
    ) -> list[list[float]]:
        prefix = settings.default_prompt if prompt is None else prompt
        do_normalize = settings.normalize_default if normalize is None else normalize
        prepared_texts = [f"{prefix}{text}" if prefix else text for text in texts]

        tokenized = self.tokenizer(
            prepared_texts,
            padding=True,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**tokenized)
            embeddings = mean_pool_torch(outputs.last_hidden_state, tokenized["attention_mask"])
            if do_normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()
