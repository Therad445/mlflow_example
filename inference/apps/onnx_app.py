from __future__ import annotations

import os

from fastapi import FastAPI

from inference.backends.onnx_backend import OnnxEmbedder
from inference.schemas import EmbedRequest, EmbedResponse, HealthResponse

app = FastAPI(title="rubert-mini-frida onnx service")
embedder = OnnxEmbedder()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        backend=embedder.backend_name,
        model_id=embedder.model_id,
        pid=os.getpid(),
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    embeddings = embedder.encode(payload.texts, prompt=payload.prompt, normalize=payload.normalize)
    return EmbedResponse(
        embeddings=embeddings,
        dim=len(embeddings[0]),
        count=len(embeddings),
        backend=embedder.backend_name,
    )
