from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI

from inference.backends.dynamic_batcher import DynamicBatchingService
from inference.schemas import EmbedRequest, EmbedResponse, HealthResponse

app = FastAPI(title="rubert-mini-frida dynamic batching service")
service = DynamicBatchingService()


@app.on_event("shutdown")
def shutdown_event() -> None:
    service.stop()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        backend=service.backend_name,
        model_id=service.embedder.model_id,
        pid=os.getpid(),
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(payload: EmbedRequest) -> EmbedResponse:
    future = service.submit(payload.texts, payload.prompt, payload.normalize)
    embeddings = await asyncio.wrap_future(future)
    return EmbedResponse(
        embeddings=embeddings,
        dim=len(embeddings[0]),
        count=len(embeddings),
        backend=service.backend_name,
    )
