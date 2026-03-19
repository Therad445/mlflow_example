from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Список строк для кодирования")
    prompt: str | None = Field(default=None, description="Префикс модели, например 'categorize: '")
    normalize: bool | None = Field(default=None, description="L2-нормировка эмбеддингов")


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int
    backend: str


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_id: str
    pid: int
