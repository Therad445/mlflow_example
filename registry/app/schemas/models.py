from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict, Literal


Stage = Literal["draft", "staging", "production", "archived"]


class ModelCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[List[str]] = None


class ModelOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: datetime


class VersionCreate(BaseModel):
    metrics: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None
    dataset_ref: Optional[str] = None
    code_ref: Optional[str] = None
    artifact_id: Optional[str] = None


class VersionOut(BaseModel):
    id: str
    model_id: str
    version: int
    stage: Stage
    metrics: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None
    dataset_ref: Optional[str] = None
    code_ref: Optional[str] = None
    artifact_id: Optional[str] = None
    created_by: str
    created_at: datetime


class PromoteIn(BaseModel):
    stage: Stage
    comment: Optional[str] = None


class TransitionOut(BaseModel):
    id: str
    model_version_id: str
    from_stage: Stage
    to_stage: Stage
    actor: str
    comment: Optional[str] = None
    created_at: datetime


class ArtifactPresignUploadIn(BaseModel):
    object_name: str = Field(..., description="Напр. models/fraud_detector/v3/model.zip")
    content_type: Optional[str] = None


class ArtifactCommitIn(BaseModel):
    object_name: str
    sha256: str = Field(..., min_length=64, max_length=64)
    size_bytes: int = Field(..., ge=0)


class ArtifactOut(BaseModel):
    id: str
    uri: str
    sha256: str
    size_bytes: int
    created_at: datetime


class PresignedUrlOut(BaseModel):
    url: str
    method: str
    headers: dict = {}