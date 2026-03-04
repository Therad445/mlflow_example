from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_actor
from app.schemas.models import (
    ArtifactPresignUploadIn, PresignedUrlOut, ArtifactCommitIn, ArtifactOut
)
from app.services.artifacts import presign_put, presign_get, build_s3_uri
from app.services.registry import create_artifact_record

router = APIRouter()


@router.post("/artifacts/presign-upload", response_model=PresignedUrlOut)
def presign_upload(payload: ArtifactPresignUploadIn, actor: str = Depends(get_actor)):
    url, headers = presign_put(payload.object_name, payload.content_type)
    return PresignedUrlOut(url=url, method="PUT", headers=headers)


@router.get("/artifacts/presign-download")
def presign_download(object_name: str, actor: str = Depends(get_actor)):
    url = presign_get(object_name)
    return {"url": url, "method": "GET", "headers": {}}


@router.post("/artifacts/commit", response_model=ArtifactOut)
def commit_artifact(payload: ArtifactCommitIn, db: Session = Depends(get_db), actor: str = Depends(get_actor)):
    uri = build_s3_uri(payload.object_name)
    a = create_artifact_record(db, uri=uri, sha256=payload.sha256, size_bytes=payload.size_bytes)
    return ArtifactOut.model_validate(a, from_attributes=True)