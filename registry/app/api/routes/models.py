from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_actor
from app.schemas.models import ModelCreate, ModelOut, VersionCreate, VersionOut, PromoteIn
from app.services.registry import (
    create_model, list_models, get_model,
    create_version, list_versions, get_version,
    promote, get_production
)
from app.models.models import Stage

router = APIRouter()


@router.post("/models", response_model=ModelOut)
def create_model_route(payload: ModelCreate, db: Session = Depends(get_db), actor: str = Depends(get_actor)):
    m = create_model(db, payload.name, payload.description, payload.owner or actor, payload.tags)
    return ModelOut.model_validate(m, from_attributes=True)


@router.get("/models", response_model=list[ModelOut])
def list_models_route(query: str | None = None, owner: str | None = None, tag: str | None = None, db: Session = Depends(get_db)):
    items = list_models(db, query=query, owner=owner, tag=tag)
    return [ModelOut.model_validate(m, from_attributes=True) for m in items]


@router.get("/models/{model_id}", response_model=ModelOut)
def get_model_route(model_id: str, db: Session = Depends(get_db)):
    m = get_model(db, model_id)
    return ModelOut.model_validate(m, from_attributes=True)


@router.post("/models/{model_id}/versions", response_model=VersionOut)
def create_version_route(model_id: str, payload: VersionCreate, db: Session = Depends(get_db), actor: str = Depends(get_actor)):
    mv = create_version(
        db=db,
        model_id=model_id,
        actor=actor,
        metrics=payload.metrics,
        params=payload.params,
        env=payload.env,
        dataset_ref=payload.dataset_ref,
        code_ref=payload.code_ref,
        artifact_id=payload.artifact_id,
    )
    return VersionOut.model_validate(mv, from_attributes=True)


@router.get("/models/{model_id}/versions", response_model=list[VersionOut])
def list_versions_route(model_id: str, db: Session = Depends(get_db)):
    items = list_versions(db, model_id)
    return [VersionOut.model_validate(v, from_attributes=True) for v in items]


@router.get("/models/{model_id}/versions/{version}", response_model=VersionOut)
def get_version_route(model_id: str, version: int, db: Session = Depends(get_db)):
    mv = get_version(db, model_id, version)
    return VersionOut.model_validate(mv, from_attributes=True)


@router.get("/models/{model_id}/production", response_model=VersionOut)
def get_production_route(model_id: str, db: Session = Depends(get_db)):
    mv = get_production(db, model_id)
    return VersionOut.model_validate(mv, from_attributes=True)


@router.post("/models/{model_id}/versions/{version}/promote", response_model=VersionOut)
def promote_route(model_id: str, version: int, payload: PromoteIn, db: Session = Depends(get_db), actor: str = Depends(get_actor)):
    mv = promote(
        db=db,
        model_id=model_id,
        version=version,
        actor=actor,
        to_stage=Stage(payload.stage),
        comment=payload.comment,
    )
    return VersionOut.model_validate(mv, from_attributes=True)