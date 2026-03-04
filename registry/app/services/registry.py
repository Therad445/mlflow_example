from sqlalchemy.orm import Session
from sqlalchemy import select, func
from fastapi import HTTPException

from app.models.models import Model, ModelVersion, Stage, StageTransition, Artifact


def create_model(db: Session, name: str, description: str | None, owner: str | None, tags: list[str] | None) -> Model:
    existing = db.scalar(select(Model).where(Model.name == name))
    if existing:
        raise HTTPException(status_code=409, detail="Model with this name already exists")
    m = Model(name=name, description=description, owner=owner, tags=tags)
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def list_models(db: Session, query: str | None, owner: str | None, tag: str | None) -> list[Model]:
    stmt = select(Model)
    if query:
        stmt = stmt.where(Model.name.ilike(f"%{query}%"))
    if owner:
        stmt = stmt.where(Model.owner == owner)
    if tag:
        stmt = stmt.where(Model.tags.contains([tag]))
    stmt = stmt.order_by(Model.created_at.desc())
    return list(db.scalars(stmt).all())


def get_model(db: Session, model_id: str) -> Model:
    m = db.get(Model, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    return m


def next_version_number(db: Session, model_id: str) -> int:
    max_v = db.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.model_id == model_id))
    return int(max_v or 0) + 1


def create_version(
    db: Session,
    model_id: str,
    actor: str,
    metrics: dict | None,
    params: dict | None,
    env: dict | None,
    dataset_ref: str | None,
    code_ref: str | None,
    artifact_id: str | None,
) -> ModelVersion:
    _ = get_model(db, model_id)
    vnum = next_version_number(db, model_id)
    mv = ModelVersion(
        model_id=model_id,
        version=vnum,
        stage=Stage.draft,
        metrics=metrics,
        params=params,
        env=env,
        dataset_ref=dataset_ref,
        code_ref=code_ref,
        artifact_id=artifact_id,
        created_by=actor,
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


def list_versions(db: Session, model_id: str) -> list[ModelVersion]:
    _ = get_model(db, model_id)
    stmt = select(ModelVersion).where(ModelVersion.model_id == model_id).order_by(ModelVersion.version.desc())
    return list(db.scalars(stmt).all())


def get_version(db: Session, model_id: str, version: int) -> ModelVersion:
    stmt = select(ModelVersion).where(ModelVersion.model_id == model_id, ModelVersion.version == version)
    mv = db.scalar(stmt)
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")
    return mv


def get_production(db: Session, model_id: str) -> ModelVersion:
    _ = get_model(db, model_id)
    stmt = (
        select(ModelVersion)
        .where(ModelVersion.model_id == model_id, ModelVersion.stage == Stage.production)
        .order_by(ModelVersion.version.desc())
        .limit(1)
    )
    mv = db.scalar(stmt)
    if not mv:
        raise HTTPException(status_code=404, detail="No production version")
    return mv


def promote(db: Session, model_id: str, version: int, actor: str, to_stage: Stage, comment: str | None) -> ModelVersion:
    mv = get_version(db, model_id, version)
    from_stage = mv.stage

    if from_stage == to_stage:
        return mv

    # базовая логика
    # production допускаем только из staging
    if to_stage == Stage.production and from_stage != Stage.staging:
        raise HTTPException(status_code=400, detail="To promote to production, version must be in staging first")

    mv.stage = to_stage
    tr = StageTransition(
        model_version_id=mv.id,
        from_stage=from_stage,
        to_stage=to_stage,
        actor=actor,
        comment=comment,
    )
    db.add(tr)
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


def create_artifact_record(db: Session, uri: str, sha256: str, size_bytes: int) -> Artifact:
    # дедуп
    existing = db.scalar(select(Artifact).where(Artifact.sha256 == sha256))
    if existing:
        return existing
    a = Artifact(uri=uri, sha256=sha256, size_bytes=size_bytes)
    db.add(a)
    db.commit()
    db.refresh(a)
    return a