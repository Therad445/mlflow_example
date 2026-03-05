import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    String, Text, DateTime, ForeignKey, Integer, BigInteger, Enum, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Stage(str, enum.Enum):
    draft = "draft"
    staging = "staging"
    production = "production"
    archived = "archived"


class Model(Base):
    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner: Mapped[str | None] = mapped_column(String(200), index=True, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version", name="uq_model_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("models.id"), index=True, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    stage: Mapped[Stage] = mapped_column(Enum(Stage), default=Stage.draft, index=True, nullable=False)

    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    env: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    dataset_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    code_ref: Mapped[str | None] = mapped_column(Text, nullable=True)

    artifact_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=True)

    created_by: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="versions")
    artifact = relationship("Artifact")
    transitions = relationship("StageTransition", back_populates="model_version", cascade="all, delete-orphan")


class StageTransition(Base):
    __tablename__ = "stage_transitions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_versions.id"), index=True, nullable=False)
    from_stage: Mapped[Stage] = mapped_column(Enum(Stage), nullable=False)
    to_stage: Mapped[Stage] = mapped_column(Enum(Stage), nullable=False)

    actor: Mapped[str] = mapped_column(String(200), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    model_version = relationship("ModelVersion", back_populates="transitions")