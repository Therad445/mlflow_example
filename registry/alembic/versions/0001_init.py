"""init

Revision ID: 0001_init
Revises:
Create Date: 2026-03-04

"""
from alembic import op
import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as psql

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    stage_enum = psql.ENUM(
        "draft",
        "staging",
        "production",
        "archived",
        name="stage",
        create_type=False,
    )
    stage_enum.create(op.get_bind(), checkfirst=True)

    # models
    op.create_table(
        "models",
        sa.Column("id", psql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner", sa.String(length=200), nullable=True),
        sa.Column("tags", psql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_models_name", "models", ["name"], unique=True)
    op.create_index("ix_models_owner", "models", ["owner"], unique=False)

    # artifacts
    op.create_table(
        "artifacts",
        sa.Column("id", psql.UUID(as_uuid=True), primary_key=True),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_artifacts_sha256", "artifacts", ["sha256"], unique=False)

    # model_versions
    op.create_table(
        "model_versions",
        sa.Column("id", psql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_id", psql.UUID(as_uuid=True), sa.ForeignKey("models.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("stage", stage_enum, nullable=False),
        sa.Column("metrics", psql.JSONB(), nullable=True),
        sa.Column("params", psql.JSONB(), nullable=True),
        sa.Column("env", psql.JSONB(), nullable=True),
        sa.Column("dataset_ref", sa.Text(), nullable=True),
        sa.Column("code_ref", sa.Text(), nullable=True),
        sa.Column("artifact_id", psql.UUID(as_uuid=True), sa.ForeignKey("artifacts.id"), nullable=True),
        sa.Column("created_by", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("model_id", "version", name="uq_model_version"),
    )
    op.create_index("ix_model_versions_model_id", "model_versions", ["model_id"], unique=False)
    op.create_index("ix_model_versions_stage", "model_versions", ["stage"], unique=False)

    # stage_transitions
    op.create_table(
        "stage_transitions",
        sa.Column("id", psql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version_id", psql.UUID(as_uuid=True), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("from_stage", stage_enum, nullable=False),
        sa.Column("to_stage", stage_enum, nullable=False),
        sa.Column("actor", sa.String(length=200), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_stage_transitions_model_version_id", "stage_transitions", ["model_version_id"], unique=False)


def downgrade():
    op.drop_index("ix_stage_transitions_model_version_id", table_name="stage_transitions")
    op.drop_table("stage_transitions")

    op.drop_index("ix_model_versions_stage", table_name="model_versions")
    op.drop_index("ix_model_versions_model_id", table_name="model_versions")
    op.drop_table("model_versions")

    op.drop_index("ix_artifacts_sha256", table_name="artifacts")
    op.drop_table("artifacts")

    op.drop_index("ix_models_owner", table_name="models")
    op.drop_index("ix_models_name", table_name="models")
    op.drop_table("models")

    # Drop enum type last
    op.execute("DROP TYPE IF EXISTS stage")