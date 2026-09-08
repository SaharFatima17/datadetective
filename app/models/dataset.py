import uuid

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class Dataset(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - datasets: the tabular data extracted from a source."""

    __tablename__ = "datasets"

    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # Proposal Sec.19: columns whose VALUES must never be sent to an LLM.
    # Held at dataset level rather than per version, so the marking survives
    # cleaning, re-profiling and every new version. Shape: {"columns": [...]}.
    sensitive_columns: Mapped[dict | None] = mapped_column(JSONB)
    is_deleted: Mapped[bool] = mapped_column(default=False, nullable=False)

    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        foreign_keys="DatasetVersion.dataset_id",
    )


class DatasetVersion(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.10 - a new version is created rather than overwriting data.

    parent_version_id + cleaning_operations together form the lineage chain:
    you can always walk back from a cleaned version to the raw upload.
    """

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version_number"),)

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="SET NULL")
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # raw | cleaned | derived
    version_type: Mapped[str] = mapped_column(String(50), default="raw", nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int | None] = mapped_column(BigInteger)
    column_count: Mapped[int | None] = mapped_column(Integer)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    # the cleaning ops that produced this version, with reasons and affected rows
    cleaning_operations: Mapped[dict | None] = mapped_column(JSONB)
    profile_summary: Mapped[dict | None] = mapped_column(JSONB)
    created_by_agent: Mapped[str | None] = mapped_column(String(100))

    dataset: Mapped["Dataset"] = relationship(
        back_populates="versions", foreign_keys=[dataset_id]
    )
    columns: Mapped[list["DatasetColumn"]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )


class DatasetColumn(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - columns: schema, inferred type, semantic label, quality metrics."""

    __tablename__ = "columns"

    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_dtype: Mapped[str | None] = mapped_column(String(50))
    # numeric | categorical | datetime | text | id | boolean
    inferred_type: Mapped[str | None] = mapped_column(String(50))
    # revenue, date, region, customer_id ... (heuristic now, LLM Profiler in Phase 7)
    semantic_label: Mapped[str | None] = mapped_column(String(100))
    null_count: Mapped[int | None] = mapped_column(BigInteger)
    null_percentage: Mapped[float | None] = mapped_column()
    unique_count: Mapped[int | None] = mapped_column(BigInteger)
    # min/max/mean/std, top categories, sample values
    statistics: Mapped[dict | None] = mapped_column(JSONB)
    quality_issues: Mapped[dict | None] = mapped_column(JSONB)
    # Mirrors Dataset.sensitive_columns so the UI can show the flag per column.
    sensitive: Mapped[bool] = mapped_column(default=False, nullable=False)

    version: Mapped["DatasetVersion"] = relationship(back_populates="columns")
