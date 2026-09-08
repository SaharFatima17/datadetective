import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class DataSource(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - data_sources.

    Every input the system ever receives is registered here first: an uploaded
    file, a database connection, or a permitted URL. Provenance lives here so
    that any later finding can be traced back to where it came from.
    """

    __tablename__ = "data_sources"

    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # file | database | url
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # csv, xlsx, pdf, docx, pptx, postgres, mysql, html ...
    source_format: Mapped[str | None] = mapped_column(String(50))
    origin_uri: Mapped[str | None] = mapped_column(Text)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # registered | extracted | failed
    status: Mapped[str] = mapped_column(String(50), default="registered", nullable=False)
    source_metadata: Mapped[dict | None] = mapped_column(JSONB)
    is_deleted: Mapped[bool] = mapped_column(default=False, nullable=False)

    artifacts: Mapped[list["SourceArtifact"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class SourceArtifact(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - source_artifacts.

    The physical files kept for a source: the untouched original, plus anything
    extracted from it (text, tables, page snapshots).
    """

    __tablename__ = "source_artifacts"

    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    # original | extracted_text | extracted_table | snapshot | chart | report
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(100))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    artifact_metadata: Mapped[dict | None] = mapped_column(JSONB)

    source: Mapped["DataSource"] = relationship(back_populates="artifacts")
