import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class Document(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.14 - documents: RAG source metadata."""

    __tablename__ = "documents"

    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # business_doc | kpi_definition | data_dictionary | past_report | web_page
    document_type: Mapped[str | None] = mapped_column(String(50))
    organization: Mapped[str | None] = mapped_column(String(255))
    department: Mapped[str | None] = mapped_column(String(255))
    extracted_text: Mapped[str | None] = mapped_column(Text)
    # pending | chunked | embedded | failed
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)
    document_metadata: Mapped[dict | None] = mapped_column(JSONB)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.14 - document_chunks.

    The embedding column is added in Phase 6 once the vector store is chosen
    (pgvector or Qdrant, per Sec.16). For now the chunk text and its metadata
    are stored, plus a reference to wherever the vector ends up living.
    """

    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    # filters used at retrieval time: organization, date, dataset_id, investigation_id
    chunk_metadata: Mapped[dict | None] = mapped_column(JSONB)
    vector_ref: Mapped[str | None] = mapped_column(String(255))

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Feedback(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - feedback: user validation of findings and recommendations.

    Also feeds Sec.13's historical grounding: past recommendations rated useful
    are retrievable when generating new ones.
    """

    __tablename__ = "feedback"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE")
    )
    # finding | recommendation | report | forecast
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    rating: Mapped[int | None] = mapped_column(Integer)
    was_adopted: Mapped[bool | None] = mapped_column()
    comment: Mapped[str | None] = mapped_column(Text)
