import uuid

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class Brief(Base, UUIDMixin, TimestampMixin):
    """A saved brief assembled from indexed documents (proposal Sec.14).

    Kept separate from `reports` on purpose. A report belongs to an
    investigation: its figures were computed and can be recomputed. A brief
    belongs to a question asked of the knowledge base: its statements trace to
    a passage and cannot be recomputed. Storing them in one table would invite
    one screen to list both, and a reader who cannot tell them apart will trust
    the weaker one as much as the stronger.

    The sections and sources are stored as written, so reopening a brief shows
    what it said when it was made — not what the same question would return
    today, after documents have been added or removed.
    """

    __tablename__ = "briefs"

    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    sections: Mapped[list | None] = mapped_column(JSONB)
    sources: Mapped[list | None] = mapped_column(JSONB)
    # False when no model was configured and the brief is the retrieved
    # passages grouped by source rather than composed prose. Worth keeping:
    # it changes how the document should be read.
    composed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    basis: Mapped[str | None] = mapped_column(Text)