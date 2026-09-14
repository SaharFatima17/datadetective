import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class Conversation(Base, UUIDMixin, TimestampMixin):
    """A chat thread (proposal Sec.17, conversational interface).

    A conversation is not a separate investigation engine: it holds context and
    delegates to the same orchestrator every other route uses. What it owns is
    the thread itself — which dataset is in play, which investigation is
    currently open, and every message exchanged.
    """

    __tablename__ = "conversations"

    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(String(255), default="New conversation",
                                       nullable=False)
    # the dataset the thread is currently talking about, if one has been chosen
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL")
    )
    # the investigation this thread is currently working through
    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="SET NULL")
    )
    is_archived: Mapped[bool] = mapped_column(default=False, nullable=False)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base, UUIDMixin, TimestampMixin):
    """One turn in a conversation.

    `kind` tells the interface how to render the payload: plain prose, a set of
    findings, a request for data, a finished report. Keeping the structured
    result in `payload` rather than flattening it into text means the thread can
    show a real findings card on reload, not a paragraph describing one.
    """

    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # user | assistant
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    # text | findings | request | report | dataset | error
    kind: Mapped[str] = mapped_column(String(30), default="text", nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSONB)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")