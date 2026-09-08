import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin

# Investigation state machine (proposal Sec.3 / Sec.7 steps 14-16):
#   planning -> running -> awaiting_user -> running -> complete
#                                        \-> abandoned
INVESTIGATION_STATES = (
    "planning",
    "running",
    "awaiting_user",
    "complete",
    "abandoned",
    "failed",
)


class Investigation(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - investigations: question, target metric, status."""

    __tablename__ = "investigations"

    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL")
    )
    version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="SET NULL")
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    target_metric: Mapped[str | None] = mapped_column(String(255))
    dimensions: Mapped[dict | None] = mapped_column(JSONB)
    comparison_period: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(50), default="planning", nullable=False)
    plan: Mapped[dict | None] = mapped_column(JSONB)
    # guard rail: stop the supervisor looping forever (see review note on Sec.7)
    max_rounds: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    current_round: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    hypotheses: Mapped[list["Hypothesis"]] = relationship(
        back_populates="investigation", cascade="all, delete-orphan"
    )
    rounds: Mapped[list["InvestigationRound"]] = relationship(
        back_populates="investigation", cascade="all, delete-orphan"
    )


class InvestigationRound(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - investigation_rounds: each pause/resume cycle."""

    __tablename__ = "investigation_rounds"

    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    trigger: Mapped[str | None] = mapped_column(String(100))
    user_request: Mapped[str | None] = mapped_column(Text)
    user_response: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False)
    notes: Mapped[dict | None] = mapped_column(JSONB)

    investigation: Mapped["Investigation"] = relationship(back_populates="rounds")


class Hypothesis(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.11 - explicit, testable explanations that get supported or rejected."""

    __tablename__ = "hypotheses"

    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    variables: Mapped[dict | None] = mapped_column(JSONB)
    proposed_test: Mapped[str | None] = mapped_column(Text)
    # proposed | testable | blocked_missing_evidence | supported | rejected | unresolved
    status: Mapped[str] = mapped_column(String(50), default="proposed", nullable=False)
    confidence: Mapped[float | None] = mapped_column()
    rank: Mapped[int | None] = mapped_column(Integer)
    reasoning: Mapped[str | None] = mapped_column(Text)

    investigation: Mapped["Investigation"] = relationship(back_populates="hypotheses")
    evidence_links: Mapped[list["HypothesisEvidenceLink"]] = relationship(
        back_populates="hypothesis", cascade="all, delete-orphan"
    )


class MissingEvidenceRequest(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - missing_evidence_requests.

    When a hypothesis cannot be tested, the system asks instead of guessing.
    """

    __tablename__ = "missing_evidence_requests"

    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    hypothesis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hypotheses.id", ondelete="CASCADE")
    )
    round_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigation_rounds.id", ondelete="SET NULL")
    )
    # missing_field | missing_dataset | missing_document | definition | clarification
    request_type: Mapped[str] = mapped_column(String(50), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    # open | answered | unanswerable
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False)
    response: Mapped[str | None] = mapped_column(Text)


class HypothesisEvidenceLink(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - hypothesis_evidence_links.

    The join table that makes traceability real: every hypothesis points at the
    exact dataset version, document and tool run that supports or rejects it.
    """

    __tablename__ = "hypothesis_evidence_links"

    hypothesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hypotheses.id", ondelete="CASCADE"), nullable=False
    )
    # dataset_version | document | source | tool_run | finding
    evidence_type: Mapped[str] = mapped_column(String(50), nullable=False)
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # supports | rejects | inconclusive
    relation: Mapped[str | None] = mapped_column(String(50))
    note: Mapped[str | None] = mapped_column(Text)

    hypothesis: Mapped["Hypothesis"] = relationship(back_populates="evidence_links")
