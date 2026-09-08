import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class ToolRun(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - tool_runs: agent, tool, parameters, status, timing.

    This is the backbone of verification. A finding stores its tool_run_id, and
    the Verifier re-executes that exact call and compares the output rather than
    asking an LLM to check itself.
    """

    __tablename__ = "tool_runs"

    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE")
    )
    hypothesis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hypotheses.id", ondelete="SET NULL")
    )
    agent_name: Mapped[str | None] = mapped_column(String(100))
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parameters: Mapped[dict | None] = mapped_column(JSONB)
    result: Mapped[dict | None] = mapped_column(JSONB)
    # success | error | timeout
    status: Mapped[str] = mapped_column(String(50), default="success", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    # so a re-run can be compared byte-for-byte
    result_checksum: Mapped[str | None] = mapped_column(String(64))


class Finding(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.18 - a verified result with its evidence chain."""

    __tablename__ = "findings"

    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    hypothesis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hypotheses.id", ondelete="SET NULL")
    )
    tool_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tool_runs.id", ondelete="SET NULL")
    )
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    # Proposal Sec.18 requires observation, statistical evidence and interpretation
    # to be distinguishable. This column is what makes that machine-checkable:
    #   measurement - describes WHAT changed (a trend, a total). Not an explanation.
    #   association - two things move together. Not causal.
    #   driver      - the change is concentrated in a specific segment.
    # Only `driver` findings are allowed to produce recommendations.
    finding_type: Mapped[str] = mapped_column(String(50), default="measurement", nullable=False)
    evidence_summary: Mapped[str | None] = mapped_column(Text)
    magnitude: Mapped[float | None] = mapped_column()
    unit: Mapped[str | None] = mapped_column(String(50))
    # high | medium | low
    confidence: Mapped[str | None] = mapped_column(String(50))
    # pending | verified | failed_verification
    verification_status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )
    caveats: Mapped[str | None] = mapped_column(Text)


class Chart(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - charts: visualization spec and rendered artifact."""

    __tablename__ = "charts"

    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE")
    )
    finding_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("findings.id", ondelete="SET NULL")
    )
    dataset_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="SET NULL")
    )
    chart_type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    spec: Mapped[dict | None] = mapped_column(JSONB)
    storage_path: Mapped[str | None] = mapped_column(Text)


class Forecast(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.12 - forecasts, always backtested and always a range."""

    __tablename__ = "forecasts"

    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    finding_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("findings.id", ondelete="SET NULL")
    )
    target_metric: Mapped[str] = mapped_column(String(255), nullable=False)
    segment: Mapped[dict | None] = mapped_column(JSONB)
    # naive | moving_average | ets | arima | sarima | prophet | gbr
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    horizon_periods: Mapped[int | None] = mapped_column(Integer)
    frequency: Mapped[str | None] = mapped_column(String(20))
    # [{period, point, lower, upper}, ...]
    predictions: Mapped[dict | None] = mapped_column(JSONB)
    # MAPE / RMSE from rolling-origin backtest, plus the naive baseline score
    backtest_metrics: Mapped[dict | None] = mapped_column(JSONB)
    series_length: Mapped[int | None] = mapped_column(Integer)
    # ok | low_confidence | withheld_insufficient_data
    reliability: Mapped[str] = mapped_column(String(50), default="ok", nullable=False)
    withheld_reason: Mapped[str | None] = mapped_column(Text)


class Recommendation(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.13 - advisory action derived from a verified finding + forecast."""

    __tablename__ = "recommendations"

    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    finding_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("findings.id", ondelete="SET NULL")
    )
    forecast_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("forecasts.id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    # {"low": 8.0, "high": 12.0, "unit": "percent_revenue_recovered"}
    expected_impact: Mapped[dict | None] = mapped_column(JSONB)
    # the exact formula + assumptions used, so the estimate is auditable
    impact_method: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[str | None] = mapped_column(String(50))
    urgency: Mapped[str | None] = mapped_column(String(50))
    rank: Mapped[int | None] = mapped_column(Integer)
    # pending | accepted | rejected | deferred
    user_decision: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)


class MetricSnapshot(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - metric_snapshots: KPI values over time for comparison."""

    __tablename__ = "metric_snapshots"

    investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="SET NULL")
    )
    dataset_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="SET NULL")
    )
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    segment: Mapped[dict | None] = mapped_column(JSONB)
    period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    value: Mapped[float | None] = mapped_column()
    calculation_definition: Mapped[str | None] = mapped_column(Text)


class InvestigationComparison(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.14 - current vs historical investigation comparison."""

    __tablename__ = "investigation_comparisons"

    current_investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    previous_investigation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="SET NULL")
    )
    # increased | decreased | disappeared | stable | replaced
    driver_change: Mapped[str | None] = mapped_column(String(50))
    comparison_summary: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict | None] = mapped_column(JSONB)


class Report(Base, UUIDMixin, TimestampMixin):
    """Proposal Sec.15 - reports: versioned final output."""

    __tablename__ = "reports"

    investigation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    executive_summary: Mapped[str | None] = mapped_column(Text)
    content: Mapped[dict | None] = mapped_column(JSONB)
    linked_finding_ids: Mapped[dict | None] = mapped_column(JSONB)
    storage_path: Mapped[str | None] = mapped_column(Text)
    # draft | final
    status: Mapped[str] = mapped_column(String(50), default="draft", nullable=False)
