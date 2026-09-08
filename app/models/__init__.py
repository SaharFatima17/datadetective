"""All SQLAlchemy models.

Every model must be imported here, otherwise Alembic's autogenerate will not
see the table and will silently leave it out of the migration.
"""

from app.models.analysis import (
    Chart,
    Finding,
    Forecast,
    InvestigationComparison,
    MetricSnapshot,
    Recommendation,
    Report,
    ToolRun,
)
from app.models.base import Base
from app.models.dataset import Dataset, DatasetColumn, DatasetVersion
from app.models.investigation import (
    Hypothesis,
    HypothesisEvidenceLink,
    Investigation,
    InvestigationRound,
    MissingEvidenceRequest,
)
from app.models.knowledge import Document, DocumentChunk, Feedback
from app.models.source import DataSource, SourceArtifact
from app.models.user import User

__all__ = [
    "Base",
    "User",
    "DataSource",
    "SourceArtifact",
    "Dataset",
    "DatasetVersion",
    "DatasetColumn",
    "Investigation",
    "InvestigationRound",
    "Hypothesis",
    "MissingEvidenceRequest",
    "HypothesisEvidenceLink",
    "ToolRun",
    "Finding",
    "Chart",
    "Forecast",
    "Recommendation",
    "MetricSnapshot",
    "InvestigationComparison",
    "Report",
    "Document",
    "DocumentChunk",
    "Feedback",
]
