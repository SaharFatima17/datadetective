from __future__ import annotations

from typing import Any

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str | None = None
    organization: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class RoleUpdate(BaseModel):
    role: str


class SQLIngestRequest(BaseModel):
    connection_url: str = Field(..., description="Not stored - used for this call only")
    query: str
    name: str


class URLIngestRequest(BaseModel):
    subject: str | None = None
    url: str
    index_for_rag: bool = True


class DatasetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class ColumnSensitivityUpdate(BaseModel):
    sensitive: bool


class CleaningApplyRequest(BaseModel):
    approved_op_ids: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    version_id: str | None = None


class SQLQueryRequest(BaseModel):
    query: str
    version_id: str | None = None


class StatisticalTestRequest(BaseModel):
    test: str
    params: dict[str, Any] = Field(default_factory=dict)
    version_id: str | None = None


class ChartRequest(BaseModel):
    spec: dict[str, Any]
    investigation_id: str | None = None
    version_id: str | None = None


class ForecastRequest(BaseModel):
    date_column: str
    metric: str
    horizon: int = 3
    freq: str = "M"
    agg: str = "sum"
    filters: dict[str, Any] | None = None
    version_id: str | None = None


class DocumentIndexRequest(BaseModel):
    title: str
    text: str
    document_type: str = "business_doc"
    metadata: dict[str, Any] | None = None


class SubjectUpdate(BaseModel):
    # Empty or null returns the document to grouping by where it came from.
    subject: str | None = None


class SearchRequest(BaseModel):
    # Limits retrieval to one body of knowledge — a crawled site, the uploads,
    # or past reports. Without it a large crawl can bury a one-page note.
    scope: str | None = None
    query: str
    top_k: int = 5
    document_type: str | None = None


class InvestigationCreate(BaseModel):
    dataset_id: str
    question: str
    forecast_horizon: int = 3
    # optional date window; needed to compare drivers across periods (Sec.14)
    period_start: str | None = None
    period_end: str | None = None


class ResumeRequest(BaseModel):
    request_id: str
    response: str


class ConversationCreate(BaseModel):
    title: str | None = None
    dataset_id: str | None = None
    # Start a thread already attached to a finished investigation, so someone
    # who has been handed a report can ask about it without re-running anything.
    investigation_id: str | None = None


class CrawlRequest(BaseModel):
    url: str
    subject: str | None = None
    # Kept small on purpose: a crawl nobody can review is a crawl nobody trusts.
    max_pages: int = 15
    max_depth: int = 2


class ChatMessageRequest(BaseModel):
    content: str


class FeedbackRequest(BaseModel):
    rating: str = Field(..., description="useful | not_useful")
    notes: str | None = None


class ToolCallRequest(BaseModel):
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)