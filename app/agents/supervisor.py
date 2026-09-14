"""Supervisor / Planner Agent (proposal Sec.6, Sec.7 steps 10-12).

Turns an open-ended business question into a measurable plan: which column is
the target metric, which column carries time, which columns are candidate
dimensions, and what the comparison periods are.

It uses the profile (deterministic) to constrain what the LLM is allowed to
choose, so the plan can only ever reference columns that actually exist.
"""

from __future__ import annotations

import pandas as pd

from sqlalchemy.orm import Session

from app.llm.client import llm
from app.services import rag
from app.services.profiling import parse_datetimes

SYSTEM = (
    "You are the planner for a data investigation system. You are given a business "
    "question and the real column list of a dataset. Choose ONLY from the columns "
    "given - never invent a column name. Return JSON with keys: target_metric "
    "(string or null), date_column (string or null), dimensions (array of strings), "
    "reasoning (string)."
)


def lookup_definitions(db: Session, question: str, columns: list[str],
                       owner_id=None) -> dict:
    """Retrieve business definitions relevant to the question (proposal Sec.14).

    Without this the RAG layer only ever sees past reports, and the
    "multi-source investigation" use case in Sec.5 - spreadsheet metrics plus
    evidence from an uploaded document - never actually happens.
    """
    found: dict[str, dict] = {}

    # terms the user named, then the dataset's own column names
    terms = [w.strip(" ?.,") for w in question.split() if len(w) > 4]
    for term in terms[:6] + columns[:8]:
        if term.lower() in found:
            continue
        definition = rag.get_business_definition(db, term, owner_id=owner_id)
        if definition and definition.get("definition"):
            found[term.lower()] = definition

    # Uploaded documents and retrieved web pages are both background context.
    # Indexing a page and then never consulting it would make URL ingestion a
    # dead end, which is not what Sec.5 and Sec.7 describe.
    context = rag.search(
        db, question, top_k=3, document_type=["business_doc", "web_page"],
        owner_id=owner_id,
    )
    return {"definitions": list(found.values()), "context_passages": context}


def choose_columns(question: str, profile: dict, definitions: dict | None = None) -> dict:
    columns = profile.get("columns", [])
    numeric = [c["name"] for c in columns if c["inferred_type"] == "numeric"]
    dates = [c["name"] for c in columns if c["inferred_type"] == "datetime"]
    categorical = [c["name"] for c in columns if c["inferred_type"] == "categorical"]

    context_block = ""
    if definitions:
        lines = [f"- {d['term']}: {d['definition'][:300]}"
                 for d in definitions.get("definitions", [])]
        passages = [f"- {c['document_title']}: {c['content'][:300]}"
                    for c in definitions.get("context_passages", [])]
        if lines:
            context_block += "\n\nBusiness definitions retrieved:\n" + "\n".join(lines)
        if passages:
            context_block += "\n\nRelated business context:\n" + "\n".join(passages)

    try:
        plan = llm.complete_json(
            system=SYSTEM,
            prompt=(
                f"Question: {question}\n\n"
                f"Numeric columns (candidate target metrics): {numeric}\n"
                f"Datetime columns: {dates}\n"
                f"Categorical columns (candidate dimensions): {categorical}"
                f"{context_block}\n\n"
                "Create the investigation plan."
            ),
        )
    except Exception:  # noqa: BLE001
        plan = {}

    target = plan.get("target_metric")
    if target not in numeric:
        target = _guess_metric(question, columns, numeric)

    date_col = plan.get("date_column")
    if date_col not in dates:
        date_col = dates[0] if dates else None

    dimensions = [d for d in (plan.get("dimensions") or []) if d in categorical]
    if not dimensions:
        dimensions = _rank_dimensions(categorical, columns)[:3]

    return {
        "target_metric": target,
        "date_column": date_col,
        "dimensions": dimensions,
        "reasoning": plan.get("reasoning", "Selected from the profiled schema."),
        "available": {"numeric": numeric, "datetime": dates, "categorical": categorical},
        "definitions_used": [d["term"] for d in (definitions or {}).get("definitions", [])],
        "context_documents": [c["document_title"]
                              for c in (definitions or {}).get("context_passages", [])],
    }


def _guess_metric(question: str, columns: list[dict], numeric: list[str]) -> str | None:
    """Fallback when the LLM is mocked or returns something unusable."""
    q = question.lower()
    for col in numeric:
        if col.lower() in q:
            return col
    for col in columns:
        if col["inferred_type"] == "numeric" and col.get("semantic_label") == "measure":
            return col["name"]
    return numeric[0] if numeric else None


def _rank_dimensions(categorical: list[str], columns: list[dict]) -> list[str]:
    """Prefer low-cardinality columns - they segment cleanly."""
    lookup = {c["name"]: c for c in columns}

    def key(name: str):
        col = lookup.get(name, {})
        unique = col.get("unique_count") or 999
        priority = 0 if col.get("semantic_label") in {"geography", "product", "customer"} else 1
        return (priority, unique)

    return sorted([c for c in categorical if (lookup.get(c, {}).get("unique_count") or 0) <= 50], key=key)


def split_periods(df: pd.DataFrame, date_column: str) -> dict | None:
    """Default comparison: most recent third vs the period before it."""
    if not date_column or date_column not in df.columns:
        return None
    dates = parse_datetimes(df[date_column]).dropna()
    if len(dates) < 10:
        return None
    cutoff = dates.quantile(0.67)
    return {
        "current_start": str(cutoff.date()),
        "current_end": str(dates.max().date()),
        "previous_start": str(dates.min().date()),
        "previous_end": str(cutoff.date()),
        "method": "most recent third of the date range compared with everything before it",
    }