"""Baseline architectures for the experimental comparison (proposal Sec.20).

    Baseline A  LLM receives a dataset summary only. No tools.
    Baseline B  Single agent with Python/SQL tools. No RAG, no critic, no verifier.
    Baseline C  Baseline B plus retrieval.
    Proposed    The full multi-agent system (app/agents/orchestrator.py).

Each baseline is deliberately built as the honest simplest version of itself.
The point of the experiment is to find out what the extra machinery buys, so
weakening a baseline on purpose would make the result meaningless. Baselines B
and C therefore get the same tool layer and the same dataset the proposed
system uses - what they lack is hypothesis structure, criticism, mechanical
verification and the evidence-gap loop.

All four return the same shape, so `app/evaluation/metrics.py` can score them
identically.

Note on the mock LLM provider: baselines A, B and C are driven almost entirely
by model judgement, so with LLM_PROVIDER=mock they exercise the plumbing but
their scores mean nothing. A real provider is required before quoting any
comparison in a report; `run_comparison.py` prints a warning when the mock is
in use.
"""

from __future__ import annotations

import json
import time
import uuid

from sqlalchemy.orm import Session

from app.config import settings
from app.llm.client import llm
from app.models import Dataset, DatasetVersion
from app.services import ingestion, profiling, rag
from app.tools import registry

MAX_TOOL_STEPS = 8


def _summary(db: Session, dataset: Dataset) -> tuple[dict, object]:
    version = db.get(DatasetVersion, dataset.current_version_id)
    df = ingestion.load_version(version)
    profile = version.profile_summary or profiling.profile_version(db, version, df)
    compact = {
        "rows": profile["row_count"],
        "columns": [
            {"name": c["name"], "type": c["inferred_type"],
             "null_pct": c["null_percentage"], "unique": c["unique_count"],
             "statistics": {} if c.get("sensitive") else c["statistics"]}
            for c in profile["columns"]
        ],
    }
    return compact, df


def _result(system: str, question: str, answer: str, started: float,
            findings: list[dict] | None = None, tool_calls: int = 0,
            llm_calls: int = 0, prompt_chars: int = 0,
            asked_for_evidence: bool = False) -> dict:
    return {
        "system": system,
        "question": question,
        "answer": answer,
        "findings": findings or [],
        "tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "prompt_chars": prompt_chars,
        "asked_for_evidence": asked_for_evidence,
        "latency_seconds": round(time.perf_counter() - started, 2),
    }


# ===================================================================== #
# Baseline A - dataset summary only
# ===================================================================== #
A_SYSTEM = (
    "You are a data analyst. You are given a summary of a dataset and a business "
    "question. You have no ability to run code or query the data. Answer with the "
    "most likely cause. Return JSON with keys: cause (string), confidence "
    "(high|medium|low), reasoning (string)."
)


def run_baseline_a(db: Session, dataset_id: uuid.UUID, question: str) -> dict:
    started = time.perf_counter()
    dataset = db.get(Dataset, dataset_id)
    summary, _ = _summary(db, dataset)
    prompt = f"Question: {question}\n\nDataset summary:\n{json.dumps(summary, default=str)}"

    try:
        out = llm.complete_json(system=A_SYSTEM, prompt=prompt)
    except Exception as exc:  # noqa: BLE001
        out = {"cause": f"[error: {exc}]", "confidence": "low", "reasoning": ""}

    cause = str(out.get("cause", ""))
    return _result(
        "baseline_a", question, cause, started,
        # A has no tool run behind anything it says, so nothing is verifiable
        findings=[{"statement": cause, "finding_type": "driver",
                   "confidence": out.get("confidence", "medium"),
                   "verification_status": "failed_verification",
                   "tool_run_id": None}],
        llm_calls=1, prompt_chars=len(prompt) + len(A_SYSTEM),
    )


# ===================================================================== #
# Baseline B - single agent with tools
# ===================================================================== #
B_TOOLS = ["run_dataframe_code", "run_readonly_sql", "run_statistical_test",
           "inspect_dataset", "profile_columns"]
C_TOOLS = B_TOOLS + ["vector_search", "get_business_definition"]

AGENT_SYSTEM = (
    "You are a data analyst agent investigating a business question. You may call "
    "tools to inspect the data. Return JSON only.\n\n"
    "To call a tool: {\"action\": \"tool\", \"tool\": \"<name>\", \"params\": {...}}\n"
    "To finish:     {\"action\": \"answer\", \"cause\": \"...\", "
    "\"confidence\": \"high|medium|low\", \"evidence\": \"...\"}\n\n"
    "Base your answer on tool results, not on assumptions."
)


def _tool_catalogue(names: list[str]) -> str:
    return "\n".join(f"- {n}: {registry.TOOL_DESCRIPTIONS[n]}" for n in names)


def _run_single_agent(db: Session, dataset_id: uuid.UUID, question: str,
                      tools: list[str], system_name: str) -> dict:
    """The agent loop shared by baselines B and C."""
    started = time.perf_counter()
    dataset = db.get(Dataset, dataset_id)
    summary, _ = _summary(db, dataset)

    transcript: list[str] = []
    tool_calls = llm_calls = prompt_chars = 0
    last_run_id = None
    answer = None
    confidence = "medium"
    evidence = ""

    for _ in range(MAX_TOOL_STEPS):
        prompt = (
            f"Question: {question}\n"
            f"dataset_id: {dataset_id}\n\n"
            f"Available tools:\n{_tool_catalogue(tools)}\n\n"
            f"Dataset summary:\n{json.dumps(summary, default=str)}\n\n"
            + ("Results so far:\n" + "\n".join(transcript) if transcript else "")
            + "\n\nNext step?"
        )
        prompt_chars += len(prompt) + len(AGENT_SYSTEM)
        llm_calls += 1

        try:
            step = llm.complete_json(system=AGENT_SYSTEM, prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            answer = f"[error: {exc}]"
            break

        if not isinstance(step, dict) or step.get("action") == "answer":
            answer = str((step or {}).get("cause", ""))
            confidence = str((step or {}).get("confidence", "medium"))
            evidence = str((step or {}).get("evidence", ""))
            break

        name = step.get("tool")
        if name not in tools:
            transcript.append(f"Tool '{name}' is not available.")
            continue

        params = dict(step.get("params") or {})
        params.setdefault("dataset_id", str(dataset_id))
        result, run = registry.call_tool(db, name, params, agent_name=system_name)
        tool_calls += 1
        if run.status == "success":
            last_run_id = run.id
        transcript.append(f"{name}({params}) -> {json.dumps(result, default=str)[:900]}")

    if answer is None:
        answer = "No conclusion reached within the step budget."

    # A single agent states a cause without a separate verification pass, so the
    # claim is recorded as unverified even when a tool was used along the way.
    return _result(
        system_name, question, answer, started,
        findings=[{"statement": answer, "finding_type": "driver",
                   "confidence": confidence, "evidence_summary": evidence,
                   "verification_status": "pending",
                   "tool_run_id": str(last_run_id) if last_run_id else None}],
        tool_calls=tool_calls, llm_calls=llm_calls, prompt_chars=prompt_chars,
    )


def run_baseline_b(db: Session, dataset_id: uuid.UUID, question: str) -> dict:
    return _run_single_agent(db, dataset_id, question, B_TOOLS, "baseline_b")


# ===================================================================== #
# Baseline C - single agent + tools + RAG
# ===================================================================== #
def run_baseline_c(db: Session, dataset_id: uuid.UUID, question: str) -> dict:
    result = _run_single_agent(db, dataset_id, question, C_TOOLS, "baseline_c")
    # record what retrieval was available, so a null result is distinguishable
    # from retrieval never having been offered
    result["retrieval_available"] = True
    result["indexed_documents"] = len(rag.search(db, question, top_k=5))
    return result


# ===================================================================== #
# Proposed system
# ===================================================================== #
def run_proposed(db: Session, dataset_id: uuid.UUID, question: str,
                 ablation: dict | None = None,
                 period_start: str | None = None,
                 period_end: str | None = None) -> dict:
    from app.agents import orchestrator

    started = time.perf_counter()
    investigation = orchestrator.start_investigation(
        db, dataset_id, question, period_start=period_start, period_end=period_end)
    state = orchestrator.run(db, investigation, ablation=ablation)
    report = state.get("report") or {}

    from app.models import Hypothesis, ToolRun

    runs = (db.query(ToolRun)
            .filter(ToolRun.investigation_id == investigation.id).all())
    hypotheses = (db.query(Hypothesis)
                  .filter(Hypothesis.investigation_id == investigation.id).all())
    tool_calls = len(runs)

    name = "proposed"
    if ablation:
        disabled = sorted(k for k, v in ablation.items() if not v)
        if disabled:
            name = "proposed_minus_" + "_".join(disabled)

    return _result(
        name, question, report.get("executive_summary", ""), started,
        findings=report.get("findings", []),
        tool_calls=tool_calls,
        asked_for_evidence=bool(state.get("open_requests")),
    ) | {
        "investigation_id": state["investigation_id"],
        "status": state["status"],
        "unresolved_hypotheses": report.get("unresolved_hypotheses", []),
        "recommendations": report.get("recommendations", []),
        "forecasts": report.get("forecasts", []),
        "historical_comparison": report.get("historical_comparison", {}),
        "retrieval": report.get("retrieval", {}),
        # internals needed by app/evaluation/quality_metrics.py
        "hypotheses": [{"statement": h.statement, "status": h.status,
                        "confidence": h.confidence} for h in hypotheses],
        "verification": report.get("verification", []),
        "tool_runs": [{"tool_name": r.tool_name, "parameters": r.parameters,
                       "status": r.status} for r in runs],
        "profile": report.get("profile"),
    }


SYSTEMS = {
    "baseline_a": run_baseline_a,
    "baseline_b": run_baseline_b,
    "baseline_c": run_baseline_c,
    "proposed": run_proposed,
}


def using_mock_llm() -> bool:
    return settings.LLM_PROVIDER.lower() == "mock"
