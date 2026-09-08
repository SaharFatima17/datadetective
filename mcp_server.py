"""Phase 8 - MCP server (proposal Sec.9).

Exposes the SAME registry the agents already use over the Model Context Protocol,
so an external MCP client (Claude Desktop, an IDE, another agent framework) gets
exactly the tools DataDetective's own agents have - no second implementation.

Run it standalone:

    python mcp_server.py

The safety posture from Sec.19 is inherited automatically: no shell access, no
writes, SQL is SELECT-only and dataframe operations are whitelisted.

The MCP SDK changed its server API between 1.x and 2.x, so both are supported.
"""

from __future__ import annotations

import json
from typing import Any

from app.database import SessionLocal
from app.tools import registry

SERVER_NAME = "datadetective"


def _invoke(tool_name: str, arguments: dict) -> str:
    """Shared implementation - identical behaviour on either SDK version."""
    db = SessionLocal()
    try:
        result, run = registry.call_tool(db, tool_name, arguments or {}, agent_name="mcp")
        db.commit()
        payload = {"tool_run_id": str(run.id), "status": run.status, "result": result}
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        payload = {"status": "error", "error": str(exc)}
    finally:
        db.close()
    return json.dumps(payload, indent=2, default=str)


# --------------------------------------------------------------------- #
# Tool signatures. Keyword names match app/tools/registry.py exactly.
# --------------------------------------------------------------------- #
def inspect_dataset(dataset_id: str, version_id: str | None = None) -> str:
    """Return schema, shape and a sample of rows for a dataset."""
    return _invoke("inspect_dataset", {"dataset_id": dataset_id, "version_id": version_id})


def profile_columns(dataset_id: str, version_id: str | None = None) -> str:
    """Compute quality and distribution statistics for every column."""
    return _invoke("profile_columns", {"dataset_id": dataset_id, "version_id": version_id})


def run_dataframe_code(dataset_id: str, operation: str,
                       params: dict[str, Any] | None = None) -> str:
    """Run one whitelisted dataframe operation: describe, value_counts,
    groupby_aggregate, correlation, time_series, period_contribution."""
    return _invoke("run_dataframe_code",
                   {"dataset_id": dataset_id, "operation": operation, "params": params or {}})


def run_readonly_sql(dataset_id: str, query: str) -> str:
    """Run a validated SELECT query against the dataset. The table is named `data`."""
    return _invoke("run_readonly_sql", {"dataset_id": dataset_id, "query": query})


def run_statistical_test(dataset_id: str, test: str,
                         params: dict[str, Any] | None = None) -> str:
    """Run ttest, chi_square, correlation or linear_regression with effect sizes."""
    return _invoke("run_statistical_test",
                   {"dataset_id": dataset_id, "test": test, "params": params or {}})


def render_chart(dataset_id: str, spec: dict[str, Any]) -> str:
    """Render a bar, line, hist, scatter or forecast chart to PNG."""
    return _invoke("render_chart", {"dataset_id": dataset_id, "spec": spec})


def run_forecast(dataset_id: str, date_column: str, metric: str,
                 horizon: int = 3, freq: str = "M") -> str:
    """Fit and backtest a time-series model. Returns point forecasts with
    confidence intervals, or an explanation of why the forecast was withheld."""
    return _invoke("run_forecast", {"dataset_id": dataset_id, "date_column": date_column,
                                    "metric": metric, "horizon": horizon, "freq": freq})


def vector_search(query: str, top_k: int = 5, document_type: str | None = None) -> str:
    """Semantic search across indexed documents and past investigations."""
    return _invoke("vector_search",
                   {"query": query, "top_k": top_k, "document_type": document_type})


def get_business_definition(term: str) -> str:
    """Retrieve a KPI or data-dictionary definition from the knowledge base."""
    return _invoke("get_business_definition", {"term": term})


def list_sources() -> str:
    """List every registered data source."""
    return _invoke("list_sources", {})




def ingest_source(filename: str, content_base64: str) -> str:
    """Register a base64-encoded file, preserving the original and its provenance.
    Tabular files become datasets; documents must then be passed to extract_document."""
    return _invoke("ingest_source",
                   {"filename": filename, "content_base64": content_base64})


def extract_document(source_id: str, document_type: str = "business_doc") -> str:
    """Extract text from a registered document and index it for retrieval."""
    return _invoke("extract_document",
                   {"source_id": source_id, "document_type": document_type})


def fetch_url(url: str, index_for_retrieval: bool = True) -> str:
    """Retrieve permitted web content and store a snapshot with provenance."""
    return _invoke("fetch_url", {"url": url, "index_for_retrieval": index_for_retrieval})


def apply_cleaning_plan(dataset_id: str, approved_op_ids: list | None = None) -> str:
    """Apply a cleaning plan, creating a NEW dataset version. Safe operations run
    automatically; destructive ones only when their op_id is in approved_op_ids."""
    return _invoke("apply_cleaning_plan",
                   {"dataset_id": dataset_id, "approved_op_ids": approved_op_ids or []})


def save_finding(investigation_id: str, statement: str,
                 evidence_summary: str | None = None, tool_run_id: str | None = None,
                 finding_type: str = "measurement", confidence: str = "medium") -> str:
    """Persist a finding. finding_type is measurement, association or driver.
    A finding with no tool_run_id cannot pass verification."""
    return _invoke("save_finding", {
        "investigation_id": investigation_id, "statement": statement,
        "evidence_summary": evidence_summary, "tool_run_id": tool_run_id,
        "finding_type": finding_type, "confidence": confidence})


def generate_recommendations(investigation_id: str) -> str:
    """Turn verified driver findings into ranked, evidence-linked actions."""
    return _invoke("generate_recommendations", {"investigation_id": investigation_id})


def save_recommendation(investigation_id: str, action: str,
                        finding_id: str | None = None,
                        rationale: str | None = None) -> str:
    """Persist a recommendation. Refused if its finding is not verified."""
    return _invoke("save_recommendation", {
        "investigation_id": investigation_id, "action": action,
        "finding_id": finding_id, "rationale": rationale})


def save_report(investigation_id: str, executive_summary: str,
                title: str | None = None) -> str:
    """Persist a versioned investigation report and index it for later retrieval."""
    return _invoke("save_report", {"investigation_id": investigation_id,
                                   "executive_summary": executive_summary,
                                   "title": title})


def request_missing_evidence(investigation_id: str, question: str,
                             request_type: str = "clarification",
                             reason: str | None = None) -> str:
    """Raise a user-facing data request and pause the investigation."""
    return _invoke("request_missing_evidence", {
        "investigation_id": investigation_id, "question": question,
        "request_type": request_type, "reason": reason})


def resume_investigation(investigation_id: str, request_id: str, response: str) -> str:
    """Validate a user response and resume a paused investigation."""
    return _invoke("resume_investigation", {"investigation_id": investigation_id,
                                            "request_id": request_id,
                                            "response": response})


def compare_investigations(current_id: str, previous_ids: list | None = None) -> str:
    """Compare drivers and findings against previous investigations on the same
    dataset. Reports whether the driver has changed since last time."""
    return _invoke("compare_investigations",
                   {"current_id": current_id, "previous_ids": previous_ids})


def detect_drift(dataset_id: str, from_version: str | None = None,
                 to_version: str | None = None) -> str:
    """Compare two dataset versions for distribution drift and schema changes."""
    return _invoke("detect_drift", {"dataset_id": dataset_id,
                                    "from_version": from_version,
                                    "to_version": to_version})


EXPOSED = [
    # analysis
    inspect_dataset, profile_columns, run_dataframe_code, run_readonly_sql,
    run_statistical_test, render_chart, run_forecast,
    # retrieval
    vector_search, get_business_definition,
    # ingestion
    list_sources, ingest_source, extract_document, fetch_url,
    # data preparation
    apply_cleaning_plan, detect_drift,
    # persistence
    save_finding, generate_recommendations, save_recommendation, save_report,
    # investigation state
    request_missing_evidence, resume_investigation, compare_investigations,
]


# --------------------------------------------------------------------- #
def build_server():
    """Returns a runnable server on whichever MCP SDK version is installed."""
    try:  # MCP SDK 2.x
        from mcp.server.mcpserver import MCPServer

        server = MCPServer(SERVER_NAME, instructions=__doc__)
        for fn in EXPOSED:
            server.add_tool(fn, name=fn.__name__,
                            description=(fn.__doc__ or "").strip())
        return ("v2", server)

    except ImportError:  # MCP SDK 1.x
        from mcp.server import Server
        from mcp.types import TextContent, Tool

        server = Server(SERVER_NAME)
        by_name = {fn.__name__: fn for fn in EXPOSED}

        @server.list_tools()
        async def _list_tools() -> list[Tool]:
            return [
                Tool(
                    name=name,
                    description=registry.TOOL_DESCRIPTIONS.get(
                        name, (fn.__doc__ or "").strip()
                    ),
                    inputSchema=_schema_for(name),
                )
                for name, fn in by_name.items()
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
            return [TextContent(type="text", text=_invoke(name, arguments))]

        return ("v1", server)


def _schema_for(name: str) -> dict:
    """JSON schema for the 1.x path, derived from the function signature."""
    import inspect

    fn = {f.__name__: f for f in EXPOSED}[name]
    props, required = {}, []
    type_map = {str: "string", int: "integer", float: "number",
                bool: "boolean", dict: "object"}

    for pname, param in inspect.signature(fn).parameters.items():
        annotation = param.annotation
        base = getattr(annotation, "__origin__", None)
        if base is None and annotation in type_map:
            json_type = type_map[annotation]
        elif "dict" in str(annotation):
            json_type = "object"
        elif "int" in str(annotation):
            json_type = "integer"
        else:
            json_type = "string"
        props[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {"type": "object", "properties": props, "required": required}


def main() -> None:
    version, server = build_server()
    if version == "v2":
        server.run()
    else:
        import asyncio

        from mcp.server.stdio import stdio_server

        async def _run():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        asyncio.run(_run())


if __name__ == "__main__":
    main()
