"""Pack an investigation's whole evidence chain into one file.

Every claim this system makes points at a calculation that produced it, and the
interface shows that chain on screen. On screen is not the same as in someone
else's hands: a reader who wants to check the work has to sit at this machine,
with this database running, and click through.

A bundle is the same chain as a file. It holds the report, the charts, and —
the part that matters — every tool call with its parameters, its result
checksum and the finding it supports. Someone can open it anywhere and follow a
number back to the arithmetic that made it, without this system running at all.

Deliberately not a snapshot of the database. The bundle carries what was
recorded at the time of the investigation, so it stays true even after the
dataset is cleaned again or the documents change.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import (
    Chart,
    Dataset,
    DatasetVersion,
    Finding,
    Investigation,
    Report,
    ToolRun,
)

HTML_PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>{question} — evidence</title>
<style>
  :root {{
    --ink: #16202b; --muted: #5d6b7e; --line: #dfe5ec;
    --teal: #0a8f73; --teal-bg: #e9f6f2; --amber: #b7791f; --slate: #64748b;
    --page: #f3f5f8; --card: #ffffff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 40px 20px; background: var(--page); color: var(--ink);
    font: 16px/1.65 -apple-system, "Segoe UI", system-ui, sans-serif;
  }}
  main {{ max-width: 820px; margin: 0 auto; }}
  .sheet {{
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 38px 42px; box-shadow: 0 10px 30px -22px rgba(14,22,32,.5);
  }}
  .kicker {{
    font: 600 11.5px/1 ui-monospace, "SF Mono", Menlo, monospace;
    letter-spacing: .09em; text-transform: uppercase; color: var(--muted);
  }}
  h1 {{ font-size: 27px; line-height: 1.25; margin: 10px 0 6px; }}
  h2 {{
    font-size: 17px; margin: 34px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 11px;
  }}
  h2::before {{
    content: ""; width: 4px; height: 15px; border-radius: 3px; background: var(--teal);
  }}
  .lede {{
    background: var(--teal-bg); border-left: 4px solid var(--teal);
    border-radius: 10px; padding: 16px 20px; margin: 22px 0 0; font-size: 16px;
  }}
  .meta {{ display: flex; flex-wrap: wrap; gap: 22px; font-size: 13px; color: var(--muted); }}
  .meta b {{ color: var(--ink); font-weight: 600; }}
  .finding {{ border-left: 3px solid var(--line); padding: 2px 0 2px 16px; margin: 18px 0; }}
  .finding.driver {{ border-color: var(--teal); }}
  .finding.association {{ border-color: var(--amber); }}
  .finding.measurement {{ border-color: var(--slate); }}
  .tag {{
    font: 600 10.5px/1 ui-monospace, Menlo, monospace; letter-spacing: .07em;
    text-transform: uppercase; color: var(--muted);
  }}
  .claim {{ font-size: 16.5px; margin: 5px 0 8px; }}
  .how {{ font-size: 13.5px; color: var(--muted); }}
  .how code {{ font-size: 12.5px; }}
  .caveat {{ font-size: 13.5px; font-style: italic; color: var(--muted); margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 6px; }}
  th {{ text-align: left; font-weight: 600; color: var(--muted); padding: 7px 8px; }}
  td {{ padding: 7px 8px; border-top: 1px solid var(--line); vertical-align: top; }}
  td.num {{ text-align: right; font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; }}
  img {{ max-width: 100%; border: 1px solid var(--line); border-radius: 10px; margin-top: 8px; }}
  .note {{ font-size: 13px; color: var(--muted); }}
  .files {{ font-size: 13.5px; }}
  .files code {{
    background: var(--page); padding: 2px 6px; border-radius: 5px;
    font-family: ui-monospace, Menlo, monospace; font-size: 12.5px;
  }}
  .limits li {{ font-size: 13.5px; color: var(--muted); margin-bottom: 6px; }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .sheet {{ border: 0; box-shadow: none; padding: 0; }}
  }}
</style>
<main><div class="sheet">
  <p class="kicker">Evidence bundle</p>
  <h1>{question}</h1>
  <div class="meta">
    <span>Dataset <b>{dataset}</b></span>
    <span>Rows <b>{rows}</b></span>
    <span>Findings <b>{n_findings}</b></span>
    <span>Calculations <b>{n_runs}</b></span>
    <span>Generated <b>{generated}</b></span>
  </div>

  <div class="lede">{summary}</div>

  {warning_html}
  {charts_html}

  <h2>What was found</h2>
  {findings_html}

  {extra_html}

  <h2>Every calculation behind these numbers</h2>
  <p class="note">
    Each row is one calculation the system ran. The checksum is a fingerprint of
    what it returned: run the same tool with the same parameters on the same
    dataset and the fingerprint should match. That is what makes a number here
    checkable rather than merely stated.
  </p>
  <table>
    <tr><th>Tool</th><th>Parameters</th><th>Status</th><th>Checksum</th></tr>
    {runs_html}
  </table>

  <h2>Reading the rest of this folder</h2>
  <p class="files">
    <code>tool_runs.json</code> — the table above, in full, including every
    result.<br>
    <code>findings.json</code> — each finding with the id of the calculation
    that produced it.<br>
    <code>report.md</code> — the report as plain text.<br>
    <code>charts/</code> — the figures.
  </p>

  <h2>What this bundle does not claim</h2>
  <ul class="limits">
    <li>A checksum proves a calculation ran and what it returned. It does not
        prove the question was the right one to ask, or that the data is a
        faithful record of what happened.</li>
    <li>Findings marked <b>association</b> are correlations. They are not
        evidence of cause, and this system does not present them as such.</li>
    {limits_html}
  </ul>
</div></main>
</html>
"""

README = """DataDetective — evidence bundle
{generated}

Investigation : {question}
Dataset       : {dataset}
Rows          : {rows}
Checksum      : {checksum}

WHAT IS IN HERE

  report.md          the report as written, with findings, forecast,
                     recommendations and the limitations that apply
  findings.json      each finding, its type, and the id of the tool run
                     that produced it
  tool_runs.json     every calculation performed: the tool, the exact
                     parameters, how long it took, and a checksum of the
                     result it returned
  charts/            figures referenced by the report

HOW TO CHECK A NUMBER

  1. Find the claim in report.md.
  2. Look it up in findings.json and note its tool_run_id.
  3. Find that id in tool_runs.json. The params are the exact inputs the
     calculation used.
  4. Run the same operation on the same dataset version. The result
     checksum should match the one recorded here.

WHAT THIS BUNDLE DOES NOT CLAIM

  A checksum proves a calculation was performed and what it returned. It
  does not prove the question was the right one to ask, or that the data
  is a faithful record of what happened. Those remain matters of
  judgement, and the report states its own limitations for that reason.

  Findings marked "association" are correlations. They are not evidence of
  cause and the system does not present them as such.
"""


def build(db: Session, investigation: Investigation) -> tuple[bytes, str]:
    """Return the zip bytes and a filename for one investigation."""
    report = (
        db.query(Report)
        .filter(Report.investigation_id == investigation.id)
        .order_by(Report.version_number.desc())
        .first()
    )
    findings = (
        db.query(Finding)
        .filter(Finding.investigation_id == investigation.id)
        .order_by(Finding.created_at)
        .all()
    )
    runs = (
        db.query(ToolRun)
        .filter(ToolRun.investigation_id == investigation.id)
        .order_by(ToolRun.created_at)
        .all()
    )
    dataset = db.get(Dataset, investigation.dataset_id)
    version = db.get(DatasetVersion, investigation.version_id)
    charts = (
        db.query(Chart)
        .filter(Chart.investigation_id == investigation.id)
        .all()
    )

    profile = (version.profile_summary or {}) if version else {}
    buffer = io.BytesIO()

    chart_files = [(c, Path(c.storage_path or "")) for c in charts]
    chart_files = [(c, path) for c, path in chart_files if path.exists()]

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        # An index page first. A folder of JSON is readable by whoever wrote
        # the system and by nobody else; the people who most need to check a
        # claim — a manager, a supervisor, an examiner — open a folder, see
        # `tool_runs.json`, and close it again. This opens in any browser and
        # says the same thing in a form they can read.
        bundle.writestr("index.html", _index_page(
            investigation, report, findings, runs, dataset, version, chart_files))

        bundle.writestr("README.txt", README.format(
            generated=datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC"),
            question=investigation.question,
            dataset=dataset.name if dataset else "unknown",
            rows=profile.get("row_count", "unknown"),
            checksum=(version.checksum_sha256 if version else "unknown"),
        ))

        if report:
            bundle.writestr("report.md", _report_markdown(investigation, report))
            bundle.writestr("report.json", _dump({
                "question": investigation.question,
                "version": report.version_number,
                "executive_summary": report.executive_summary,
                **(report.content or {}),
            }))

        bundle.writestr("findings.json", _dump([
            {
                "id": str(f.id),
                "type": f.finding_type,
                "statement": f.statement,
                "confidence": f.confidence,
                "verification_status": f.verification_status,
                "evidence_summary": f.evidence_summary,
                "caveats": f.caveats,
                # the link that makes the rest of this bundle useful
                "tool_run_id": str(f.tool_run_id) if f.tool_run_id else None,
            }
            for f in findings
        ]))

        bundle.writestr("tool_runs.json", _dump([
            {
                "id": str(r.id),
                "tool": r.tool_name,
                "agent": r.agent_name,
                "params": r.parameters,
                "status": r.status,
                "error": r.error_message,
                "duration_ms": r.duration_ms,
                "result_checksum": r.result_checksum,
                "created_at": r.created_at,
            }
            for r in runs
        ]))

        for _, path in chart_files:
            bundle.write(path, f"charts/{path.name}")

    stem = "".join(c if c.isalnum() else "-" for c in investigation.question.lower())
    stem = "-".join(filter(None, stem.split("-")))[:60] or "investigation"
    return buffer.getvalue(), f"evidence-{stem}.zip"


def _index_page(investigation, report, findings, runs, dataset, version,
                chart_files) -> str:
    """The bundle's front door, written for someone who did not build this."""
    import html

    content = (report.content or {}) if report else {}
    profile = (version.profile_summary or {}) if version else {}
    esc = html.escape

    # findings, each with the calculation that produced it
    blocks = []
    for f in findings:
        kind = f.finding_type or "measurement"
        label = {"driver": "The explanation",
                 "association": "Correlation — not a cause",
                 "measurement": "A measurement — not an explanation"}.get(kind, kind)
        parts = [f'<div class="finding {esc(kind)}">',
                 f'<p class="tag">{esc(label)}</p>',
                 f'<p class="claim">{esc(f.statement or "")}</p>']
        if f.evidence_summary:
            parts.append(f'<p class="how">How this was worked out: '
                         f'{esc(f.evidence_summary)}</p>')
        if f.caveats:
            parts.append(f'<p class="caveat">{esc(f.caveats)}</p>')
        parts.append("</div>")
        blocks.append("".join(parts))
    findings_html = "".join(blocks) or (
        '<p class="note">No explanation was established. The change was '
        'measured, but nothing in the data accounted for it — so none was '
        'named.</p>')

    rows = []
    for r in runs:
        params = ", ".join(f"{k}={v}" for k, v in (r.parameters or {}).items()
                           if k not in {"dataset_id", "version_id"})
        rows.append(
            f"<tr><td>{esc(r.tool_name or '')}</td>"
            f"<td class=\"how\">{esc(params[:150])}</td>"
            f"<td>{esc(r.status or '')}</td>"
            f"<td class=\"num\">{esc((r.result_checksum or '')[:12])}</td></tr>")

    warning = content.get("mix_warning")
    warning_html = ""
    if warning:
        warning_html = (
            '<div class="lede" style="background:#fdf6e7;border-color:#b7791f">'
            '<p class="tag" style="color:#b7791f">Read the total with care</p>'
            f'<p>{esc(warning["note"])}</p></div>')

    charts_html = "".join(
        f'<h2>{esc(c.title or "Chart")}</h2>'
        f'<img src="charts/{esc(path.name)}" alt="{esc(c.title or "chart")}">'
        for c, path in chart_files)

    extra = []
    forecasts = content.get("forecasts") or []
    if forecasts:
        fc = forecasts[0]
        # The accuracy figures live under `backtest`, not on the forecast
        # itself. Read from the wrong place they come out as "None%", which
        # reads as a broken page rather than a missing number.
        bt = fc.get("backtest") or {}
        if bt.get("mape") is not None:
            extra.append(
                "<h2>What happens next</h2>"
                f'<p>The projection uses a {esc(str(fc.get("model")))} model. On '
                f'periods it had not seen, it was off by {bt["mape"]}% on '
                f'average, against {bt.get("naive_mape")}% for simply assuming '
                "next month matches last month. A projection that cannot beat "
                "that simple guess is not shown at all.</p>")
        else:
            extra.append(
                "<h2>What happens next</h2>"
                "<p>No projection is shown: it did not pass validation against "
                "periods held back from it.</p>")

    recs = content.get("recommendations") or []
    if recs:
        items = "".join(
            f"<li>{esc(r.get('action', ''))}</li>" for r in recs)
        extra.append(f"<h2>What to do</h2><ul>{items}</ul>"
                     '<p class="note">Impact figures are scenario estimates, '
                     "not measurements: they state what recovery would follow "
                     "if the action closed the stated share of the gap.</p>")

    unresolved = content.get("unresolved_hypotheses") or []
    if unresolved:
        items = "".join(f"<li>{esc(h)}</li>" for h in unresolved)
        extra.append("<h2>What could not be settled</h2>"
                     '<p class="note">Raised during the investigation and not '
                     "testable with the data available. Listed rather than "
                     "quietly dropped, so the limits of the answer are "
                     f"visible.</p><ul class=\"limits\">{items}</ul>")

    concerns = (content.get("critique") or {}).get("concerns") or []
    if concerns:
        items = "".join(f"<li>{esc(c)}</li>" for c in concerns)
        extra.append("<h2>What the reviewer objected to</h2>"
                     f'<ul class="limits">{items}</ul>')

    limits = content.get("limitations") or []
    limits_html = "".join(f"<li>{esc(l)}</li>" for l in limits[:4])

    return HTML_PAGE.format(
        question=esc(investigation.question),
        dataset=esc(dataset.name if dataset else "unknown"),
        rows=profile.get("row_count", "unknown"),
        n_findings=len(findings),
        n_runs=len(runs),
        generated=datetime.now(timezone.utc).strftime("%d %B %Y"),
        summary=esc((report.executive_summary if report else "") or ""),
        warning_html=warning_html,
        charts_html=charts_html,
        findings_html=findings_html,
        extra_html="".join(extra),
        runs_html="".join(rows),
        limits_html=limits_html,
    )


def _dump(payload) -> str:
    return json.dumps(payload, indent=2, default=str, ensure_ascii=False)


def _report_markdown(investigation: Investigation, report: Report) -> str:
    """The report as plain text, readable without this application."""
    content = report.content or {}
    out = [
        f"# {investigation.question}",
        "",
        f"Report version {report.version_number} · "
        f"{report.created_at:%d %B %Y}",
        "",
        report.executive_summary or "",
        "",
    ]

    warning = content.get("mix_warning")
    if warning:
        out += ["## Read the total with care", "", warning["note"], ""]

    findings = content.get("findings") or []
    if findings:
        out += ["## Findings", ""]
        for i, f in enumerate(findings, 1):
            out += [
                f"### {i}. {f.get('finding_type', '').upper()}",
                "",
                f.get("statement", ""),
                "",
                f"*Evidence:* {f.get('evidence_summary', '')}",
                "",
            ]
            if f.get("caveats"):
                out += [f"*What it does not say:* {f['caveats']}", ""]

    forecasts = content.get("forecasts") or []
    for f in forecasts:
        bt = f.get("backtest") or {}
        out += [
            "## Outlook", "",
            f"{f.get('model')} model, backtest error {bt.get('mape')}% against a "
            f"naive baseline of {bt.get('naive_mape')}%. "
            f"Reliability: {f.get('reliability')}.",
            "",
        ]

    recs = content.get("recommendations") or []
    if recs:
        out += ["## Recommended actions", ""]
        for i, r in enumerate(recs, 1):
            out += [f"{i}. {r.get('action', '')}", ""]
            if r.get("impact_low") is not None:
                out += [
                    f"   Estimated recovery {r['impact_low']} to "
                    f"{r['impact_high']} — a scenario estimate, not a "
                    f"measurement.", "",
                ]

    unresolved = content.get("unresolved_hypotheses") or []
    if unresolved:
        out += ["## Left unresolved", "",
                "Raised but not testable with the data available. Listed "
                "rather than dropped, so the limits of the answer are visible.",
                ""]
        out += [f"- {h}" for h in unresolved] + [""]

    concerns = (content.get("critique") or {}).get("concerns") or []
    if concerns:
        out += ["## Reviewer concerns", ""] + [f"- {c}" for c in concerns] + [""]

    consulted = (content.get("retrieval") or {}).get("context_documents") or []
    if consulted:
        out += ["## Background consulted", ""] + [f"- {d}" for d in consulted] + [""]

    limitations = content.get("limitations") or []
    if limitations:
        out += ["## Limitations", ""] + [f"- {l}" for l in limitations] + [""]

    return "\n".join(out)