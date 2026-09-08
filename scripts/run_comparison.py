"""Four-way architecture comparison (proposal Sec.20).

    python scripts/generate_benchmark.py     # once
    python scripts/run_comparison.py         # no server needed

Runs Baselines A, B, C, the proposed system, and a set of ablations across every
benchmark scenario, then writes benchmarks/comparison.json and prints the table.

This runs in-process against the database rather than through the API, because
the baselines are architectures, not endpoints - there is no HTTP surface for
"a single agent without a critic".

Options:
    --systems baseline_a,proposed     limit which systems run
    --ablations                       also run the proposed-minus-X variants
    --repeats N                       run each combination N times (LLM output
                                      varies, so a single run is not evidence)
"""

from __future__ import annotations

import argparse
import json
import warnings
import statistics
import sys
import uuid
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal  # noqa: E402
from app.evaluation import baselines, metrics  # noqa: E402
from app.services import ingestion, profiling, rag  # noqa: E402

BENCH = ROOT / "benchmarks"

QUESTIONS = {
    "regional_decline": "Why did revenue decline?",
    "stock_shortage": "Why did revenue fall in recent months?",
    "price_increase": "Why did revenue drop?",
    "missing_evidence": "Why did revenue drop sharply?",
    "multi_source": "Why did revenue decline?",
    "changed_driver": "Why did revenue decline?",
}

ABLATION_VARIANTS = {
    "proposed_minus_critic": {"critic": False},
    "proposed_minus_verification": {"verification": False},
    "proposed_minus_rag": {"rag": False},
    "proposed_minus_evidence_gap": {"evidence_gap": False},
}


def load_scenario(db, truth: dict) -> uuid.UUID:
    """Ingest one benchmark scenario and return its dataset id."""
    path = BENCH / truth["file"]
    content = path.read_bytes()
    source = ingestion.register_file_source(db, path.name, content)
    # Prefixed so these are distinguishable from your own uploads in the UI,
    # and so scripts/reset_workspace.py can remove them without touching
    # anything you added by hand.
    dataset = ingestion.create_dataset_from_source(
        db, source, name=f"[eval] {path.stem}"
    )
    from app.models import DatasetVersion

    version = db.get(DatasetVersion, dataset.current_version_id)
    profiling.profile_version(db, version, ingestion.load_version(version))

    # multi-source scenarios ship a companion document that only RAG-capable
    # systems can use - that asymmetry is the point of the scenario
    doc = truth.get("document")
    if doc:
        text = doc.get("text")
        if not text and doc.get("file"):
            text = (BENCH / doc["file"]).read_text()
        if text:
            rag.index_document(db, title=doc["title"], text=text,
                               document_type=doc.get("document_type", "business_doc"))
    db.commit()
    return dataset.id


def run_one(db, system: str, dataset_id, question: str, truth: dict) -> dict:
    # The changed-driver scenario only means anything if there IS a previous
    # investigation to differ from, so one is run first and discarded.
    scope = truth.get("scope") or {}

    # The changed-driver scenario only means anything if an EARLIER period was
    # investigated first, so that run happens here and its result is discarded -
    # what is scored is whether the second run notices the driver differs.
    if truth.get("tests_driver_change") and system.startswith("proposed"):
        earlier = truth.get("earlier_scope") or {}
        baselines.run_proposed(db, dataset_id,
                               "Why did revenue decline in the earlier period?",
                               period_start=earlier.get("start"),
                               period_end=earlier.get("end"))
        db.commit()

    if system in baselines.SYSTEMS:
        if system == "proposed":
            return baselines.run_proposed(db, dataset_id, question,
                                          period_start=scope.get("start"),
                                          period_end=scope.get("end"))
        return baselines.SYSTEMS[system](db, dataset_id, question)
    return baselines.run_proposed(db, dataset_id, question,
                                  ablation=ABLATION_VARIANTS[system],
                                  period_start=scope.get("start"),
                                  period_end=scope.get("end"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="baseline_a,baseline_b,baseline_c,proposed")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    truth_file = BENCH / "ground_truth.json"
    if not truth_file.exists():
        sys.exit("Run scripts/generate_benchmark.py first.")
    truths = json.loads(truth_file.read_text())

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if args.ablations:
        systems += list(ABLATION_VARIANTS)

    if baselines.using_mock_llm():
        print("!" * 72)
        print("LLM_PROVIDER is 'mock'. Baselines A, B and C are driven by model")
        print("judgement, so their scores here test the plumbing only and must NOT")
        print("be quoted as an experimental result. Set a real provider in .env")
        print("before running the comparison for the report.")
        print("!" * 72 + "\n")

    db = SessionLocal()
    per_system: dict[str, list[dict]] = {s: [] for s in systems}
    detail = []

    try:
        for truth in truths:
            name = truth["scenario"]
            if truth.get("quality_benchmark"):
                # profiling/cleaning benchmark - scored by run_quality_evaluation.py
                continue
            question = QUESTIONS.get(name, "Why did the metric change?")
            print(f"\n=== {name} ===")

            for system in systems:
                scores = []
                for _ in range(args.repeats):
                    dataset_id = load_scenario(db, truth)   # fresh copy per run
                    try:
                        run = run_one(db, system, dataset_id, question, truth)
                        db.commit()
                    except Exception as exc:  # noqa: BLE001
                        db.rollback()
                        run = {"system": system, "answer": "", "findings": [],
                               "error": str(exc)}
                    scored = metrics.score_run(truth, run)
                    scored["scenario"] = name
                    scored["system"] = system
                    scores.append(scored)
                    detail.append(scored)

                per_system[system].extend(scores)
                hits = sum(bool(s["root_cause_rank_1"]) for s in scores)
                lead = (scores[0].get("lead_answer") or "")[:88]
                print(f"  {system:28s} {hits}/{len(scores)}  {lead}")
    finally:
        db.close()

    rows = [metrics.aggregate(s, per_system[s]) for s in systems if per_system[s]]
    print("\n" + "=" * 72)
    print("ARCHITECTURE COMPARISON (proposal Sec.20)\n")
    print(metrics.format_table(rows))

    output = {
        "llm_provider_was_mock": baselines.using_mock_llm(),
        "repeats": args.repeats,
        "scenarios": [t["scenario"] for t in truths if not t.get("quality_benchmark")],
        "summary": rows,
        "detail": detail,
    }
    (BENCH / "comparison.json").write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWritten to {BENCH / 'comparison.json'}")

    if args.repeats > 1:
        print("\nVariance across repeats (root cause @1):")
        for system in systems:
            per_run = [bool(s["root_cause_rank_1"]) for s in per_system[system]]
            if len(per_run) > 1:
                print(f"  {system:28s} mean {statistics.mean(per_run):.2f} "
                      f"over {len(per_run)} runs")


if __name__ == "__main__":
    main()