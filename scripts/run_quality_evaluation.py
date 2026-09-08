"""Data-quality, cleaning and internal-behaviour evaluation (proposal Sec.20).

    python scripts/generate_benchmark.py     # once
    python scripts/run_quality_evaluation.py # no server needed

Scores the layers underneath the conclusion:

  profiling   precision and recall against defects injected on purpose
  cleaning    were those defects repaired without destroying valid rows
  arithmetic  does the reported magnitude match the injected one
  hypotheses  relevance, resolution rate, verification accuracy
  statistics  was each test valid for the columns it ran on

Writes benchmarks/quality.json.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal  # noqa: E402
from app.evaluation import baselines, quality_metrics as qm  # noqa: E402
from app.models import DatasetVersion  # noqa: E402
from app.services import cleaning, ingestion, profiling  # noqa: E402

BENCH = ROOT / "benchmarks"

QUESTIONS = {
    "regional_decline": "Why did revenue decline?",
    "stock_shortage": "Why did revenue fall in recent months?",
    "price_increase": "Why did revenue drop?",
    "multi_source": "Why did revenue decline?",
    "changed_driver": "Why did revenue decline?",
}


def ingest(db, truth: dict):
    path = BENCH / truth["file"]
    source = ingestion.register_file_source(db, path.name, path.read_bytes())
    dataset = ingestion.create_dataset_from_source(
        db, source, name=f"[eval] {path.stem}"
    )
    version = db.get(DatasetVersion, dataset.current_version_id)
    profile = profiling.profile_version(db, version, ingestion.load_version(version))
    db.commit()
    return dataset, version, profile


# --------------------------------------------------------------------- #
def score_quality_scenario(db, truth: dict) -> dict:
    dataset, version, profile = ingest(db, truth)
    result = {"scenario": truth["scenario"]}
    result.update(qm.quality_detection(truth, profile))

    before = ingestion.load_version(version)
    plan = cleaning.propose_plan(before)
    # approve everything: the question is whether full cleaning is CORRECT,
    # not whether the approval gate works (that is covered by the unit tests)
    approved = [op["op_id"] for op in plan["operations"] if not op["safe"]]
    cleaned, log = cleaning.apply_plan(before, plan, approved)

    new_version = ingestion.new_version(db, dataset, cleaned, version,
                                        version_type="cleaned",
                                        cleaning_operations={"operations": log},
                                        created_by_agent="evaluation")
    after_profile = profiling.profile_version(db, new_version, cleaned)
    db.commit()

    result.update(qm.cleaning_correctness(truth, before, cleaned, after_profile))
    result["cleaning_operations_run"] = sum(1 for o in log if o.get("executed"))
    return result


def score_investigation_scenario(db, truth: dict) -> dict:
    dataset, version, profile = ingest(db, truth)
    scope = truth.get("scope") or {}

    if truth.get("tests_driver_change"):
        earlier = truth.get("earlier_scope") or {}
        baselines.run_proposed(db, dataset.id,
                               "Why did revenue decline in the earlier period?",
                               period_start=earlier.get("start"),
                               period_end=earlier.get("end"))
        db.commit()

    run = baselines.run_proposed(db, dataset.id,
                                 QUESTIONS.get(truth["scenario"],
                                               "Why did the metric change?"),
                                 period_start=scope.get("start"),
                                 period_end=scope.get("end"))
    db.commit()

    column_types = {c["name"]: c["inferred_type"] for c in profile.get("columns", [])}
    result = {"scenario": truth["scenario"]}
    result.update(qm.numerical_accuracy(truth, run))
    result.update(qm.hypothesis_quality(run))
    result.update(qm.test_appropriateness(run, column_types))
    return result


def _mean(rows: list[dict], key: str):
    values = [r[key] for r in rows
              if isinstance(r.get(key), (int, float)) and r.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else None


def main() -> None:
    truth_file = BENCH / "ground_truth.json"
    if not truth_file.exists():
        sys.exit("Run scripts/generate_benchmark.py first.")
    truths = json.loads(truth_file.read_text())

    db = SessionLocal()
    quality_rows, investigation_rows = [], []
    try:
        for truth in truths:
            name = truth["scenario"]
            try:
                if truth.get("quality_benchmark"):
                    row = score_quality_scenario(db, truth)
                    quality_rows.append(row)
                    print(f"\n=== {name} (profiling and cleaning) ===")
                    print(f"  detection   recall {row['quality_recall']}  "
                          f"precision {row['quality_precision']}  f1 {row['quality_f1']}")
                    if row["quality_missed"]:
                        print(f"  missed      {row['quality_missed']}")
                    if row["quality_spurious"]:
                        print(f"  spurious    {row['quality_spurious']}")
                    if row.get("left_unrepaired_by_design"):
                        print(f"  by design   left alone: "
                              f"{row['left_unrepaired_by_design']}")
                    print(f"  cleaning    resolved {row['defects_resolved']}/"
                          f"{row['defects_repairable']} repairable  "
                          f"retention {row['valid_data_retention']}  "
                          f"over-cleaned {row['rows_over_cleaned']} rows")
                elif name in QUESTIONS:
                    row = score_investigation_scenario(db, truth)
                    investigation_rows.append(row)
                    print(f"\n=== {name} (investigation internals) ===")
                    if row.get("numerical_scored"):
                        print(f"  arithmetic  expected {row['numerical_expected']}  "
                              f"reported {row['numerical_reported']}  "
                              f"correct {row['numerical_correct']}")
                    print(f"  hypotheses  relevance {row['hypothesis_relevance']}  "
                          f"resolved {row['hypothesis_resolution_rate']}  "
                          f"verification {row['verification_accuracy']}")
                    print(f"  statistics  {row['tests_appropriate']}/{row['tests_run']} "
                          f"appropriate")
                    if row.get("test_problems"):
                        print(f"              {row['test_problems']}")
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                print(f"\n=== {name} === ERROR: {exc}")

        feedback = qm.recommendation_feedback(db)
    finally:
        db.close()

    summary = {
        "quality_recall": _mean(quality_rows, "quality_recall"),
        "quality_precision": _mean(quality_rows, "quality_precision"),
        "defect_resolution_rate": _mean(quality_rows, "defect_resolution_rate"),
        "valid_data_retention": _mean(quality_rows, "valid_data_retention"),
        "numerical_accuracy": (
            round(sum(bool(r.get("numerical_correct")) for r in investigation_rows)
                  / len([r for r in investigation_rows if r.get("numerical_scored")]), 3)
            if any(r.get("numerical_scored") for r in investigation_rows) else None),
        "hypothesis_relevance": _mean(investigation_rows, "hypothesis_relevance"),
        "hypothesis_resolution_rate": _mean(investigation_rows, "hypothesis_resolution_rate"),
        "verification_accuracy": _mean(investigation_rows, "verification_accuracy"),
        "test_appropriateness": _mean(investigation_rows, "test_appropriateness"),
        **feedback,
    }

    print("\n" + "=" * 62)
    print("EVALUATION METRICS (proposal Sec.20)\n")
    for key, value in summary.items():
        print(f"  {key:28s} {value}")

    (BENCH / "quality.json").write_text(json.dumps(
        {"summary": summary, "quality": quality_rows,
         "investigations": investigation_rows}, indent=2, default=str))
    print(f"\nWritten to {BENCH / 'quality.json'}")


if __name__ == "__main__":
    main()