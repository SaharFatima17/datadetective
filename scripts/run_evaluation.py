"""Phase 13 - Evaluation runner (proposal Sec.20).

Runs every benchmark scenario through the full system and scores it against the
known ground truth. This is the number that matters for Sec.24's central claim:
does the system rank the TRUE cause first?

    python scripts/generate_benchmark.py     # once
    python scripts/run_evaluation.py         # needs the API running

Writes benchmarks/results.json.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

QUESTIONS = {
    "regional_decline": "Why did revenue decline?",
    "stock_shortage": "Why did revenue fall in recent months?",
    "price_increase": "Why did revenue drop?",
    "missing_evidence": "Why did revenue drop sharply?",
}


# The API requires authentication (proposal Sec.19), so the harness creates a
# throwaway account for the run. Every scenario then belongs to one user, which
# also exercises the ownership path.
HEADERS: dict[str, str] = {}


def authenticate() -> None:
    email = f"eval-{uuid.uuid4().hex[:10]}@example.com"
    password = "evaluation-run-1"
    r = httpx.post(f"{BASE}/api/auth/register",
                   json={"email": email, "password": password, "full_name": "Evaluation"},
                   timeout=60)
    if r.status_code not in (201, 409):
        r.raise_for_status()
    r = httpx.post(f"{BASE}/api/auth/login",
                   json={"email": email, "password": password}, timeout=60)
    r.raise_for_status()
    HEADERS["Authorization"] = f"Bearer {r.json()['access_token']}"


def upload(path: Path) -> str:
    with path.open("rb") as fh:
        r = httpx.post(f"{BASE}/api/sources/upload", files={"file": (path.name, fh, "text/csv")},
                       headers=HEADERS, timeout=180)
    r.raise_for_status()
    return r.json()["dataset_id"]


def investigate(dataset_id: str, question: str) -> dict:
    r = httpx.post(f"{BASE}/api/investigations",
                   json={"dataset_id": dataset_id, "question": question},
                   headers=HEADERS, timeout=600)
    r.raise_for_status()
    return r.json()


def score(truth: dict, result: dict) -> dict:
    report = result.get("report") or {}
    findings = report.get("findings", [])
    statements = " ".join(f.get("statement", "") for f in findings).lower()

    driver_col = (truth.get("true_driver_column") or "").lower()
    driver_val = (truth.get("true_driver_value") or "")
    driver_val = driver_val.lower() if driver_val else ""

    # the missing-evidence scenario is scored on behaviour, not on a cause
    if truth["scenario"] == "missing_evidence":
        asked = bool(result.get("open_requests")) or bool(report.get("unresolved_hypotheses"))
        # A measurement ("revenue fell 37%") is an observation, not a claimed cause -
        # stating it is correct behaviour. Only a driver or association claim counts
        # as asserting an explanation the data cannot support.
        asserted = any(
            f.get("finding_type") in {"driver", "association"} and f.get("confidence") == "high"
            for f in findings
        )
        return {
            "scenario": truth["scenario"],
            "criterion": "asks instead of asserting",
            "asked_for_evidence": asked,
            "asserted_high_confidence_cause": asserted,
            "passed": asked and not asserted,
        }

    # score against the strongest EXPLANATORY finding, not a plain measurement
    causal = [f for f in findings if f.get("finding_type") in {"driver", "association"}]
    lead = causal[0] if causal else (findings[0] if findings else None)
    lead_text = (lead or {}).get("statement", "").lower()
    # Guard the empty case: `"" in text` is always True, which would make a
    # scenario with no declared driver pass trivially.
    top_hit = bool(lead) and (
        bool(driver_col and driver_col in lead_text)
        or bool(driver_val and driver_val in lead_text)
    )
    any_hit = (bool(driver_col and driver_col in statements)
               or bool(driver_val and driver_val in statements))

    return {
        "scenario": truth["scenario"],
        "criterion": "true cause ranked first",
        "true_cause": truth["true_cause"],
        "top_finding": lead.get("statement") if lead else None,
        "found_at_rank_1": top_hit,
        "found_anywhere": bool(any_hit),
        "findings_count": len(findings),
        "all_verified": all(f.get("verification_status") == "verified" for f in findings)
        if findings else False,
        "passed": top_hit,
    }


def main() -> None:
    truth_file = BENCH / "ground_truth.json"
    if not truth_file.exists():
        sys.exit("Run scripts/generate_benchmark.py first.")

    try:
        httpx.get(f"{BASE}/health", timeout=10).raise_for_status()
    except Exception:
        sys.exit(f"API is not reachable at {BASE}. Start it with: uvicorn app.main:app")

    authenticate()
    truths = json.loads(truth_file.read_text())
    results = []

    for truth in truths:
        name = truth["scenario"]
        if truth.get("quality_benchmark"):
            # scored by scripts/run_quality_evaluation.py instead
            continue
        print(f"\n=== {name} ===")
        try:
            dataset_id = upload(BENCH / truth["file"])
            started = time.perf_counter()
            outcome = investigate(dataset_id, QUESTIONS.get(name, "Why did the metric change?"))
            elapsed = round(time.perf_counter() - started, 1)
            scored = score(truth, outcome)
            scored["latency_seconds"] = elapsed
            results.append(scored)
            print(f"  passed: {scored['passed']}   ({elapsed}s)")
            if scored.get("top_finding"):
                print(f"  top finding: {scored['top_finding'][:110]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            results.append({"scenario": name, "passed": False, "error": str(exc)})

    passed = sum(1 for r in results if r.get("passed"))
    summary = {
        "total_scenarios": len(results),
        "passed": passed,
        "accuracy": round(passed / len(results), 3) if results else 0,
        "results": results,
    }
    (BENCH / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 50}")
    print(f"Root-cause accuracy: {passed}/{len(results)} = {summary['accuracy']:.0%}")
    print(f"Written to {BENCH / 'results.json'}")


if __name__ == "__main__":
    main()
