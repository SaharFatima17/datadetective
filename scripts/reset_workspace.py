"""Remove data the evaluation scripts created.

Each comparison run gives every system a fresh copy of every scenario, on
purpose — otherwise one system's investigations would appear in the next one's
history and the comparison would be meaningless. The cost is that a single
`run_comparison.py --ablations` leaves 48 datasets behind.

    python scripts/reset_workspace.py                 # show what is there
    python scripts/reset_workspace.py --evaluation    # remove only [eval] data
    python scripts/reset_workspace.py --all           # remove everything

Nothing is deleted without a flag, and `--all` asks before it runs. Deletion
here is permanent: it removes the database rows AND the stored files, which is
the point — a soft delete would leave the disk full and the counts wrong.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import (  # noqa: E402
    DataSource,
    Dataset,
    Document,
    Investigation,
    Report,
)

EVAL_PREFIX = "[eval]"


def summarise(db) -> dict:
    datasets = db.query(Dataset).all()
    evaluation = [d for d in datasets if d.name.startswith(EVAL_PREFIX)]
    return {
        "datasets": len(datasets),
        "evaluation_datasets": len(evaluation),
        "your_datasets": len(datasets) - len(evaluation),
        "investigations": db.query(Investigation).count(),
        "reports": db.query(Report).count(),
        "sources": db.query(DataSource).count(),
        "documents": db.query(Document).count(),
    }


def remove(db, datasets: list[Dataset]) -> int:
    """Delete datasets, their investigations, their sources and their files."""
    if not datasets:
        return 0

    dataset_ids = [d.id for d in datasets]
    source_ids = [d.source_id for d in datasets if d.source_id]

    # Investigations first: findings, hypotheses, tool runs and reports all
    # cascade from them, so removing the investigation clears the rest.
    investigations = (
        db.query(Investigation).filter(Investigation.dataset_id.in_(dataset_ids)).all()
    )
    investigation_ids = {str(inv.id) for inv in investigations}

    # Each finished investigation also indexes its report into the knowledge
    # base, so later investigations can compare against it. Those entries do
    # not cascade, and leaving them behind would let a deleted run keep
    # influencing future retrieval.
    for doc in db.query(Document).filter(Document.document_type == "past_report").all():
        if str((doc.document_metadata or {}).get("investigation_id")) in investigation_ids:
            db.delete(doc)

    for inv in investigations:
        db.delete(inv)
    db.flush()

    storage = Path(settings.STORAGE_DIR)
    for dataset in datasets:
        shutil.rmtree(storage / "datasets" / str(dataset.id), ignore_errors=True)
        db.delete(dataset)
    db.flush()

    for source in db.query(DataSource).filter(DataSource.id.in_(source_ids)).all():
        shutil.rmtree(storage / "sources" / str(source.id), ignore_errors=True)
        db.delete(source)

    db.commit()
    return len(datasets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="store_true",
                        help="remove datasets created by the evaluation scripts")
    parser.add_argument("--all", action="store_true",
                        help="remove every dataset, investigation and document")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        counts = summarise(db)
        print("Currently stored:")
        for key, value in counts.items():
            print(f"  {key.replace('_', ' '):24s} {value}")

        if not (args.evaluation or args.all):
            print("\nNothing removed. Pass --evaluation or --all to delete.")
            return

        if args.all:
            if not args.yes:
                reply = input(
                    "\nThis deletes EVERYTHING, including your own uploads. Type 'delete' to confirm: "
                )
                if reply.strip().lower() != "delete":
                    print("Cancelled.")
                    return
            targets = db.query(Dataset).all()
            for doc in db.query(Document).all():
                db.delete(doc)
        else:
            targets = [
                d for d in db.query(Dataset).all() if d.name.startswith(EVAL_PREFIX)
            ]
            if not targets:
                print(
                    f"\nNo datasets named '{EVAL_PREFIX} …' were found. Datasets from "
                    "evaluation runs made before this script existed are not tagged, "
                    "so remove those from the Sources screen or use --all."
                )
                return

        removed = remove(db, targets)
        print(f"\nRemoved {removed} dataset(s) and everything attached to them.")
        print("\nNow stored:")
        for key, value in summarise(db).items():
            print(f"  {key.replace('_', ' '):24s} {value}")
    finally:
        db.close()


if __name__ == "__main__":
    main()