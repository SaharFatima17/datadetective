from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.core.deps import authorize_dataset, get_current_user, require_role
from app.database import get_db
from app.models import DataSource, Dataset, DatasetVersion, User
from app.models import DatasetColumn
from app.schemas.requests import (
    CleaningApplyRequest,
    ColumnSensitivityUpdate,
    DatasetUpdate,
    SQLIngestRequest,
    URLIngestRequest,
)
from app.services import cleaning, dataquality, ingestion, profiling, rag

router = APIRouter(prefix="/api", tags=["data"])


def _dataset(db: Session, dataset_id: uuid.UUID, user: User) -> Dataset:
    """Fetch a dataset the caller is allowed to use (proposal Sec.19)."""
    return authorize_dataset(db, dataset_id, user)


def _version(db: Session, ds: Dataset, version_id: str | None) -> DatasetVersion:
    v = db.get(DatasetVersion, uuid.UUID(version_id)) if version_id else db.get(
        DatasetVersion, ds.current_version_id
    )
    if not v:
        raise HTTPException(404, "Dataset version not found")
    return v


# ===================== Phase 2: ingestion ============================ #
@router.post("/sources/upload")
async def upload_source(file: UploadFile = File(...),
                        user: User = Depends(require_role("admin", "analyst")),
                        db: Session = Depends(get_db)):
    """Upload a file. Tabular files become datasets; documents are indexed for RAG."""
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB} MB limit")

    ext = Path(file.filename).suffix.lower()
    source = ingestion.register_file_source(db, file.filename, content, owner_id=user.id)
    response = {"source_id": str(source.id), "name": source.name,
                "checksum": source.checksum_sha256, "size_bytes": source.size_bytes}

    try:
        if ext in ingestion.TABULAR_EXTS:
            dataset = ingestion.create_dataset_from_source(db, source)
            version = db.get(DatasetVersion, dataset.current_version_id)
            df = ingestion.load_version(version)
            summary = profiling.profile_version(db, version, df)
            response.update(kind="dataset", dataset_id=str(dataset.id),
                            version_id=str(version.id), profile=summary)

        elif ext in ingestion.DOCUMENT_EXTS:
            artifact = next(a for a in source.artifacts if a.artifact_type == "original")
            text = ingestion.extract_text(Path(artifact.storage_path))
            doc = rag.index_document(db, owner_id=user.id, title=file.filename, text=text,
                                     document_type="business_doc", source_id=source.id)
            source.status = "extracted"
            response.update(kind="document", document_id=str(doc.id),
                            chunks=len(doc.chunks), characters=len(text))
        else:
            raise HTTPException(415, f"Unsupported file type: {ext}")

        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(400, f"Ingestion failed: {exc}") from exc

    return response


@router.post("/sources/sql")
def ingest_sql(payload: SQLIngestRequest,
               user: User = Depends(require_role("admin", "analyst")),
               db: Session = Depends(get_db)):
    try:
        dataset = ingestion.ingest_from_sql(db, payload.connection_url, payload.query,
                                           payload.name, owner_id=user.id)
        version = db.get(DatasetVersion, dataset.current_version_id)
        summary = profiling.profile_version(db, version, ingestion.load_version(version))
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(400, str(exc)) from exc
    return {"dataset_id": str(dataset.id), "version_id": str(version.id), "profile": summary}


@router.post("/sources/url")
def ingest_url(payload: URLIngestRequest,
               user: User = Depends(require_role("admin", "analyst")),
               db: Session = Depends(get_db)):
    try:
        source = ingestion.fetch_url(db, payload.url, owner_id=user.id)
        result = {"source_id": str(source.id), "url": payload.url}
        if payload.index_for_rag:
            artifact = source.artifacts[0]
            text = ingestion.extract_text(Path(artifact.storage_path))
            doc = rag.index_document(db, owner_id=user.id, title=payload.url, text=text,
                                     document_type="web_page", source_id=source.id)
            source.status = "extracted"
            result.update(document_id=str(doc.id), chunks=len(doc.chunks))
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(400, str(exc)) from exc
    return result


@router.get("/sources")
def list_sources(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(DataSource).filter(DataSource.is_deleted.is_(False))
    if user.role != "admin":
        query = query.filter(
            # Rows with no owner are created by the evaluation scripts and by
            # internal tooling. They used to be visible to everyone, which meant
            # a brand new account opened onto someone else's workspace. They now
            # belong to administrators only.
            DataSource.owner_id == user.id
        )
    rows = query.all()
    return {"count": len(rows), "sources": [
        {"id": str(s.id), "name": s.name, "type": s.source_type, "format": s.source_format,
         "status": s.status, "size_bytes": s.size_bytes, "created_at": s.created_at}
        for s in rows
    ]}


# ===================== datasets ====================================== #
@router.get("/datasets")
def list_datasets(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(Dataset).filter(Dataset.is_deleted.is_(False))
    if user.role != "admin":
        query = query.filter(Dataset.owner_id == user.id)
    rows = query.all()
    return {"count": len(rows), "datasets": [
        {"id": str(d.id), "name": d.name, "description": d.description,
         "versions": len(d.versions), "current_version_id": str(d.current_version_id)
         if d.current_version_id else None, "created_at": d.created_at}
        for d in rows
    ]}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: uuid.UUID, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    ds = _dataset(db, dataset_id, user)
    return {
        "id": str(ds.id), "name": ds.name, "description": ds.description,
        "current_version_id": str(ds.current_version_id) if ds.current_version_id else None,
        "versions": [
            {"id": str(v.id), "number": v.version_number, "type": v.version_type,
             "rows": v.row_count, "columns": v.column_count,
             "parent_version_id": str(v.parent_version_id) if v.parent_version_id else None,
             "cleaning_operations": v.cleaning_operations, "created_at": v.created_at}
            for v in sorted(ds.versions, key=lambda v: v.version_number)
        ],
    }


@router.patch("/datasets/{dataset_id}")
def update_dataset(dataset_id: uuid.UUID, payload: DatasetUpdate,
                   user: User = Depends(require_role("admin", "analyst")),
                   db: Session = Depends(get_db)):
    ds = _dataset(db, dataset_id, user)
    if payload.name:
        ds.name = payload.name
    if payload.description is not None:
        ds.description = payload.description
    db.commit()
    return {"id": str(ds.id), "name": ds.name, "description": ds.description}


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: uuid.UUID,
                   user: User = Depends(require_role("admin", "analyst")),
                   db: Session = Depends(get_db)):
    ds = _dataset(db, dataset_id, user)
    ds.is_deleted = True          # soft delete - lineage is never destroyed
    db.commit()
    return {"id": str(ds.id), "deleted": True}


@router.get("/datasets/{dataset_id}/preview")
def preview(dataset_id: uuid.UUID, rows: int = Query(20, le=200),
            version_id: str | None = None, user: User = Depends(get_current_user),
            db: Session = Depends(get_db)):
    ds = _dataset(db, dataset_id, user)
    df = ingestion.load_version(_version(db, ds, version_id))
    head = df.head(rows)
    return {
        "columns": [str(c) for c in df.columns],
        "rows": head.astype(object).where(head.notna(), None).to_dict(orient="records"),
        "total_rows": len(df),
    }


# ===================== Phase 3: profiling ============================ #
@router.get("/datasets/{dataset_id}/profile")
def get_profile(dataset_id: uuid.UUID, version_id: str | None = None,
                refresh: bool = False, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    ds = _dataset(db, dataset_id, user)
    version = _version(db, ds, version_id)
    if version.profile_summary and not refresh:
        return version.profile_summary
    summary = profiling.profile_version(db, version, ingestion.load_version(version))
    db.commit()
    return summary


@router.get("/datasets/{dataset_id}/health")
def health_dashboard(dataset_id: uuid.UUID, version_id: str | None = None,
                     user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    ds = _dataset(db, dataset_id, user)
    version = _version(db, ds, version_id)
    summary = version.profile_summary or profiling.profile_version(
        db, version, ingestion.load_version(version)
    )
    db.commit()
    issues = summary["dataset_issues"] + [i for c in summary["columns"] for i in c["issues"]]
    return {
        "health_score": summary["health_score"],
        "row_count": summary["row_count"],
        "column_count": summary["column_count"],
        "issue_counts": summary["issue_counts"],
        "issues": sorted(issues, key=lambda i: {"high": 0, "medium": 1, "low": 2}[i["severity"]]),
    }


@router.get("/datasets/{dataset_id}/drift")
def drift(dataset_id: uuid.UUID, from_version: str | None = None,
          to_version: str | None = None, user: User = Depends(get_current_user),
          db: Session = Depends(get_db)):
    """Compare two versions' distributions (proposal Sec.10, data drift).

    Defaults to the two most recent versions, which answers the question that
    actually matters after cleaning: did the operation change the population,
    or only tidy it?
    """
    ds = _dataset(db, dataset_id, user)
    ordered = sorted(ds.versions, key=lambda v: v.version_number)
    if len(ordered) < 2 and not (from_version and to_version):
        raise HTTPException(409, "This dataset has only one version, so there is "
                                 "nothing to compare it against")

    later = _version(db, ds, to_version) if to_version else ordered[-1]
    earlier = _version(db, ds, from_version) if from_version else ordered[-2]

    report = dataquality.detect_drift(ingestion.load_version(earlier),
                                      ingestion.load_version(later))
    return {"from_version": {"id": str(earlier.id), "number": earlier.version_number,
                             "type": earlier.version_type},
            "to_version": {"id": str(later.id), "number": later.version_number,
                           "type": later.version_type},
            **report}


# ============ sensitive columns (proposal Sec.19) ==================== #
@router.get("/datasets/{dataset_id}/columns")
def list_columns(dataset_id: uuid.UUID, version_id: str | None = None,
                 user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ds = _dataset(db, dataset_id, user)
    version = _version(db, ds, version_id)
    marked = set((ds.sensitive_columns or {}).get("columns", []))
    rows = (
        db.query(DatasetColumn)
        .filter(DatasetColumn.version_id == version.id)
        .order_by(DatasetColumn.position)
        .all()
    )
    return {
        "dataset_id": str(ds.id),
        "version_id": str(version.id),
        "sensitive_columns": sorted(marked),
        "columns": [
            {"name": c.name, "inferred_type": c.inferred_type,
             "semantic_label": c.semantic_label, "sensitive": c.sensitive}
            for c in rows
        ],
    }


@router.patch("/datasets/{dataset_id}/columns/{column_name}")
def set_column_sensitivity(dataset_id: uuid.UUID, column_name: str,
                           payload: ColumnSensitivityUpdate,
                           user: User = Depends(require_role("admin", "analyst")),
                           db: Session = Depends(get_db)):
    """Mark a column sensitive so its values never reach the LLM.

    The flag lives on the dataset, not on one version, so it survives cleaning
    and re-profiling. The column stays fully usable by the local tool layer.
    """
    ds = _dataset(db, dataset_id, user)
    version = _version(db, ds, None)

    known = {
        c.name for c in
        db.query(DatasetColumn).filter(DatasetColumn.version_id == version.id).all()
    }
    if column_name not in known:
        raise HTTPException(404, f"No column named '{column_name}' in this dataset")

    marked = set((ds.sensitive_columns or {}).get("columns", []))
    marked.add(column_name) if payload.sensitive else marked.discard(column_name)
    ds.sensitive_columns = {"columns": sorted(marked)}

    # re-profile so stored samples for a newly sensitive column are dropped
    profiling.profile_version(db, version, ingestion.load_version(version),
                              sensitive=sorted(marked))
    db.commit()
    return {"dataset_id": str(ds.id), "column": column_name,
            "sensitive": payload.sensitive, "sensitive_columns": sorted(marked)}


# ===================== Phase 4: cleaning ============================= #
@router.get("/datasets/{dataset_id}/cleaning-plan")
def cleaning_plan(dataset_id: uuid.UUID, version_id: str | None = None,
                  user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    """Read-only: proposes operations without changing anything."""
    ds = _dataset(db, dataset_id, user)
    df = ingestion.load_version(_version(db, ds, version_id))
    return cleaning.propose_plan(df)


@router.post("/datasets/{dataset_id}/cleaning-plan/apply")
def apply_cleaning(dataset_id: uuid.UUID, payload: CleaningApplyRequest,
                   version_id: str | None = None,
                   user: User = Depends(require_role("admin", "analyst")),
                   db: Session = Depends(get_db)):
    """Applies the plan and creates a NEW version. The input version is untouched."""
    ds = _dataset(db, dataset_id, user)
    parent = _version(db, ds, version_id)
    df = ingestion.load_version(parent)

    plan = cleaning.propose_plan(df)
    cleaned, log = cleaning.apply_plan(df, plan, payload.approved_op_ids)

    version = ingestion.new_version(
        db, ds, cleaned, parent,
        version_type="cleaned",
        cleaning_operations={"operations": log, "parent_version": str(parent.id)},
        created_by_agent="cleaning",
    )
    after = profiling.profile_version(db, version, cleaned)  # inherits sensitivity from the dataset
    db.commit()

    return {
        "new_version_id": str(version.id),
        "version_number": version.version_number,
        "executed": [o for o in log if o.get("executed")],
        "skipped": [o for o in log if not o.get("executed")],
        "before": {"rows": parent.row_count,
                   "health_score": (parent.profile_summary or {}).get("health_score")},
        "after": {"rows": version.row_count, "health_score": after["health_score"]},
    }