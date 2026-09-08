from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Confirms the API is up and the database is reachable."""
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"

    return {"status": "ok", "database": db_status}


@router.get("/health/tables")
def list_tables(db: Session = Depends(get_db)):
    """Lists the tables that actually exist, so you can verify the migration ran."""
    rows = db.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
    ).fetchall()
    tables = [r[0] for r in rows]
    return {"count": len(tables), "tables": tables}
