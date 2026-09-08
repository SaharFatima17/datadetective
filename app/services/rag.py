"""Phase 6 - RAG (proposal Sec.14).

Chunks are embedded and stored in document_chunks with their vector in JSONB.
Retrieval is cosine similarity in Python plus structured metadata filters, which
is the "vector search + filters" the proposal asks for at a scale that fits an
FYP without a separate vector service.

Past investigations and their reports are indexed here too, which is what makes
Sec.14's current-vs-historical comparison possible.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.llm.embeddings import cosine_similarity, embedder
from app.models import (
    Document,
    DocumentChunk,
    Feedback,
    Finding,
    Investigation,
    Recommendation,
    Report,
)


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking: keeps paragraphs together where they fit."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) <= size:
                current = para
            else:
                for i in range(0, len(para), size - overlap):
                    piece = para[i : i + size]
                    if piece.strip():
                        chunks.append(piece.strip())
                current = ""
    if current:
        chunks.append(current)
    return chunks


def index_document(
    db: Session,
    title: str,
    text: str,
    document_type: str = "business_doc",
    source_id: uuid.UUID | None = None,
    metadata: dict | None = None,
) -> Document:
    doc = Document(
        source_id=source_id,
        title=title,
        document_type=document_type,
        extracted_text=text[:1_000_000],
        status="pending",
        document_metadata=metadata,
    )
    db.add(doc)
    db.flush()

    pieces = chunk_text(text)
    if pieces:
        vectors = embedder.embed(pieces)
        for i, (piece, vector) in enumerate(zip(pieces, vectors)):
            db.add(
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=piece,
                    token_count=len(piece.split()),
                    chunk_metadata={
                        "vector": vector,
                        "document_type": document_type,
                        **(metadata or {}),
                    },
                )
            )
    doc.status = "embedded"
    db.flush()
    return doc


def search(
    db: Session,
    query: str,
    top_k: int = 5,
    document_type: str | None = None,
    min_score: float = 0.0,
) -> list[dict]:
    """Semantic search with an optional structured filter (proposal Sec.14)."""
    q_vec = embedder.embed_one(query)

    stmt = db.query(DocumentChunk, Document).join(Document, DocumentChunk.document_id == Document.id)
    if document_type:
        stmt = stmt.filter(Document.document_type == document_type)

    scored = []
    for chunk, doc in stmt.all():
        vector = (chunk.chunk_metadata or {}).get("vector")
        if not vector:
            continue
        score = cosine_similarity(q_vec, vector)
        if score >= min_score:
            scored.append(
                {
                    "chunk_id": str(chunk.id),
                    "document_id": str(doc.id),
                    "document_title": doc.title,
                    "document_type": doc.document_type,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "score": round(score, 4),
                }
            )

    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]


def index_investigation_report(db: Session, investigation: Investigation, report: Report) -> Document:
    """Makes a finished investigation retrievable by later ones (Sec.14)."""
    parts = [
        f"Question: {investigation.question}",
        f"Target metric: {investigation.target_metric or 'unspecified'}",
        "",
        report.executive_summary or "",
    ]
    for finding in (report.content or {}).get("findings", []):
        parts.append(f"Finding: {finding.get('statement', '')}")
        parts.append(f"Evidence: {finding.get('evidence_summary', '')}")
    for rec in (report.content or {}).get("recommendations", []):
        parts.append(f"Recommendation: {rec.get('action', '')}")

    return index_document(
        db,
        title=f"Investigation report: {investigation.question[:120]}",
        text="\n".join(p for p in parts if p),
        document_type="past_report",
        metadata={
            "investigation_id": str(investigation.id),
            "report_id": str(report.id),
            "dataset_id": str(investigation.dataset_id) if investigation.dataset_id else None,
        },
    )


def past_feedback_for_driver(db: Session, driver_text: str,
                             limit: int = 20) -> dict:
    """How past recommendations for a similar driver were rated (Sec.13 point 4).

    Matched on the driver column name rather than the whole sentence, because
    "region = 'South' contributed 99%" and "region = 'North' contributed 71%"
    are the same KIND of finding and past ratings for one inform the other.
    """
    key = (driver_text or "").split("=")[0].strip().lower()
    if not key:
        return {"matched": 0, "useful": 0, "not_useful": 0, "examples": []}

    rows = (
        db.query(Feedback, Recommendation, Finding)
        .join(Recommendation, Feedback.target_id == Recommendation.id)
        .outerjoin(Finding, Recommendation.finding_id == Finding.id)
        .filter(Feedback.target_type == "recommendation")
        .order_by(Feedback.created_at.desc())
        .limit(200)
        .all()
    )

    useful, not_useful, examples = 0, 0, []
    for feedback, rec, finding in rows:
        statement = (finding.statement if finding else "") or rec.action or ""
        if key not in statement.lower():
            continue
        if feedback.was_adopted:
            useful += 1
        else:
            not_useful += 1
        if len(examples) < limit:
            examples.append({"action": rec.action,
                             "was_adopted": bool(feedback.was_adopted),
                             "comment": feedback.comment})

    return {"driver_key": key, "matched": useful + not_useful,
            "useful": useful, "not_useful": not_useful, "examples": examples}


def get_business_definition(db: Session, term: str) -> dict | None:
    """Proposal Sec.9 - get_business_definition(term)."""
    hits = search(db, term, top_k=3, document_type="kpi_definition")
    if not hits:
        hits = search(db, term, top_k=3, document_type="data_dictionary")
    if not hits:
        return None
    best = hits[0]
    return {
        "term": term,
        "definition": best["content"],
        "source": best["document_title"],
        "score": best["score"],
    }
