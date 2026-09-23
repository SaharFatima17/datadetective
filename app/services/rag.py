"""Phase 6 - RAG (proposal Sec.14).

Chunks are embedded and stored in document_chunks with their vector in JSONB.
Retrieval is cosine similarity in Python plus structured metadata filters, which
is the "vector search + filters" the proposal asks for at a scale that fits an
FYP without a separate vector service.

Past investigations and their reports are indexed here too, which is what makes
Sec.14's current-vs-historical comparison possible.
"""

from __future__ import annotations

import logging

import uuid

from sqlalchemy.orm import Session

from app.llm.client import llm
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


logger = logging.getLogger(__name__)


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
    owner_id: uuid.UUID | None = None,
    subject: str | None = None,
) -> Document:
    doc = Document(
        source_id=source_id,
        owner_id=owner_id,
        title=title,
        document_type=document_type,
        extracted_text=text[:1_000_000],
        status="pending",
        # The subject is stored alongside whatever the caller passed, so a
        # document can be grouped at the moment it is indexed rather than
        # needing a second step.
        document_metadata={**(metadata or {}),
                           **({"subject": subject.strip()[:120]}
                              if (subject or "").strip() else {})} or None,
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
                        # which model produced this vector; vectors from two
                        # models cannot be compared with each other
                        "embedding": embedder.signature,
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
    document_type: str | list[str] | None = None,
    min_score: float = 0.0,
    owner_id=None,
    scope: str | None = None,
) -> list[dict]:
    """Semantic search with an optional structured filter (proposal Sec.14).

    `document_type` accepts a list, because "business context" is not one kind
    of document: a KPI sheet, an incident note and a retrieved web page all
    serve the same purpose during planning.
    """
    q_vec = embedder.embed_one(query)

    stmt = db.query(DocumentChunk, Document).join(Document, DocumentChunk.document_id == Document.id)
    if isinstance(document_type, (list, tuple, set)):
        stmt = stmt.filter(Document.document_type.in_(list(document_type)))
    elif document_type:
        stmt = stmt.filter(Document.document_type == document_type)
    if owner_id is not None:
        # Retrieval is scoped the same way the listings are: an investigation
        # must not be informed by another account's documents or reports.
        stmt = stmt.filter(Document.owner_id == owner_id)

    scored = []
    skipped = 0
    # A scope narrows retrieval to one body of knowledge. Applied here rather
    # than by filtering results afterwards, so the top_k are the best matches
    # within that body instead of whatever survived a global ranking.
    for chunk, doc in stmt.all():
        vector = (chunk.chunk_metadata or {}).get("vector")
        if not vector:
            continue
        # A chunk embedded by a different model lives in a different vector
        # space. Comparing across the two produces a number that looks like a
        # similarity and means nothing, so those chunks are skipped rather than
        # quietly ranked. `scripts/reindex_embeddings.py` brings them back.
        if len(vector) != len(q_vec):
            skipped += 1
            continue
        if scope and scope_key(doc) != scope:
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
                    # the page address, when the document came from the web —
                    # a citation the reader can actually open
                    "url": (doc.document_metadata or {}).get("url"),
                    "scope": scope_key(doc),
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
        # A past report belongs to whoever ran the investigation. Without this
        # it is ownerless, and every other account sees it in their knowledge
        # base and retrieves it while planning their own investigations.
        owner_id=investigation.owner_id,
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


# A similarity below this is not a definition, it is the nearest thing in the
# index. Without a floor, every word in the question comes back "defined" by
# whatever chunk happened to rank first, and the report then claims it consulted
# a definition of "recently".
MIN_DEFINITION_SCORE = 0.35


ANSWER_SYSTEM = (
    "Answer the question using only the passages provided. Every claim must be "
    "supported by them. If they do not contain the answer, say so plainly "
    "instead of filling the gap. Cite sources as [1], [2] matching the numbered "
    "passages. Be brief: three or four sentences."
)


def scope_key(doc) -> str:
    """Which body of knowledge a document belongs to.

    Grouping matters because a crawled site can hold a hundred passages while
    an incident note holds one, and a similarity search over everything lets
    the large body bury the small one — the small one often being the document
    that actually answers the question.

    The subject the person gave it wins. Grouping by where a document came from
    was the first attempt and it breaks on the obvious case: a company's website
    and that same company's pitch deck are one subject, but one was crawled and
    the other uploaded, so they landed in different bodies and choosing either
    lost half the material. Source is only the fallback when no subject is set.
    """
    meta = doc.document_metadata or {}
    subject = (meta.get("subject") or "").strip()
    if subject:
        return f"subject:{subject.lower()}"

    url = meta.get("url") or ""
    if url.startswith("http"):
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower().removeprefix("www.")
        return f"site:{host}"
    if doc.document_type == "past_report":
        return "reports"
    return "uploads"


def scope_label(key: str) -> str:
    if key.startswith("subject:"):
        return key[8:].title()
    if key.startswith("site:"):
        return key[5:]
    return {"reports": "Past investigation reports",
            "uploads": "Uploaded documents and definitions"}.get(key, key)


def list_scopes(db: Session, owner_id=None) -> list[dict]:
    """The bodies of knowledge available, so a question can be aimed at one."""
    query = db.query(Document)
    if owner_id is not None:
        query = query.filter(Document.owner_id == owner_id)

    grouped: dict[str, dict] = {}
    for doc in query.all():
        key = scope_key(doc)
        entry = grouped.setdefault(
            key, {"key": key, "label": scope_label(key), "documents": 0,
                  "chunks": 0, "assigned": key.startswith("subject:")})
        entry["documents"] += 1
        entry["chunks"] += len(doc.chunks)
    return sorted(grouped.values(), key=lambda g: -g["chunks"])


def set_subject(db: Session, document_id, subject: str | None) -> dict:
    """Move a document into a subject, or back to its source grouping.

    Existing documents were indexed before subjects existed, so this is how
    they get grouped without re-uploading them: the text and its vectors are
    untouched, only the label that decides which body they belong to.
    """
    doc = db.get(Document, document_id)
    if not doc:
        raise ValueError("No such document")
    meta = dict(doc.document_metadata or {})
    cleaned = (subject or "").strip()
    if cleaned:
        meta["subject"] = cleaned[:120]
    else:
        meta.pop("subject", None)
    doc.document_metadata = meta
    db.flush()
    for chunk in doc.chunks:
        cm = dict(chunk.chunk_metadata or {})
        cm["subject"] = meta.get("subject")
        chunk.chunk_metadata = cm
    db.commit()
    return {"id": str(doc.id), "title": doc.title, "scope": scope_key(doc)}


def model_status() -> str:
    """Why a composed answer is missing: not set up, or set up and failing.

    These are different situations and used to share one message — "no
    language model is configured" — which was false whenever a model *was*
    configured and a call had simply failed on quota or a dropped connection.
    Telling someone their setup is missing when it is not sends them looking
    for a problem that does not exist.
    """
    provider = (getattr(llm, "provider", "") or "").lower()
    return "not_configured" if provider in {"", "mock"} else "unavailable"


def fallback_notice() -> str:
    if model_status() == "not_configured":
        return ("No language model is configured, so this is the closest "
                "matching passage rather than a written answer.")
    return ("The language model is unavailable right now — often a usage "
            "limit that resets daily — so this is the closest matching passage "
            "rather than a written answer.")


def answer_from_documents(db: Session, question: str, top_k: int = 6,
                          owner_id=None, scope: str | None = None,
                          on_compose=None) -> dict:
    """Answer a question from indexed documents, with citations (Sec.14).

    This is the document counterpart to an investigation, and it is deliberately
    weaker in what it claims. An investigation computes a number and can re-run
    the calculation to prove it; a document answer can only point at the passage
    it came from. So the passages are returned alongside the answer and the
    answer is never presented as a measurement.

    Without a configured model the passages are returned as they are. That is an
    honest degradation: quoting the source is a worse reading experience than a
    written answer, but it is never a fabricated one.
    """
    hits = search(db, question, top_k=top_k, owner_id=owner_id,
                  min_score=0.05, scope=scope)
    if not hits:
        return {
            "answered": False,
            "answer": ("Nothing in the indexed documents addresses that. Add a "
                       "page or document that covers it and ask again."),
            "sources": [],
        }

    numbered = "\n\n".join(
        f"[{i + 1}] from \"{h['document_title']}\":\n{h['content'][:1200]}"
        for i, h in enumerate(hits)
    )

    answer, grounded = None, True
    if on_compose:
        # the slow part starts here; say so while it runs
        on_compose(len(hits))
    try:
        raw = llm.complete(system=ANSWER_SYSTEM,
                           prompt=f"Question: {question}\n\nPassages:\n{numbered}")
        text = (raw or "").strip()
        if text and not text.startswith("["):
            answer = text
    except Exception:  # noqa: BLE001
        answer = None

    if answer is None:
        grounded = False
        answer = fallback_notice()

    # When the passages come from unrelated bodies of knowledge, say so. A
    # composed answer that draws on a café's incident note and a software
    # company's product page reads as one coherent statement about one subject,
    # and there is nothing in its wording to tell the reader it is not.
    bodies = {h["scope"] for h in hits}
    mixed = len(bodies) > 1 and scope is None

    return {
        "answered": True,
        "answer": answer,
        "composed": grounded,
        "scopes_used": sorted(bodies),
        "mixed_sources": mixed,
        "mixed_note": (
            "These passages come from more than one body of knowledge "
            f"({', '.join(scope_label(b) for b in sorted(bodies))}). Check the "
            "sources below before treating this as one subject, or ask the "
            "question against a single source."
        ) if mixed else None,
        "sources": [
            {"n": i + 1, "title": h["document_title"], "score": h["score"],
             "excerpt": h["content"][:400], "type": h["document_type"]}
            for i, h in enumerate(hits)
        ],
        "note": ("Sourced from indexed documents. Unlike an investigation "
                 "finding, this is not a computed figure — it can be traced to "
                 "the passage it came from, not re-calculated."),
    }


BRIEF_SYSTEM = (
    "You write a short factual brief from supplied passages. Return JSON only:\n"
    '{"title": "...", "summary": "...", '
    '"sections": [{"heading": "...", "body": "...", "cites": [1, 2]}]}\n'
    "Rules: every sentence must come from the passages. Cite the numbered "
    "passages you used in `cites`. Four to six sections. If the passages do not "
    "cover something, leave it out rather than filling it in. Do not invent "
    "figures, dates, names or claims."
)


def compose_brief(db: Session, topic: str, top_k: int = 14,
                  owner_id=None, scope: str | None = None) -> dict:
    """Assemble a brief on a topic from indexed documents (proposal Sec.14).

    This is not an investigation report and must never be mistaken for one. An
    investigation computes figures and can re-run the calculation that produced
    each of them; a brief can only point at the passage a statement came from.
    The two are different kinds of claim, so this returns its own shape, carries
    its own wording, and is never filed under Reports.

    Without a configured model the sections are the retrieved passages grouped
    by the page they came from. That is a worse read than composed prose and a
    truthful one: nothing is asserted that was not retrieved.
    """
    hits = search(db, topic, top_k=top_k, owner_id=owner_id,
                  min_score=0.05, scope=scope)
    if not hits:
        return {"available": False,
                "reason": "Nothing indexed covers that topic."}

    numbered = "\n\n".join(
        f"[{i + 1}] from \"{h['document_title']}\":\n{h['content'][:1500]}"
        for i, h in enumerate(hits)
    )

    composed, data = True, None
    try:
        result = llm.complete_json(
            system=BRIEF_SYSTEM,
            prompt=f"Topic: {topic}\n\nPassages:\n{numbered}")
        if isinstance(result, dict) and result.get("sections"):
            data = result
    except Exception:  # noqa: BLE001
        data = None

    if data is None:
        composed = False
        # Group what was retrieved by source rather than pretending to a
        # narrative the model did not write.
        by_source: dict[str, list[tuple[int, str]]] = {}
        for i, h in enumerate(hits):
            by_source.setdefault(h["document_title"], []).append((i + 1, h["content"]))
        data = {
            "title": topic,
            "summary": (("No language model is configured"
                         if model_status() == "not_configured" else
                         "The language model is unavailable right now")
                        + ", so this is the "
                        "retrieved material grouped by source rather than a "
                        "written brief. Nothing has been summarised or inferred."),
            "sections": [
                {"heading": title,
                 "body": "\n\n".join(c[:600] for _, c in parts),
                 "cites": [n for n, _ in parts]}
                for title, parts in list(by_source.items())[:8]
            ],
        }

    return {
        "available": True,
        "composed": composed,
        "topic": topic,
        "title": data.get("title") or topic,
        "summary": data.get("summary", ""),
        "sections": data.get("sections", []),
        "sources": [
            {"n": i + 1, "title": h["document_title"], "type": h["document_type"],
             "score": h["score"], "excerpt": h["content"][:300],
             "url": h.get("url")}
            for i, h in enumerate(hits)
        ],
        "basis": (
            "Sourced from indexed pages and documents. Unlike an investigation "
            "report, nothing here is a computed figure: each statement traces to "
            "the passage it came from, and cannot be re-calculated."
        ),
    }


def get_business_definition(db: Session, term: str, owner_id=None) -> dict | None:
    """Proposal Sec.9 - get_business_definition(term)."""
    hits = search(db, term, top_k=3, document_type="kpi_definition",
                  min_score=MIN_DEFINITION_SCORE, owner_id=owner_id)
    if not hits:
        hits = search(db, term, top_k=3, document_type="data_dictionary",
                      min_score=MIN_DEFINITION_SCORE, owner_id=owner_id)
    if not hits:
        return None
    best = hits[0]
    return {
        "term": term,
        "definition": best["content"],
        "source": best["document_title"],
        "score": best["score"],
    }