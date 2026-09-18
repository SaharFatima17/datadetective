"""Show what the knowledge base actually holds, and what a question retrieves.

Run this when an answer says nothing was found although the document is listed:

    python scripts/inspect_knowledge.py
    python scripts/inspect_knowledge.py "what does zylo do"

It prints each document with the size of its vectors and how much text was
extracted, then runs the question through retrieval. The three failures it is
meant to separate:

    vector size differs      the document was indexed with another embedding
                             model and is being skipped — run reindex_embeddings
    very little text         the page was fetched but nothing readable came out,
                             which is what happens to sites that render their
                             content with JavaScript
    text is fine, no match   retrieval is working and the page genuinely does
                             not answer the question
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal  # noqa: E402
from app.llm.embeddings import embedder  # noqa: E402
from app.models import Document  # noqa: E402
from app.services import rag  # noqa: E402


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "what does this company do"
    db = SessionLocal()
    try:
        probe = embedder.embed_one("dimension probe")
        print(f"embedding model : {embedder.signature}  ({len(probe)} dimensions)\n")

        docs = db.query(Document).all()
        if not docs:
            print("Nothing is indexed.")
            return

        print(f"{'owner':10s} {'dims':>6s} {'chunks':>7s} {'chars':>8s}  title")
        print("-" * 78)
        for d in sorted(docs, key=lambda x: x.title):
            chunk = d.chunks[0] if d.chunks else None
            meta = (chunk.chunk_metadata or {}) if chunk else {}
            dims = len(meta.get("vector") or [])
            chars = sum(len(c.content or "") for c in d.chunks)
            flag = "  <- vector size differs" if dims and dims != len(probe) else ""
            if chars < 400:
                flag = "  <- very little text extracted"
            owner = str(d.owner_id)[:8] if d.owner_id else "(none)"
            print(f"{owner:10s} {dims:6d} {len(d.chunks):7d} {chars:8d}  "
                  f"{d.title[:38]}{flag}")

        print(f"\nretrieval for: {question!r}")
        hits = rag.search(db, question, top_k=6)
        if not hits:
            print("  nothing retrieved at all")
        for h in hits:
            print(f"  {h['score']:<8} {h['document_type']:12s} {h['document_title'][:44]}")
            print(f"           {h['content'][:110].strip()}")
    finally:
        db.close()


if __name__ == "__main__":
    main()