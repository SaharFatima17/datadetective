"""Re-embed every indexed document with the currently configured model.

Vectors from two different embedding models are not comparable: the dimensions
differ, and even at equal dimensions the axes mean different things. So the
moment EMBEDDING_PROVIDER or EMBEDDING_MODEL changes in .env, everything already
indexed becomes unusable — retrieval skips those chunks rather than ranking them
against a number that only looks like a similarity.

This re-embeds them.

    python scripts/reindex_embeddings.py            # show what needs it
    python scripts/reindex_embeddings.py --apply    # do it

The chunk text is not recomputed, only the vectors, so chunk boundaries and
anything referring to them stay valid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal  # noqa: E402
from app.llm.embeddings import embedder  # noqa: E402
from app.models import Document, DocumentChunk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write the new vectors (otherwise just report)")
    parser.add_argument("--batch", type=int, default=20,
                        help="chunks per provider call")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        current = embedder.signature
        target_dim = len(embedder.embed_one("dimension probe"))
        print(f"current model : {current}  ({target_dim} dimensions)\n")

        chunks = db.query(DocumentChunk).all()
        stale = []
        for chunk in chunks:
            meta = chunk.chunk_metadata or {}
            vector = meta.get("vector") or []
            if meta.get("embedding") != current or len(vector) != target_dim:
                stale.append(chunk)

        print(f"{len(chunks)} chunk(s) indexed, {len(stale)} need re-embedding.")
        if not stale:
            print("Nothing to do.")
            return
        if not args.apply:
            by_model: dict[str, int] = {}
            for c in stale:
                key = (c.chunk_metadata or {}).get("embedding", "unknown")
                by_model[key] = by_model.get(key, 0) + 1
            print("\nThey were embedded with:")
            for model, count in by_model.items():
                print(f"  {model:28s} {count}")
            print("\nRe-run with --apply to re-embed them.")
            return

        done = 0
        for i in range(0, len(stale), args.batch):
            batch = stale[i:i + args.batch]
            vectors = embedder.embed([c.content for c in batch])
            for chunk, vector in zip(batch, vectors):
                meta = dict(chunk.chunk_metadata or {})
                meta["vector"] = vector
                meta["embedding"] = current
                chunk.chunk_metadata = meta
            db.commit()
            done += len(batch)
            print(f"  {done}/{len(stale)}", end="\r", flush=True)

        # A document is only as embedded as its chunks.
        for doc in db.query(Document).all():
            if doc.status != "embedded" and doc.chunks:
                doc.status = "embedded"
        db.commit()
        print(f"\nRe-embedded {done} chunk(s) with {current}.")
    finally:
        db.close()


if __name__ == "__main__":
    main()