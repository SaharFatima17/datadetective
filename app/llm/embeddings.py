"""Embeddings for the RAG layer (proposal Sec.14).

Vectors are stored as JSONB in document_chunks and compared in Python. At FYP
scale (thousands of chunks, not millions) this is fast enough and avoids the
pgvector build step on Windows. Sec.16 allows either; swapping to pgvector or
Qdrant later only changes this file and `rag.search`.
"""

from __future__ import annotations

import hashlib
import math

from app.config import settings

DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
}


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self) -> None:
        self.provider = settings.EMBEDDING_PROVIDER.lower()
        self.model = settings.EMBEDDING_MODEL or DEFAULT_EMBEDDING_MODELS.get(
            self.provider, ""
        )
        self.api_key = settings.EMBEDDING_API_KEY or settings.LLM_API_KEY

    @property
    def signature(self) -> str:
        """Identifies the vector space these embeddings belong to.

        Vectors from different models are not comparable — different dimensions,
        and even at equal dimensions the axes mean different things. Storing the
        signature with each chunk is what lets retrieval notice the mismatch
        instead of silently returning nonsense similarities.
        """
        return f"{self.provider}:{self.model or 'default'}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "mock":
            return [_mock_embed(t) for t in texts]
        if self.provider == "openai":
            return self._openai(texts)
        if self.provider == "gemini":
            return [self._gemini_one(t) for t in texts]
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {self.provider}")

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _openai(self, texts: list[str]) -> list[list[float]]:
        import httpx

        r = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]

    def _gemini_one(self, text: str) -> list[float]:
        import httpx

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )
        r = self._request(
            httpx, url, {"content": {"parts": [{"text": text}]}}
        )
        return r.json()["embedding"]["values"]

    def _request(self, httpx, url: str, payload: dict):
        """POST with backoff. Indexing a document is dozens of calls in a row,
        so a free-tier rate limit is met on the first document, not the tenth."""
        import random
        import time

        attempts = max(1, settings.LLM_MAX_RETRIES)
        for attempt in range(attempts):
            # No up-front sleep. This used to wait LLM_MIN_INTERVAL_MS before
            # every request — six seconds with the free-tier setting — which
            # is a budget meant for the chat model, not for embeddings, and
            # was paid unconditionally. Planning an investigation embeds
            # several terms in a row, so a single run slept for tens of
            # seconds doing nothing. A rate limit, if it comes, is handled by
            # the backoff below, which is where waiting belongs.
            r = httpx.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                raise EmbeddingError(
                    f"{self.provider} has no embedding model called "
                    f"'{self.model}'. Run `python scripts/list_models.py` and "
                    "set EMBEDDING_MODEL in .env."
                )
            if r.status_code not in (408, 429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise EmbeddingError(
                    f"{self.provider} embeddings returned {r.status_code}: "
                    f"{r.text[:200]}"
                )
            time.sleep(min(30.0, 2 ** attempt + random.uniform(0, 1)))
        raise EmbeddingError("unreachable")


def _mock_embed(text: str) -> list[float]:
    """Deterministic bag-of-words hash vector.

    Not semantically meaningful, but stable and it does put documents sharing
    vocabulary near each other - enough to build and test the retrieval
    plumbing before an API key exists.
    """
    dim = settings.EMBEDDING_DIM
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


embedder = Embedder()