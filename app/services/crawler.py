"""Crawl a website into the knowledge base.

`fetch_url` retrieves one page. That is enough when someone links a specific
article, and useless when the thing they want the system to know about is a
company — whose information is spread across an about page, a products page, a
pricing page and a handful of linked PDFs.

This walks a site from a starting address: it follows links that stay on the
same domain, breadth first, and it downloads documents it finds linked there.
Each page and document becomes its own knowledge-base entry with its own URL, so
an answer can point at the page it came from rather than at "the website".

Three limits are deliberate and not configurable upward from the API:

    pages      a crawl that wanders is a crawl nobody can review
    depth      three clicks from the entry point covers a normal site
    size       one large PDF should not consume the whole budget

Nothing here executes page scripts, so single-page applications that render
their content in the browser will yield little. That is a real limitation and
the caller is told rather than left with empty pages.
"""

from __future__ import annotations

import re
import uuid
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

from sqlalchemy.orm import Session

from app.models import DataSource, Document
from app.services import ingestion, rag

MAX_PAGES = 25
MAX_DEPTH = 3
MAX_BYTES = 8 * 1024 * 1024

# Addresses that are never worth a request: assets, feeds, and the endless
# calendar/tag/search pages that make a crawler look like a denial of service.
SKIP_PATTERN = re.compile(
    r"\.(css|js|png|jpe?g|gif|svg|ico|woff2?|ttf|mp4|zip|gz)(\?|$)"
    r"|/(tag|tags|category|categories|search|login|signin|signup|cart|feed)(/|\?|$)",
    re.I,
)
DOC_PATTERN = re.compile(r"\.(pdf|docx?|pptx?|txt|csv)(\?|$)", re.I)


class _LinkParser(HTMLParser):
    """Pull hrefs out of a page without adding a parsing dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.title: str | None = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and not self.title:
            self.title = data.strip()[:200]


def _same_site(a: str, b: str) -> bool:
    """Treat www and the bare domain as one site, and nothing else as the same."""
    ha = urlparse(a).netloc.lower().removeprefix("www.")
    hb = urlparse(b).netloc.lower().removeprefix("www.")
    return ha == hb


def _normalise(url: str) -> str:
    """Drop the fragment and any trailing slash so one page is fetched once."""
    clean, _ = urldefrag(url)
    return clean.rstrip("/") or clean


def crawl_site(
    db: Session,
    start_url: str,
    owner_id: uuid.UUID | None = None,
    max_pages: int = MAX_PAGES,
    max_depth: int = MAX_DEPTH,
    subject: str | None = None,
) -> dict:
    """Walk a site and index every page and document it finds.

    Returns a summary rather than raising on the first bad page: one broken
    link in a site of twenty is not a reason to discard the other nineteen, and
    the caller needs to see which addresses failed.
    """
    import httpx

    if not start_url.lower().startswith(("http://", "https://")):
        raise ValueError("The address must start with http:// or https://")

    max_pages = max(1, min(max_pages, MAX_PAGES))
    max_depth = max(0, min(max_depth, MAX_DEPTH))

    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(_normalise(start_url), 0)])
    indexed: list[dict] = []
    failures: list[dict] = []
    documents_found = 0

    headers = {
        "User-Agent": ingestion.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en",
    }

    with httpx.Client(follow_redirects=True, timeout=30, headers=headers) as client:
        while queue and len(indexed) < max_pages:
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                failures.append({"url": url, "reason": f"HTTP {exc.response.status_code}"})
                continue
            except httpx.HTTPError as exc:
                failures.append({"url": url, "reason": str(exc)[:120]})
                continue

            content = response.content
            if len(content) > MAX_BYTES:
                failures.append({"url": url, "reason": "larger than the size limit"})
                continue

            content_type = response.headers.get("content-type", "").lower()
            is_document = ("html" not in content_type) or DOC_PATTERN.search(url)

            try:
                source = ingestion.register_url_source(
                    db, str(response.url), content, content_type, owner_id=owner_id)
                text = ingestion.extract_text_from_bytes(
                    content, content_type, str(response.url))
            except Exception as exc:  # noqa: BLE001
                failures.append({"url": url, "reason": f"could not read: {exc}"[:120]})
                continue

            if len(text.strip()) < 120:
                # A page whose text will not answer anything is noise in
                # retrieval, and noise in retrieval is worse than a gap.
                failures.append({"url": url, "reason": "too little readable text"})
                continue

            parser = _LinkParser()
            if not is_document:
                try:
                    parser.feed(content.decode(response.encoding or "utf-8", "ignore"))
                except Exception:  # noqa: BLE001
                    pass

            title = parser.title or _title_from_url(str(response.url))

            # Crawling the same site twice used to double the knowledge base:
            # each run created fresh documents, and retrieval then returned the
            # same passage several times over. A page already indexed from this
            # address is replaced rather than duplicated.
            existing = [
                d for d in db.query(Document).filter(
                    Document.document_type == "web_page",
                    Document.owner_id == owner_id).all()
                if (d.document_metadata or {}).get("url") == str(response.url)
            ]
            for old in existing:
                db.delete(old)
            if existing:
                db.flush()

            doc = rag.index_document(
                db,
                owner_id=owner_id,
                subject=subject,
                title=title,
                text=text,
                document_type="web_page",
                source_id=source.id,
                metadata={"url": str(response.url), "crawled_from": start_url},
            )
            source.status = "extracted"
            db.flush()

            indexed.append({"url": str(response.url), "title": title,
                            "chunks": len(doc.chunks),
                            "kind": "document" if is_document else "page"})
            if is_document:
                documents_found += 1

            if is_document or depth >= max_depth:
                continue

            for href in parser.links:
                target = _normalise(urljoin(str(response.url), href))
                if (target in seen or SKIP_PATTERN.search(target)
                        or not target.lower().startswith(("http://", "https://"))
                        or not _same_site(target, start_url)):
                    continue
                queue.append((target, depth + 1))

    db.commit()
    return {
        "start_url": start_url,
        "pages_indexed": len(indexed) - documents_found,
        "documents_indexed": documents_found,
        "total_chunks": sum(i["chunks"] for i in indexed),
        "indexed": indexed,
        "failed": failures[:10],
        "note": (
            "Pages are read as delivered; scripts are not executed, so a site "
            "that renders its content in the browser will yield little."
            if not indexed else
            "Each page is stored separately, so an answer can cite the page it "
            "came from rather than the site as a whole."
        ),
    }


def _title_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    last = path.rsplit("/", 1)[-1] if path else ""
    if not last:
        return urlparse(url).netloc
    return last.replace("-", " ").replace("_", " ")[:200]