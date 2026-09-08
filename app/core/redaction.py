"""Sensitive-column redaction for anything sent to the LLM (proposal Sec.19).

A column marked sensitive stays fully usable by the deterministic tool layer -
it can still be grouped, joined, counted and tested, because those run locally.
What must never leave the machine is its VALUES.

The redactor is installed once per investigation and read inside
`LLMClient.prepare()`, which is the single point every prompt passes through.
Putting it there rather than at each call site means a new agent added later
cannot accidentally bypass it.

The model is still told the column exists and that its values were withheld,
so it can reason about the column without seeing the data.
"""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy.orm import Session

REDACTED = "[REDACTED]"

# Values shorter than this are skipped: scrubbing "1" or "NA" out of a prompt
# would corrupt unrelated text without protecting anything meaningful.
MIN_VALUE_LENGTH = 3
MAX_TRACKED_VALUES = 5000


class Redactor:
    """Holds the sensitive column names and their distinct values."""

    def __init__(self, columns: list[str] | None = None,
                 values: set[str] | None = None) -> None:
        self.columns = sorted(columns or [])
        self.values = values or set()
        self._pattern = self._compile()

    def _compile(self):
        usable = sorted(
            (v for v in self.values if len(v) >= MIN_VALUE_LENGTH),
            key=len, reverse=True,          # longest first, so substrings don't win
        )
        if not usable:
            return None
        return re.compile("|".join(re.escape(v) for v in usable), re.IGNORECASE)

    @property
    def active(self) -> bool:
        return bool(self.columns)

    def scrub(self, text: str | None) -> str | None:
        """Replace every sensitive value found in the text."""
        if not text or not self.active:
            return text
        out = text
        if self._pattern is not None:
            out = self._pattern.sub(REDACTED, out)
        return out

    def note(self) -> str:
        if not self.active:
            return ""
        return (
            "\n\nData handling: the following columns are marked sensitive and their "
            f"values have been withheld from you: {', '.join(self.columns)}. "
            f"Any value shown as {REDACTED} came from one of them. You may reason "
            "about these columns and propose tests that use them - the tools run "
            "locally and can read the real values - but never guess or invent what "
            "the withheld values are."
        )


EMPTY = Redactor()
_active: ContextVar[Redactor] = ContextVar("active_redactor", default=EMPTY)


def current() -> Redactor:
    return _active.get()


@contextmanager
def use(redactor: Redactor):
    """Install a redactor for the duration of a block."""
    token = _active.set(redactor)
    try:
        yield redactor
    finally:
        _active.reset(token)


# --------------------------------------------------------------------- #
def sensitive_columns(db: Session, dataset_id: uuid.UUID) -> list[str]:
    from app.models import Dataset

    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        return []
    return list((dataset.sensitive_columns or {}).get("columns", []))


def build(db: Session, dataset_id: uuid.UUID, df=None) -> Redactor:
    """Build a redactor for a dataset, collecting the values to withhold."""
    columns = sensitive_columns(db, dataset_id)
    if not columns:
        return EMPTY

    values: set[str] = set()
    if df is not None:
        for column in columns:
            if column not in df.columns:
                continue
            for value in df[column].dropna().unique()[:MAX_TRACKED_VALUES]:
                text = str(value).strip()
                if len(text) >= MIN_VALUE_LENGTH:
                    values.add(text)
    return Redactor(columns, values)
