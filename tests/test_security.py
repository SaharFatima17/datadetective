"""Encryption and redaction tests (proposal Sec.19).

The redaction tests are the important ones: they assert on exactly what
`LLMClient.prepare()` would transmit, which is the single point every prompt
passes through before reaching a provider.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core import crypto, redaction
from app.llm.client import LLMClient


# ------------------------------------------------------------- encryption
@pytest.fixture
def encryption_on(monkeypatch):
    from cryptography.fernet import Fernet

    from app.config import settings

    monkeypatch.setattr(settings, "ENCRYPT_AT_REST", True)
    monkeypatch.setattr(settings, "ENCRYPTION_KEY", Fernet.generate_key().decode())
    yield


@pytest.fixture
def encryption_off(monkeypatch):
    """Force encryption off regardless of what the local .env says."""
    from app.config import settings

    monkeypatch.setattr(settings, "ENCRYPT_AT_REST", False)
    yield


def test_files_are_unreadable_on_disk_when_encryption_is_on(tmp_path, encryption_on):
    path = tmp_path / "secret.csv"
    payload = b"customer_email,revenue\nalice@example.com,100\n"

    crypto.write_bytes(path, payload)
    raw = path.read_bytes()

    assert raw != payload
    assert b"alice@example.com" not in raw
    assert crypto.is_encrypted(raw)
    assert crypto.read_bytes(path) == payload


def test_materialize_gives_a_readable_path_then_cleans_up(tmp_path, encryption_on):
    path = tmp_path / "data.csv"
    crypto.write_bytes(path, b"a,b\n1,2\n")

    with crypto.materialize(path) as readable:
        temp_path = readable
        assert pd.read_csv(readable).shape == (1, 2)

    assert not temp_path.exists(), "temporary plaintext file must be removed"


def test_plaintext_files_still_read_after_encryption_is_enabled(tmp_path, encryption_on):
    """A store written before encryption was turned on must keep working."""
    path = tmp_path / "legacy.csv"
    path.write_bytes(b"a,b\n1,2\n")          # written directly, unencrypted

    assert crypto.read_bytes(path) == b"a,b\n1,2\n"
    with crypto.materialize(path) as readable:
        assert readable == path              # no copy made


def test_encryption_off_writes_plaintext(tmp_path, encryption_off):
    path = tmp_path / "open.csv"
    crypto.write_bytes(path, b"hello")
    assert path.read_bytes() == b"hello"
    assert not crypto.is_encrypted(path.read_bytes())


# -------------------------------------------------------------- redaction
def test_sensitive_values_never_reach_the_llm():
    r = redaction.Redactor(
        columns=["customer_email"],
        values={"alice@example.com", "bob@example.com"},
    )
    prompt = (
        "Verified finding: customer_email = 'alice@example.com' contributed 64% "
        "of the decline, with bob@example.com second."
    )

    with redaction.use(r):
        system_out, prompt_out = LLMClient.prepare("You write reports.", prompt)

    assert "alice@example.com" not in prompt_out
    assert "bob@example.com" not in prompt_out
    assert redaction.REDACTED in prompt_out
    # the column itself is still named, so the model knows it exists
    assert "customer_email" in prompt_out
    assert "customer_email" in system_out
    assert "withheld" in system_out.lower()


def test_redaction_is_case_insensitive_and_prefers_longest_match():
    r = redaction.Redactor(columns=["name"], values={"Ann", "Ann Marie"})
    with redaction.use(r):
        _, out = LLMClient.prepare("s", "ann marie and ANN appear here")
    assert "Ann Marie" not in out and "ann marie" not in out
    assert out.count(redaction.REDACTED) == 2


def test_nothing_changes_when_no_column_is_marked():
    with redaction.use(redaction.EMPTY):
        system, prompt = LLMClient.prepare("sys", "revenue fell in South")
    assert system == "sys" and prompt == "revenue fell in South"


def test_very_short_values_are_not_scrubbed():
    """Scrubbing '1' or 'NA' would corrupt unrelated text without protecting anything."""
    r = redaction.Redactor(columns=["code"], values={"1", "NA", "LONGVALUE"})
    with redaction.use(r):
        _, out = LLMClient.prepare("s", "1 NA LONGVALUE")
    assert out.startswith("1 NA ")
    assert "LONGVALUE" not in out


def test_build_collects_values_only_for_marked_columns(monkeypatch):
    df = pd.DataFrame({
        "customer_email": ["alice@example.com", "bob@example.com"],
        "region": ["North", "South"],
    })

    class _Dataset:
        sensitive_columns = {"columns": ["customer_email"]}

    class _DB:
        def get(self, *_):
            return _Dataset()

    r = redaction.build(_DB(), "any-id", df)
    assert r.columns == ["customer_email"]
    assert "alice@example.com" in r.values
    assert "North" not in r.values          # unmarked column values are untouched

    with redaction.use(r):
        _, out = LLMClient.prepare("s", "alice@example.com is in North")
    assert "alice@example.com" not in out
    assert "North" in out
