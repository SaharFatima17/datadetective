"""At-rest encryption for stored files (proposal Sec.19).

Every byte the system persists under STORAGE_DIR - original uploads, dataset
versions, extracted artifacts - passes through here.

Two design points worth knowing:

1. Encrypted blobs carry a `DDENC1:` marker. Reads check for it, so a store
   that already holds plaintext files keeps working after encryption is turned
   on, and turning it off again does not orphan the files already encrypted.

2. pandas, DuckDB and the document extractors all open files by path, so an
   encrypted file cannot be handed to them directly. `materialize()` decrypts
   to a temporary file for the duration of a `with` block and removes it
   afterwards. Plaintext files are passed straight through with no copy.
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

MARKER = b"DDENC1:"


class EncryptionError(RuntimeError):
    pass

#use Fernet symmetric encryption to encrypt and decrypt data at rest, with a marker to indicate encrypted files.
def _fernet():
    from cryptography.fernet import Fernet

    key = settings.ENCRYPTION_KEY.strip()
    if not key:
        raise EncryptionError(
            "ENCRYPT_AT_REST is on but ENCRYPTION_KEY is empty in .env. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        )
    try:
        return Fernet(key.encode())
    except Exception:
        # Accept an arbitrary passphrase too, by deriving a valid Fernet key
        # from it. Convenient for a student setup; a generated key is better.
        derived = base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest())
        return Fernet(derived)

#store the given data in a file at the given path
def enabled() -> bool:
    return bool(settings.ENCRYPT_AT_REST)


def is_encrypted(blob: bytes) -> bool:
    return blob.startswith(MARKER)


def encrypt(data: bytes) -> bytes:
    if not enabled():
        return data
    return MARKER + _fernet().encrypt(data)


def decrypt(blob: bytes) -> bytes:
    if not is_encrypted(blob):
        return blob            # written before encryption was enabled
    from cryptography.fernet import InvalidToken

    try:
        return _fernet().decrypt(blob[len(MARKER):])
    except InvalidToken as exc:
        raise EncryptionError(
            "Stored file could not be decrypted. ENCRYPTION_KEY has probably "
            "changed since it was written."
        ) from exc


# --------------------------------------------------------------------- #
def write_bytes(path: str | Path, data: bytes) -> int:
    """Write a file, encrypting it if encryption is on. Returns bytes written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = encrypt(data)
    path.write_bytes(blob)
    return len(blob)


def read_bytes(path: str | Path) -> bytes:
    """Read a file, decrypting it if it was stored encrypted."""
    return decrypt(Path(path).read_bytes())


@contextmanager
def materialize(path: str | Path):
    """Yield a real filesystem path holding plaintext content.

    Used wherever a library must open the file itself (pandas, DuckDB, pypdf).
    If the file is not encrypted, its own path is yielded and nothing is copied.
    """
    path = Path(path)
    with path.open("rb") as fh:
        head = fh.read(len(MARKER))

    if head != MARKER:
        yield path
        return

    plaintext = decrypt(path.read_bytes())
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, prefix="dd-")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(plaintext)
        yield Path(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
