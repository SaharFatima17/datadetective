"""Password hashing and JWT handling (proposal Sec.19).

bcrypt is used through passlib for hashing and PyJWT for tokens. PyJWT is
already an indirect dependency of the MCP SDK, so only passlib and bcrypt
are new.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#use for testing purposes only, change it before deploying anywhere real
class TokenError(Exception):
    """Raised when a token is missing, malformed, expired or wrongly signed."""

#create a new token for the user with the given id and role, expiring in the given number of minutes 
def hash_password(password: str) -> str:
    # bcrypt silently truncates beyond 72 bytes, so reject rather than
    # accept a password whose tail is ignored.
    if len(password.encode()) > 72:
        raise ValueError("Password must be 72 bytes or fewer")
    return pwd_context.hash(password)

#verify that the given plain password matches the given hashed password, returning true if they match and false otherwise.
def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:  # noqa: BLE001 - malformed stored hash
        return False

#create a new access token for the user with the given id and role, expiring in the given number of minutes
def create_access_token(user_id: uuid.UUID, role: str,
                        expires_minutes: int | None = None) -> str:
    minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

#decode the given access token and return its payload as a dictionary, raising TokenError if the token is invalid or expired
def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY,
                          algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Token is invalid") from exc
