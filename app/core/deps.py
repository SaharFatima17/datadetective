"""FastAPI auth dependencies (proposal Sec.19).

`get_current_user` resolves the bearer token to a User.
`require_role(...)` restricts a route to named roles.
`authorize_dataset` enforces per-dataset ownership: only the owner, or a
user with the admin role, may touch a dataset.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import TokenError, decode_access_token
from app.database import get_db
from app.models import Dataset, Investigation, User

# auto_error=False so a missing header produces our own 401 message rather
# than FastAPI's generic one.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(token: str | None = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)) -> User:
    if not token:
        raise CREDENTIALS_ERROR
    try:
        payload = decode_access_token(token)
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc),
                            headers={"WWW-Authenticate": "Bearer"}) from exc

    subject = payload.get("sub")
    if not subject:
        raise CREDENTIALS_ERROR
    try:
        user = db.get(User, uuid.UUID(subject))
    except ValueError as exc:
        raise CREDENTIALS_ERROR from exc

    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User is inactive or no longer exists")
    return user


def require_role(*roles: str):
    """Dependency factory: `Depends(require_role("admin"))`."""

    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"This action requires one of these roles: {', '.join(roles)}",
            )
        return user

    return dependency


def is_admin(user: User) -> bool:
    return user.role == "admin"


def authorize_dataset(db: Session, dataset_id: uuid.UUID, user: User) -> Dataset:
    """Fetch a dataset and confirm the caller may use it.

    404 when it does not exist, 403 when it belongs to somebody else. Datasets
    created before authentication existed have no owner and stay open, so
    enabling auth does not orphan earlier work.
    """
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.is_deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    if dataset.owner_id is not None and dataset.owner_id != user.id and not is_admin(user):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "This dataset belongs to another user")
    return dataset


def authorize_investigation(db: Session, investigation_id: uuid.UUID,
                            user: User) -> Investigation:
    investigation = db.get(Investigation, investigation_id)
    if not investigation:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Investigation not found")
    if (investigation.owner_id is not None
            and investigation.owner_id != user.id and not is_admin(user)):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "This investigation belongs to another user")
    return investigation
