from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.config import settings
from app.core.deps import get_current_user, require_role
from app.core.security import create_access_token, hash_password, verify_password
from app.database import get_db
from app.models import User
from app.schemas.requests import LoginRequest, RegisterRequest, RoleUpdate

router = APIRouter(prefix="/api/auth", tags=["auth"])

ALLOWED_ROLES = {"admin", "analyst", "viewer"}


def _public(user: User) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "organization": user.organization,
        "is_active": user.is_active,
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """The first account created becomes the admin; later ones are analysts."""
    email = payload.email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "That email is already registered")

    try:
        hashed = hash_password(payload.password)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    first_user = db.query(User).count() == 0
    user = User(
        email=email,
        full_name=payload.full_name,
        hashed_password=hashed,
        organization=payload.organization,
        role="admin" if first_user else "analyst",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _public(user)

#create a new access token for the user with the given id and role 
@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.strip().lower()).first()
    if not user or not user.hashed_password or not verify_password(
        payload.password, user.hashed_password
    ):
        # Same message either way - do not reveal which emails exist.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This account is disabled")

    return {
        "access_token": create_access_token(user.id, user.role),
        "token_type": "bearer",
        "expires_in_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "user": _public(user),
    }

#returns the current user's information based on the access token provided in the request 
@router.post("/token", include_in_schema=False)
def login_form(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 password-flow form, so the Authorize button in /docs works."""
    return login(LoginRequest(email=form.username, password=form.password), db)


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return _public(user)

#use the given access token to retrieve the current user's information 
@router.get("/users")
def list_users(_: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    return {"users": [_public(u) for u in db.query(User).order_by(User.created_at).all()]}


@router.patch("/users/{user_id}/role")
def set_role(user_id: str, payload: RoleUpdate,
             admin: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    if payload.role not in ALLOWED_ROLES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Role must be one of: {', '.join(sorted(ALLOWED_ROLES))}")
    import uuid as _uuid

    try:
        target = db.get(User, _uuid.UUID(user_id))
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found") from exc
    if not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if target.id == admin.id and payload.role != "admin":
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "You cannot remove your own admin role")

    target.role = payload.role
    db.commit()
    return _public(target)
