"""Authentication and access-control tests (proposal Sec.19).

These need a live database, so they are skipped when DATABASE_URL is unset or
unreachable. Each run uses unique emails so it can be repeated without a reset.
"""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Skips only if the database is genuinely unreachable.

    The connection string is read the same way the application reads it - from
    .env via app.config - rather than from an OS environment variable, so these
    tests run on a normal local setup instead of silently skipping.
    """
    from sqlalchemy import text

    from app.database import engine
    from app.main import app

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database not reachable: {exc}")
    return TestClient(app)


def _register(client, password="supersecret1", **extra):
    email = f"user-{uuid.uuid4().hex[:10]}@example.com"
    r = client.post("/api/auth/register",
                    json={"email": email, "password": password,
                          "full_name": "Test User", **extra})
    assert r.status_code == 201, r.text
    return email, password, r.json()


def _login(client, email, password):
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------ register
def test_register_login_and_me(client):
    email, password, created = _register(client)
    assert created["email"] == email
    assert created["role"] in {"admin", "analyst"}
    assert "hashed_password" not in created

    token = _login(client, email, password)
    me = client.get("/api/auth/me", headers=_auth(token))
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_duplicate_email_is_rejected(client):
    email, password, _ = _register(client)
    again = client.post("/api/auth/register",
                        json={"email": email, "password": password})
    assert again.status_code == 409


def test_wrong_password_is_rejected(client):
    email, _, _ = _register(client)
    r = client.post("/api/auth/login", json={"email": email, "password": "not-the-password"})
    assert r.status_code == 401
    # the message must not reveal whether the account exists
    assert "email or password" in r.json()["detail"].lower()


def test_short_password_is_rejected(client):
    r = client.post("/api/auth/register",
                    json={"email": f"x-{uuid.uuid4().hex[:8]}@example.com", "password": "short"})
    assert r.status_code == 422


# ----------------------------------------------------------------------- 401
@pytest.mark.parametrize("method,path", [
    ("get", "/api/datasets"),
    ("get", "/api/sources"),
    ("get", "/api/investigations"),
    ("get", "/api/tools"),
    ("get", "/api/auth/me"),
])
def test_protected_routes_reject_anonymous_requests(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_a_garbage_token_is_rejected(client):
    r = client.get("/api/auth/me", headers=_auth("not-a-real-token"))
    assert r.status_code == 401


def test_health_stays_public(client):
    """Health checks must work before anyone has an account."""
    assert client.get("/health").status_code == 200


# ----------------------------------------------------------------------- 403
def _upload_dataset(client, token) -> str:
    csv = "date,region,revenue\n2024-01-01,North,100\n2024-02-01,South,80\n"
    r = client.post("/api/sources/upload",
                    files={"file": ("owned.csv", io.BytesIO(csv.encode()), "text/csv")},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


def test_non_owner_gets_403_on_someone_elses_dataset(client):
    owner_email, owner_pw, _ = _register(client)
    owner_token = _login(client, owner_email, owner_pw)
    dataset_id = _upload_dataset(client, owner_token)

    other_email, other_pw, _ = _register(client)
    other_token = _login(client, other_email, other_pw)

    for path in [
        f"/api/datasets/{dataset_id}",
        f"/api/datasets/{dataset_id}/profile",
        f"/api/datasets/{dataset_id}/health",
        f"/api/datasets/{dataset_id}/cleaning-plan",
    ]:
        r = client.get(path, headers=_auth(other_token))
        assert r.status_code == 403, f"{path} returned {r.status_code}"

    # and the owner still gets through
    assert client.get(f"/api/datasets/{dataset_id}",
                      headers=_auth(owner_token)).status_code == 200


def test_analytics_routes_also_enforce_dataset_ownership(client):
    owner_email, owner_pw, _ = _register(client)
    owner_token = _login(client, owner_email, owner_pw)
    dataset_id = _upload_dataset(client, owner_token)

    other_email, other_pw, _ = _register(client)
    other_token = _login(client, other_email, other_pw)

    r = client.post(f"/api/datasets/{dataset_id}/analyze",
                    json={"operation": "describe", "params": {}},
                    headers=_auth(other_token))
    assert r.status_code == 403

    r = client.post(f"/api/datasets/{dataset_id}/query-sql",
                    json={"query": "SELECT * FROM data"},
                    headers=_auth(other_token))
    assert r.status_code == 403


def test_owner_only_sees_their_own_datasets_in_the_list(client):
    owner_email, owner_pw, _ = _register(client)
    owner_token = _login(client, owner_email, owner_pw)
    dataset_id = _upload_dataset(client, owner_token)

    other_email, other_pw, _ = _register(client)
    other_token = _login(client, other_email, other_pw)

    visible = {d["id"] for d in
               client.get("/api/datasets", headers=_auth(other_token)).json()["datasets"]}
    assert dataset_id not in visible

    mine = {d["id"] for d in
            client.get("/api/datasets", headers=_auth(owner_token)).json()["datasets"]}
    assert dataset_id in mine


def test_missing_dataset_is_404_not_403(client):
    email, pw, _ = _register(client)
    token = _login(client, email, pw)
    r = client.get(f"/api/datasets/{uuid.uuid4()}", headers=_auth(token))
    assert r.status_code == 404


# ---------------------------------------------------------------- role gates
def test_admin_only_routes_reject_an_analyst(client):
    email, pw, created = _register(client)
    if created["role"] == "admin":
        pytest.skip("this run created the first account, which is admin by design")
    token = _login(client, email, pw)
    assert client.get("/api/auth/users", headers=_auth(token)).status_code == 403