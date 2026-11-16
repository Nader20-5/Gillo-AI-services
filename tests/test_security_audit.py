import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from jose import jwt
from bson import ObjectId

from app.main import app
from app.config import Settings, settings
from app.database import sanitize_mongo_uri
from tests.conftest import HEADERS_DEV, HEADERS_TL, HEADERS_PM, HEADERS_ADMIN


# ===========================================================================
# 1) Auth Hardening Tests
# ===========================================================================

def test_mock_auth_default_disabled():
    """Verify ALLOW_MOCK_AUTH defaults to False."""
    clean_settings = Settings(
        JWT_SECRET="some-safe-dummy-secret-key-for-test-32chars",
        ENVIRONMENT="development"
    )
    assert clean_settings.ALLOW_MOCK_AUTH is False


def test_production_refuses_mock_auth():
    """Verify that Settings refuses to initialize in production with ALLOW_MOCK_AUTH=True."""
    with pytest.raises(ValueError, match="ALLOW_MOCK_AUTH must be False in production"):
        Settings(
            ENVIRONMENT="production",
            ALLOW_MOCK_AUTH=True,
            JWT_SECRET="a-very-strong-production-secret-with-more-than-32-chars"
        )


def test_production_refuses_default_or_short_jwt_secret():
    """Verify production rejects default secrets or secrets shorter than 32 chars."""
    with pytest.raises(ValueError, match="secure 'JWT_SECRET'"):
        Settings(
            ENVIRONMENT="production",
            JWT_SECRET="gillo_super_secret_dev_jwt_key_2026",
            ALLOW_MOCK_AUTH=False
        )

    with pytest.raises(ValueError, match="at least 32 characters"):
        Settings(
            ENVIRONMENT="production",
            JWT_SECRET="short-secret-123",
            ALLOW_MOCK_AUTH=False
        )


def test_jwt_algorithm_pinning():
    """Verify that insecure algorithm 'none' or unapproved algorithms are rejected."""
    with pytest.raises(ValueError, match="Insecure or unsupported JWT_ALGORITHM"):
        Settings(
            JWT_ALGORITHM="none",
            JWT_SECRET="some-safe-dummy-secret-key-for-test-32chars"
        )

    with pytest.raises(ValueError, match="Insecure or unsupported JWT_ALGORITHM"):
        Settings(
            JWT_ALGORITHM="DES",
            JWT_SECRET="some-safe-dummy-secret-key-for-test-32chars"
        )


def test_mock_auth_forbidden_in_production_runtime(monkeypatch):
    """Even if ALLOW_MOCK_AUTH is toggled, runtime rejects mock auth in production."""
    client = TestClient(app)
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "ALLOW_MOCK_AUTH", True)

    res = client.get("/suggestions", headers=HEADERS_TL)
    assert res.status_code == 401
    assert "production" in res.json().get("detail", "").lower()


def test_jwt_requires_exp_claim(monkeypatch):
    """Verify that JWT missing 'exp' claim is rejected with HTTP 401."""
    client = TestClient(app)
    payload_no_exp = {
        "sub": "usr-nader",
        "role": "TL",
        "project_id": "proj-gillo-core"
    }
    token = jwt.encode(payload_no_exp, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    res = client.get("/suggestions", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 401
    assert "invalid or expired jwt token" in res.json().get("detail", "").lower()


# ===========================================================================
# 2) Project-Scoping Authorization Tests
# ===========================================================================

FOREIGN_TL_HEADERS = {
    "x-mock-user-id": "usr-foreign-lead",
    "x-mock-role": "TL",
    "x-mock-project-id": "proj-foreign-unauthorized"
}


def test_project_scoping_sprint_estimate(setup_test_db):
    """Verify estimate endpoint blocks cross-project access with HTTP 403."""
    client = TestClient(app)
    res = client.post("/sprints/sprint-04/estimate", headers=FOREIGN_TL_HEADERS)
    assert res.status_code == 403
    assert "access denied" in res.json().get("detail", "").lower()


def test_project_scoping_sprint_fit_check(setup_test_db):
    """Verify fit-check endpoint blocks cross-project access with HTTP 403."""
    client = TestClient(app)
    res = client.get("/sprints/sprint-04/fit-check", headers=FOREIGN_TL_HEADERS)
    assert res.status_code == 403
    assert "access denied" in res.json().get("detail", "").lower()


def test_project_scoping_sprint_scan_risks(setup_test_db):
    """Verify scan-risks endpoint blocks cross-project access with HTTP 403."""
    client = TestClient(app)
    res = client.post("/sprints/sprint-04/scan-risks", headers=FOREIGN_TL_HEADERS)
    assert res.status_code == 403
    assert "access denied" in res.json().get("detail", "").lower()


@pytest.mark.asyncio
async def test_project_scoping_suggestions_mutations(setup_test_db):
    """Verify suggestion decisions and comments block cross-project actors with HTTP 403."""
    client = TestClient(app)
    db = setup_test_db

    # Create a suggestion in proj-gillo-core
    sug_doc = {
        "project_id": "proj-gillo-core",
        "sprint_id": "sprint-04",
        "task_id": "task-act-01",
        "type": "sprint_fix",
        "status": "pending",
        "apply_status": "none",
        "dedupe_key": "audit_scope_test_1",
        "created_at": datetime.now(timezone.utc)
    }
    insert_res = await db.ai_suggestions.insert_one(sug_doc)
    sug_id = str(insert_res.inserted_id)

    # 1. Foreign TL attempts to accept -> 403
    res_accept = client.post(f"/suggestions/{sug_id}/accept", headers=FOREIGN_TL_HEADERS)
    assert res_accept.status_code == 403
    assert "access denied" in res_accept.json().get("detail", "").lower()

    # 2. Foreign TL attempts to reject -> 403
    res_reject = client.post(
        f"/suggestions/{sug_id}/reject",
        json={"decision_reason": "Cross-project unauthorized rejection"},
        headers=FOREIGN_TL_HEADERS
    )
    assert res_reject.status_code == 403
    assert "access denied" in res_reject.json().get("detail", "").lower()

    # 3. Foreign TL attempts to dismiss -> 403
    res_dismiss = client.post(
        f"/suggestions/{sug_id}/dismiss",
        json={"decision_reason": "Cross-project unauthorized dismissal"},
        headers=FOREIGN_TL_HEADERS
    )
    assert res_dismiss.status_code == 403
    assert "access denied" in res_dismiss.json().get("detail", "").lower()

    # 4. Foreign TL attempts to add comment -> 403
    res_comment = client.post(
        f"/suggestions/{sug_id}/comments",
        json={"body": "Cross-project malicious comment"},
        headers=FOREIGN_TL_HEADERS
    )
    assert res_comment.status_code == 403
    assert "access denied" in res_comment.json().get("detail", "").lower()

    # Verify authorized TL in proj-gillo-core can comment successfully
    res_valid_comment = client.post(
        f"/suggestions/{sug_id}/comments",
        json={"body": "Authorized team lead comment"},
        headers=HEADERS_TL
    )
    assert res_valid_comment.status_code == 200


# ===========================================================================
# 3) Input Validation & DoS Guardrail Tests
# ===========================================================================

def test_suggestions_sort_by_allowlist(setup_test_db):
    """Verify sort_by is strictly validated against allowed fields."""
    client = TestClient(app)

    # Allowed sort fields succeed
    for field in ["created_at", "severity", "status", "type", "confidence"]:
        res = client.get(f"/suggestions?sort_by={field}", headers=HEADERS_TL)
        assert res.status_code == 200

    # Unapproved or malicious field returns 422
    res_bad = client.get("/suggestions?sort_by=password_hash", headers=HEADERS_TL)
    assert res_bad.status_code == 422
    assert "invalid sort_by field" in res_bad.json().get("detail", "").lower()


def test_oversized_chat_query_rejected(setup_test_db):
    """Verify that oversized chat query (>2000 chars) is rejected with 422."""
    client = TestClient(app)
    huge_query = "A" * 2500
    res = client.post("/chat", json={"query": huge_query}, headers=HEADERS_DEV)
    assert res.status_code == 422


def test_oversized_suggestion_comment_rejected(setup_test_db):
    """Verify that oversized comment body (>2000 chars) is rejected with 422."""
    client = TestClient(app)
    fake_id = str(ObjectId())
    huge_body = "C" * 2500
    res = client.post(f"/suggestions/{fake_id}/comments", json={"body": huge_body}, headers=HEADERS_TL)
    assert res.status_code == 422


# ===========================================================================
# 4) Info Leak & Sensitive Logging Tests
# ===========================================================================

def test_sanitize_mongo_uri():
    """Verify credentials in MongoDB URIs are masked before logging."""
    raw_uri = "mongodb://dbadmin:SuperSecretPass123@cluster0.prod.internal:27017/gillo_ai?authSource=admin"
    sanitized = sanitize_mongo_uri(raw_uri)
    assert "SuperSecretPass123" not in sanitized
    assert "dbadmin:****@" in sanitized

    plain_uri = "mongodb://localhost:27017/gillo_ai"
    assert sanitize_mongo_uri(plain_uri) == plain_uri


def test_health_check_no_internal_exception_leak(monkeypatch):
    """Verify health check returns 'unreachable' without exposing raw exception text."""
    from app.database import db

    class FailingClient:
        @property
        def admin(self):
            class FailingAdmin:
                async def command(self, *args, **kwargs):
                    raise ConnectionResetError("Internal socket 10.244.0.15:27017 auth token dropped")
            return FailingAdmin()

    monkeypatch.setattr(db, "client", FailingClient())
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["database"] == "unreachable"
    assert "10.244.0.15" not in res.text
    assert "auth token" not in res.text


# ===========================================================================
# 5) Follow-up: Secret Pattern Hygiene Test
# ===========================================================================

def test_no_secret_patterns_in_repo():
    """
    Fails if any file under the repository (excluding .venv and .git)
    contains common API key patterns:
    - Groq API keys: gsk_[A-Za-z0-9]{20,}
    - OpenAI / standard API keys: sk-[A-Za-z0-9]{20,}
    - AWS Access Keys: AKIA[0-9A-Z]{16}
    """
    import os
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    patterns = [
        re.compile(r"gsk_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}")
    ]

    excluded_dirs = {".venv", ".git", "__pycache__", ".pytest_cache"}
    violating_files = []

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for f in files:
            if f.endswith((".pkl", ".parquet", ".bin", ".joblib", ".pyc")):
                continue
            # Skip test itself to avoid self-match on regex pattern literals
            if f == "test_security_audit.py":
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    for line_idx, line in enumerate(fp, 1):
                        for pat in patterns:
                            if pat.search(line):
                                violating_files.append(f"{fpath}:{line_idx}")
            except Exception:
                pass

    assert not violating_files, f"Found unredacted secret patterns in repository: {violating_files}"


# ===========================================================================
# 6) Follow-up: Scoping Fail-Closed Tests
# ===========================================================================

def test_scoping_fail_closed_token_without_project_id(setup_test_db):
    """
    Non-admin token without project_id must be rejected with 403 on every
    sprint, suggestion, chat, and draft endpoint, unless explicit 'all_projects': true claim is set.
    """
    client = TestClient(app)
    now = datetime.now(timezone.utc)

    # 1. Token without project_id
    payload_no_project = {
        "sub": "usr-nader",
        "role": "TL",
        "exp": now + timedelta(hours=2)
    }
    token_no_project = jwt.encode(payload_no_project, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    headers_no_project = {"Authorization": f"Bearer {token_no_project}"}

    endpoints = [
        ("GET", "/suggestions"),
        ("POST", "/sprints/sprint-04/estimate", None),
        ("GET", "/sprints/sprint-04/fit-check", None),
        ("POST", "/sprints/sprint-04/scan-risks", None),
        ("POST", "/standup/draft", None),
        ("POST", "/sprints/sprint-04/retro/draft", None),
        ("POST", "/release-requests/rel-v1.0/draft", None),
        ("POST", "/chat", {"query": "Status"}),
    ]

    for method, url, *body in endpoints:
        json_body = body[0] if body else None
        if method == "GET":
            res = client.get(url, headers=headers_no_project)
        else:
            res = client.post(url, json=json_body, headers=headers_no_project)
        assert res.status_code == 403, f"Expected 403 for missing project_id on {method} {url}, got {res.status_code}"
        assert "project_id" in res.json().get("detail", "").lower() or "all_projects" in res.json().get("detail", "").lower()

    # 2. Token with explicit all_projects: true is allowed
    payload_all_projects = {
        "sub": "usr-nader",
        "role": "TL",
        "all_projects": True,
        "exp": now + timedelta(hours=2)
    }
    token_all_projects = jwt.encode(payload_all_projects, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    headers_all_projects = {"Authorization": f"Bearer {token_all_projects}"}

    res_all_sug = client.get("/suggestions", headers=headers_all_projects)
    assert res_all_sug.status_code == 200

    res_all_sprint = client.get("/sprints/sprint-04/fit-check", headers=headers_all_projects)
    assert res_all_sprint.status_code == 200

    # 3. Admin remains blocked everywhere even with all_projects: true and project_id
    admin_payload = {
        "sub": "usr-admin",
        "role": "Admin",
        "project_id": "proj-gillo-core",
        "all_projects": True,
        "exp": now + timedelta(hours=2)
    }
    token_admin = jwt.encode(admin_payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    headers_admin = {"Authorization": f"Bearer {token_admin}"}

    for method, url, *body in endpoints:
        json_body = body[0] if body else None
        res = client.get(url, headers=headers_admin) if method == "GET" else client.post(url, json=json_body, headers=headers_admin)
        assert res.status_code == 403, f"Admin was not blocked with 403 on {method} {url}"


# ===========================================================================
# 7) Follow-up: Rate Limiting & Forged Token Test
# ===========================================================================

def test_forged_unsigned_token_cannot_consume_other_user_rate_limit(monkeypatch):
    """
    Verify that check_llm_rate_limit does not trust unverified JWT claims.
    A forged token claiming sub='usr-victim' with an invalid signature must not key on usr-victim.
    """
    from starlette.requests import Request
    from app.middleware.rate_limit import check_llm_rate_limit, _request_timestamps

    _request_timestamps.clear()

    # Create a forged token claiming sub="usr-victim" signed with a completely different secret
    forged_token = jwt.encode({"sub": "usr-victim"}, "attacker-untrusted-secret", algorithm="HS256")

    # Scope with request headers
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": [
            (b"authorization", f"Bearer {forged_token}".encode()),
            (b"host", b"testserver"),
        ],
        "client": ("192.168.1.100", 54321),
    }
    req = Request(scope)

    # Calling rate limit dependency should key on client IP ('ip:192.168.1.100'), NOT on 'user:usr-victim'
    check_llm_rate_limit(req)

    assert "user:usr-victim" not in _request_timestamps
    assert "ip:192.168.1.100" in _request_timestamps
    assert len(_request_timestamps["ip:192.168.1.100"]) == 1


# ===========================================================================
# 8) Follow-up: CORS Configuration Tests
# ===========================================================================

def test_cors_configured_origins():
    """Verify configured CORS origins are honored while unauthorized origins are not reflected."""
    client = TestClient(app)

    # Allowed origin http://localhost:3000
    res_allowed = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert res_allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"

    # Disallowed origin http://unauthorized-attacker.com
    res_disallowed = client.options(
        "/health",
        headers={
            "Origin": "http://unauthorized-attacker.com",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert res_disallowed.headers.get("access-control-allow-origin") != "http://unauthorized-attacker.com"

