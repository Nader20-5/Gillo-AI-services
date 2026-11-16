import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.conftest import HEADERS_ADMIN, HEADERS_DEV, HEADERS_TL, HEADERS_PM


def test_admin_blocked_on_all_ai_endpoints(setup_test_db):
    """
    Core Security Guardrail:
    Admin role must receive HTTP 403 Forbidden on all AI endpoints.
    """
    client = TestClient(app)

    endpoints = [
        ("GET", "/suggestions"),
        ("POST", "/sprints/sprint-04/estimate"),
        ("GET", "/sprints/sprint-04/fit-check"),
        ("POST", "/sprints/sprint-04/scan-risks"),
        ("POST", "/standup/draft"),
        ("POST", "/sprints/sprint-04/retro/draft"),
        ("POST", "/release-requests/rel-v1.0/draft"),
        ("POST", "/release-requests/rel-v1.1/explain-block"),
        ("POST", "/chat", {"query": "Hello"}),
    ]

    for item in endpoints:
        method = item[0]
        url = item[1]
        json_body = item[2] if len(item) > 2 else None

        if method == "GET":
            res = client.get(url, headers=HEADERS_ADMIN)
        else:
            res = client.post(url, json=json_body, headers=HEADERS_ADMIN)

        assert res.status_code == 403, f"Admin was not blocked on {method} {url} (status: {res.status_code})"
        assert "forbidden" in res.json().get("detail", "").lower() or "denied" in res.json().get("detail", "").lower()


def test_developer_role_matrix(setup_test_db):
    """
    Developers are restricted to personal standup drafts and chat.
    Management endpoints (Inbox, Sprint Planning, Releases) must return 403.
    """
    client = TestClient(app)

    # 1. Management endpoints blocked for Developer
    blocked_for_dev = [
        ("GET", "/suggestions"),
        ("POST", "/sprints/sprint-04/estimate"),
        ("GET", "/sprints/sprint-04/fit-check"),
        ("POST", "/sprints/sprint-04/scan-risks"),
        ("POST", "/sprints/sprint-04/retro/draft"),
        ("POST", "/release-requests/rel-v1.0/draft"),
    ]
    for method, url in blocked_for_dev:
        res = client.get(url, headers=HEADERS_DEV) if method == "GET" else client.post(url, headers=HEADERS_DEV)
        assert res.status_code == 403, f"Developer was not blocked on {method} {url}"

    # 2. Permitted endpoints for Developer
    res_standup = client.post("/standup/draft", headers=HEADERS_DEV)
    assert res_standup.status_code == 200, res_standup.text

    res_chat = client.post("/chat", json={"query": "What are my tasks?"}, headers=HEADERS_DEV)
    assert res_chat.status_code == 200, res_chat.text


def test_pm_tl_role_matrix_and_project_scoping(setup_test_db):
    """
    PM and TL are authorized for management endpoints, but strictly scoped to their project.
    """
    client = TestClient(app)

    # Allowed in their project
    res_sug = client.get("/suggestions", headers=HEADERS_PM)
    assert res_sug.status_code == 200

    # Cross-project access attempt must be rejected with 403
    foreign_headers = {
        "x-mock-user-id": "usr-alex",
        "x-mock-role": "PM",
        "x-mock-project-id": "proj-foreign-unauthorized"
    }
    res_cross = client.get("/suggestions?project_id=proj-gillo-core", headers=foreign_headers)
    assert res_cross.status_code == 403
