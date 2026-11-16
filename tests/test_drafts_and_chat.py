import pytest
from fastapi.testclient import TestClient
from app.main import app

HEADERS_DEV = {
    "x-mock-user-id": "usr-nader",
    "x-mock-role": "Developer",
    "x-mock-project-id": "proj-gillo-core"
}

HEADERS_TL = {
    "x-mock-user-id": "usr-nader",
    "x-mock-role": "TL",
    "x-mock-project-id": "proj-gillo-core"
}


def test_drafts_and_chat_lifecycle():
    with TestClient(app) as client:
        # 1. Feature 5: Personal Standup Draft
        res_su = client.post("/standup/draft", headers=HEADERS_DEV)
        assert res_su.status_code == 200, f"Standup failed: {res_su.text}"
        data_su = res_su.json()
        assert data_su["kind"] == "standup"
        assert "yesterday" in data_su["content"]
        assert "today" in data_su["content"]
        assert "blockers" in data_su["content"]

        # 2. Feature 5: Sprint Retrospective Draft
        res_retro = client.post("/sprints/sprint-04/retro/draft", headers=HEADERS_TL)
        assert res_retro.status_code == 200, f"Retro failed: {res_retro.text}"
        data_retro = res_retro.json()
        assert data_retro["kind"] == "retro"
        assert "what_went_well" in data_retro["content"]
        assert "what_needs_improvement" in data_retro["content"]
        assert "action_items" in data_retro["content"]

        # 3. Feature 3: Release Documentation Draft
        res_rel = client.post("/release-requests/rel-v1.0/draft", headers=HEADERS_TL)
        assert res_rel.status_code == 200, f"Release draft failed: {res_rel.text}"
        data_rel = res_rel.json()
        assert data_rel["kind"] == "release_summary"
        assert "rollback_plan" in data_rel["content"]
        assert len(data_rel["content"]["rollback_plan"]) >= 1

        # 4. Feature 3: Release Blocker Diagnostics
        res_blk = client.post("/release-requests/rel-v1.1/explain-block", headers=HEADERS_TL)
        assert res_blk.status_code == 200, f"Block explainer failed: {res_blk.text}"
        data_blk = res_blk.json()
        assert data_blk["kind"] == "block_explanation"
        assert "diagnostics" in data_blk["content"]
        assert "recommendations" in data_blk["content"]

        # 5. Feature 6: Read-only Chat with Tool Calling and Sources
        res_chat = client.post(
            "/chat",
            json={"query": "What are my tasks and how is the sprint going?"},
            headers=HEADERS_DEV
        )
        assert res_chat.status_code == 200, f"Chat failed: {res_chat.text}"
        data_chat = res_chat.json()
        assert "answer" in data_chat
        assert len(data_chat["answer"]) > 10
        assert "sources" in data_chat
        assert len(data_chat["sources"]) >= 1
