import pytest
from fastapi.testclient import TestClient
from app.main import app

HEADERS_TL = {
    "x-mock-user-id": "usr-nader",
    "x-mock-role": "TL",
    "x-mock-project-id": "proj-gillo-core"
}

HEADERS_ADMIN = {
    "x-mock-user-id": "usr-admin",
    "x-mock-role": "Admin",
    "x-mock-project-id": "proj-gillo-core"
}


def test_sprint_intelligence_and_risk_scanner_pipeline():
    """
    Integration test testing the complete Sprint Intelligence lifecycle:
    1. Model 1 Effort Estimation on Sprint 4 tasks
    2. Capacity Fit Check (fits, tight, over) and auto-generating proposals
    3. Model 2 + SHAP Early Risk Scanner identifying slipping tasks
    4. Admin Block Guardrail enforcement (403 Forbidden)
    """
    with TestClient(app) as client:
        # 1. Feature 1: Model 1 Task Effort Estimation
        res_est = client.post("/sprints/sprint-04/estimate", headers=HEADERS_TL)
        assert res_est.status_code == 200, f"Estimate failed: {res_est.text}"
        data_est = res_est.json()
        assert data_est["sprint_id"] == "sprint-04"
        assert data_est["tasks_estimated"] >= 1
        assert data_est["total_low_hours"] > 0
        assert data_est["total_high_hours"] >= data_est["total_low_hours"]
        assert len(data_est["estimates"]) >= 1

        # 2. Feature 1: Capacity Fit Check
        res_fit = client.get("/sprints/sprint-04/fit-check", headers=HEADERS_TL)
        assert res_fit.status_code == 200, f"Fit check failed: {res_fit.text}"
        data_fit = res_fit.json()
        assert data_fit["sprint_id"] == "sprint-04"
        assert data_fit["verdict"] in ["fits", "tight", "over"]
        assert "per_user" in data_fit

        # 3. Feature 2: Model 2 + SHAP Early Risk Warning Scan
        res_risk = client.post("/sprints/sprint-04/scan-risks", headers=HEADERS_TL)
        assert res_risk.status_code == 200, f"Risk scan failed: {res_risk.text}"
        data_risk = res_risk.json()
        assert data_risk["sprint_id"] == "sprint-04"
        assert "active_tasks_scanned" in data_risk
        assert data_risk["active_tasks_scanned"] >= 1

        # 4. Ethical Guardrail: Admin blocked from sprint planning
        res_admin = client.post("/sprints/sprint-04/estimate", headers=HEADERS_ADMIN)
        assert res_admin.status_code == 403
        assert "strictly forbidden" in res_admin.json()["detail"].lower()
