"""
Demo Runner for Taskove AI Service.

Seeds an in-memory test database, calls live FastAPI endpoints through TestClient
as PM, saves raw responses to reports/demo_outputs.json, and prints compact tables
from that file only:
- POST /sprints/{id}/estimate
- POST /sprints/{id}/scan-risks
For: sprint-04, sprint-05, sprint-small-active, sprint-early-active.
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mongomock_motor import AsyncMongoMockClient
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_database, db, init_indexes
from app.config import settings
from scripts.seed_data import seed_data, ensure_utc


HEADERS_PM = {
    "x-mock-user-id": "usr-alex",
    "x-mock-role": "PM",
    "x-mock-project-id": "proj-gillo-core"
}

SPRINT_IDS = [
    "sprint-04",
    "sprint-05",
    "sprint-small-active",
    "sprint-early-active"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Taskove Demo Runner & Endpoint Output Capture")
    parser.add_argument(
        "--now",
        type=str,
        default="2026-11-14T12:00:00Z",
        help="ISO UTC timestamp to use as reference now (e.g. 2026-11-14T12:00:00Z). Default: 2026-11-14T12:00:00Z."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join(ROOT_DIR, "reports", "demo_outputs.json"),
        help="Path to save full raw JSON responses."
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default=os.path.join(ROOT_DIR, "reports", "demo_table.txt"),
        help="Path to save formatted compact tables."
    )
    return parser.parse_args()


async def initialize_and_seed_db(ref_now: datetime) -> AsyncMongoMockClient:
    """Sets up an in-memory MongoMock database and seeds demo data."""
    mock_client = AsyncMongoMockClient()
    mock_database = mock_client["gillo_ai_demo"]

    db.client = mock_client
    db.db = mock_database
    await init_indexes()
    app.dependency_overrides[get_database] = lambda: mock_database
    settings.ALLOW_MOCK_AUTH = True

    await seed_data(mock_database, now=ref_now, reset=True)
    return mock_client


def execute_endpoints(ref_now: datetime) -> Dict[str, Any]:
    """Calls estimate and scan-risks endpoints for the target sprints using TestClient."""
    client = TestClient(app)
    results: Dict[str, Any] = {
        "metadata": {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "reference_now": ref_now.isoformat(),
            "target_sprints": SPRINT_IDS
        },
        "sprints": {}
    }

    for sp_id in SPRINT_IDS:
        print(f"[demo_run] Calling POST /sprints/{sp_id}/estimate ...")
        est_res = client.post(f"/sprints/{sp_id}/estimate", headers=HEADERS_PM)
        if est_res.status_code != 200:
            raise RuntimeError(f"Estimate failed for {sp_id}: {est_res.status_code} {est_res.text}")
        est_data = est_res.json()

        print(f"[demo_run] Calling POST /sprints/{sp_id}/scan-risks ...")
        scan_res = client.post(f"/sprints/{sp_id}/scan-risks", headers=HEADERS_PM)
        if scan_res.status_code != 200:
            raise RuntimeError(f"Scan-risks failed for {sp_id}: {scan_res.status_code} {scan_res.text}")
        scan_data = scan_res.json()

        # Strict check: NO raw probability percentage in user-facing client response
        for st in scan_data.get("scored_tasks", []):
            assert "probability" not in st, f"Leaked 'probability' in {st}"
            assert "raw_probability" not in st, f"Leaked 'raw_probability' in {st}"
            assert "calibrated_probability" not in st, f"Leaked 'calibrated_probability' in {st}"
            assert "_telemetry_calibrated_prob" not in st, f"Leaked '_telemetry_calibrated_prob' in {st}"

        results["sprints"][sp_id] = {
            "estimate": est_data,
            "scan_risks": scan_data
        }

    return results


def format_tables_from_json(json_path: str) -> str:
    """Reads raw JSON from disk only and prints compact formatted tables."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines: List[str] = []
    lines.append("=" * 115)
    lines.append(f"REAL CAPTURED OUTPUTS FROM {os.path.basename(json_path)}")
    lines.append(f"Reference Now: {data['metadata']['reference_now']} | Captured: {data['metadata']['captured_at']}")
    lines.append("=" * 115)
    lines.append("")

    # 1. Estimation Table
    lines.append("TABLE 1: STORY POINT ESTIMATION (POST /sprints/{id}/estimate)")
    lines.append("-" * 115)
    header_est = f"{'Sprint':<14} | {'Task ID':<13} | {'Type':<10} | {'Seeded':<6} | {'Raw SP':<8} | {'Snapped':<7} | {'Interval':<12} | {'Method':<20} | {'Fallback Reason'}"
    lines.append(header_est)
    lines.append("-" * 115)

    for sp_id in SPRINT_IDS:
        sp_data = data["sprints"].get(sp_id, {})
        est = sp_data.get("estimate", {})
        estimates = est.get("estimates", [])
        for e in estimates:
            tid = e.get("task_id", "")
            t_type = e.get("task_type", "feature")
            orig_sp = str(e.get("original_story_points") if e.get("original_story_points") is not None else "-")
            raw_sp = f"{e.get('raw_story_points', 0.0):.2f}" if e.get("raw_story_points") is not None else "-"
            snapped = str(e.get("story_points", "-"))
            interval = str(e.get("story_points_interval", "-"))
            method = e.get("sp_method", "-")
            reason = str(e.get("sp_fallback_reason") or "None")
            lines.append(f"{sp_id:<14} | {tid:<13} | {t_type:<10} | {orig_sp:<6} | {raw_sp:<8} | {snapped:<7} | {interval:<12} | {method:<20} | {reason}")
        if estimates:
            lines.append("-" * 115)

    lines.append("")
    lines.append("TABLE 2: SPRINT CARRY-OVER RISK SCAN (POST /sprints/{id}/scan-risks)")
    lines.append("-" * 115)
    header_scan = f"{'Sprint':<14} | {'Rank':<4} | {'Task ID':<14} | {'Tier':<6} | {'Method':<20} | {'Model Version':<22} | {'Top Contributing Factors'}"
    lines.append(header_scan)
    lines.append("-" * 115)

    for sp_id in SPRINT_IDS:
        sp_data = data["sprints"].get(sp_id, {})
        scan = sp_data.get("scan_risks", {})
        method_summary = scan.get("method", "")
        fallback_msg = scan.get("fallback_reason")
        k_val = scan.get("k_threshold", 0)
        n_open = scan.get("active_tasks_scanned", 0)

        lines.append(f"[{sp_id}] Active Tasks: {n_open}, k threshold: {k_val}, Method: {method_summary}" + (f" (Fallback: {fallback_msg})" if fallback_msg else ""))
        scored = scan.get("scored_tasks", [])
        for st in scored:
            rank = str(st.get("risk_rank", "-"))
            tid = st.get("task_id", "")
            tier = st.get("risk_tier", "-")
            method = st.get("method", "-")
            version = st.get("model_version", "-")
            factors_raw = st.get("top_contributing_factors", [])
            if factors_raw and isinstance(factors_raw[0], dict):
                factors = "; ".join(f"{f['plain_text']}" for f in factors_raw)
            elif factors_raw:
                factors = ", ".join(str(f) for f in factors_raw)
            else:
                factors = st.get("factor_note") or st.get("note") or "No specific driver stood out"
            lines.append(f"{sp_id:<14} | {rank:<4} | {tid:<14} | {tier:<6} | {method:<20} | {version:<22} | {factors}")
        lines.append("-" * 115)

    return "\n".join(lines)


def main():
    args = parse_args()
    ref_now_str = args.now.replace("Z", "+00:00")
    ref_now = datetime.fromisoformat(ref_now_str)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_table), exist_ok=True)

    print(f"[demo_run] Initializing in-memory DB and seeding data relative to {ref_now.isoformat()} ...")
    asyncio.run(initialize_and_seed_db(ref_now))

    print(f"[demo_run] Executing FastAPI endpoints as PM ...")
    raw_results = execute_endpoints(ref_now)

    print(f"[demo_run] Writing raw JSON response dump to {args.output_json} ...")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)
    print(f"  - Saved {os.path.getsize(args.output_json)} bytes to {args.output_json}")

    print(f"[demo_run] Formatting compact table strictly from {args.output_json} ...")
    table_text = format_tables_from_json(args.output_json)

    with open(args.output_table, "w", encoding="utf-8") as f:
        f.write(table_text)
    print(f"  - Saved table to {args.output_table}\n")

    print(table_text)


if __name__ == "__main__":
    main()
