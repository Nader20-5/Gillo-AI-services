from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.database import get_database
from app.models.drafts import AIDraft, AIRun
from app.services.llm import AgileDraftService
from app.middleware.auth import CurrentUser, get_current_user, require_roles
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agile Drafts & Release Assistant (Features 3 & 5)"])

auth_pm_or_tl = require_roles(["PM", "TL"])
auth_dev_or_lead = require_roles(["Developer", "PM", "TL"])


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def serialize_doc(doc: dict) -> dict:
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc


# ---------------------------------------------------------------------------
# Feature 5: Personal Standup Draft
# ---------------------------------------------------------------------------
@router.post("/standup/draft", response_model=dict)
async def generate_standup_draft(
    current_user: CurrentUser = Depends(auth_dev_or_lead),
    db=Depends(get_database),
):
    """
    Feature 5: Generates a personal daily standup draft for the authenticated developer.
    Enforces that developers only generate drafts for themselves.
    Admin gets 403.
    """
    user_id = current_user.user_id
    project_id = current_user.project_id or "proj-gillo-core"

    # 1. Fetch user tasks
    tasks_cursor = db.task_snapshots.find({"assignee_id": user_id})
    user_tasks = await tasks_cursor.to_list(length=50)

    # 2. Fetch logged hours
    time_cursor = db.time_entries.find({"user_id": user_id})
    time_entries = await time_cursor.to_list(length=50)
    logged_h = sum(te.get("hours", 0.0) for te in time_entries)

    # 3. Fetch user blockers (Privacy check: user mood is NOT in DB or output)
    blockers_cursor = db.standup_blockers.find({"user_id": user_id, "status": "active"})
    blockers = await blockers_cursor.to_list(length=20)

    # 4. Synthesize draft with honest telemetry and grounding
    draft_res = await AgileDraftService.generate_standup_draft_async(
        user_id=user_id,
        project_id=project_id,
        user_name=user_id,
        recent_tasks=user_tasks,
        logged_hours=logged_h,
        blockers=blockers,
    )

    telemetry = draft_res["telemetry"]
    grounding = draft_res["grounding_report"]

    draft_doc = {
        "kind": "standup",
        "owner_user_id": user_id,
        "project_id": project_id,
        "ref": {"user_id": user_id, "task_count": len(user_tasks)},
        "content": draft_res["content"],
        "data_sources": {"task_ids": [t.get("task_id") for t in user_tasks if t.get("task_id")]},
        "grounding_report": grounding,
        "method": telemetry["method"],
        "model_version": telemetry["model_version"],
        "generated_by": user_id,
        "status": "generated",
        "created_at": utc_now(),
    }

    insert_res = await db.ai_drafts.insert_one(draft_doc)
    draft_doc["_id"] = insert_res.inserted_id

    # Log honest telemetry in ai_runs
    await db.ai_runs.insert_one({
        "feature": "standup_draft",
        "trigger": "user",
        "user_id": user_id,
        "model": telemetry["model_version"],
        "model_version": telemetry["model_version"],
        "method": telemetry["method"],
        "prompt_version": telemetry.get("prompt_version", "v1"),
        "tokens_in": telemetry["tokens_in"],
        "tokens_out": telemetry["tokens_out"],
        "latency_ms": telemetry["latency_ms"],
        "status": "success",
        "error": None,
        "created_at": utc_now(),
    })

    return serialize_doc(draft_doc)


# ---------------------------------------------------------------------------
# Feature 5: Sprint Retrospective Draft
# ---------------------------------------------------------------------------
@router.post("/sprints/{sprint_id}/retro/draft", response_model=dict)
async def generate_retro_draft(
    sprint_id: str,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Feature 5: Generates a sprint retrospective summary draft for PM / TL review.
    Admin gets 403. Scoped to PM / TL's project.
    """
    sprint = await db.sprint_snapshots.find_one({"sprint_id": sprint_id})
    if not sprint:
        raise HTTPException(status_code=404, detail=f"Sprint '{sprint_id}' not found")

    project_id = sprint.get("project_id", "proj-gillo-core")
    if not current_user.all_projects and project_id != current_user.project_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: sprint does not belong to user project")

    # 1. Fetch sprint tasks
    tasks_cursor = db.task_snapshots.find({"sprint_id": sprint_id})
    tasks = await tasks_cursor.to_list(length=100)

    # 2. Identify slipped / incomplete tasks
    slipped = [t for t in tasks if t.get("status") in ["blocked"] or t.get("stalled_days", 0) > 3]

    # 3. Check fit check
    fit_check = await db.sprint_fit_checks.find_one({"sprint_id": sprint_id})

    draft_res = await AgileDraftService.generate_retro_draft_async(
        sprint_id=sprint_id,
        project_id=project_id,
        sprint_name=sprint.get("name", sprint_id),
        tasks=tasks,
        slipped_tasks=slipped,
        fit_check=fit_check,
    )

    telemetry = draft_res["telemetry"]
    grounding = draft_res["grounding_report"]

    draft_doc = {
        "kind": "retro",
        "owner_user_id": current_user.user_id,
        "project_id": project_id,
        "ref": {"sprint_id": sprint_id},
        "content": draft_res["content"],
        "data_sources": {"sprint_id": sprint_id, "task_count": len(tasks)},
        "grounding_report": grounding,
        "method": telemetry["method"],
        "model_version": telemetry["model_version"],
        "generated_by": current_user.user_id,
        "status": "generated",
        "created_at": utc_now(),
    }

    insert_res = await db.ai_drafts.insert_one(draft_doc)
    draft_doc["_id"] = insert_res.inserted_id

    await db.ai_runs.insert_one({
        "feature": "retro_draft",
        "trigger": "user",
        "user_id": current_user.user_id,
        "model": telemetry["model_version"],
        "model_version": telemetry["model_version"],
        "method": telemetry["method"],
        "prompt_version": telemetry.get("prompt_version", "v1"),
        "tokens_in": telemetry["tokens_in"],
        "tokens_out": telemetry["tokens_out"],
        "latency_ms": telemetry["latency_ms"],
        "status": "success",
        "error": None,
        "created_at": utc_now(),
    })

    return serialize_doc(draft_doc)


# ---------------------------------------------------------------------------
# Feature 3: Release Request Pre-Fill Draft
# ---------------------------------------------------------------------------
@router.post("/release-requests/{release_request_id}/draft", response_model=dict)
async def generate_release_draft(
    release_request_id: str,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Feature 3: Pre-fills release summary, changelog, and rollback plan.
    Admin gets 403. Scoped to PM / TL's project.
    """
    release = await db.release_requests.find_one({"release_request_id": release_request_id})
    if not release:
        raise HTTPException(status_code=404, detail=f"Release request '{release_request_id}' not found")

    project_id = release.get("project_id", "proj-gillo-core")
    if not current_user.all_projects and project_id != current_user.project_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: release does not belong to user project")

    task_ids = release.get("task_ids", [])
    tasks_cursor = db.task_snapshots.find({"task_id": {"$in": task_ids}})
    tasks = await tasks_cursor.to_list(length=100)

    prs_cursor = db.pull_requests.find({})
    prs = await prs_cursor.to_list(length=100)

    draft_res = await AgileDraftService.generate_release_draft_async(
        release_request_id=release_request_id,
        project_id=project_id,
        version=release.get("version", "v1.0.0"),
        tasks=tasks,
        prs=prs,
    )

    telemetry = draft_res["telemetry"]
    grounding = draft_res["grounding_report"]

    draft_doc = {
        "kind": "release_summary",
        "owner_user_id": current_user.user_id,
        "project_id": project_id,
        "ref": {"release_request_id": release_request_id},
        "content": draft_res["content"],
        "data_sources": {"task_ids": task_ids},
        "grounding_report": grounding,
        "method": telemetry["method"],
        "model_version": telemetry["model_version"],
        "generated_by": current_user.user_id,
        "status": "generated",
        "created_at": utc_now(),
    }

    insert_res = await db.ai_drafts.insert_one(draft_doc)
    draft_doc["_id"] = insert_res.inserted_id

    await db.ai_runs.insert_one({
        "feature": "release_draft",
        "trigger": "user",
        "user_id": current_user.user_id,
        "model": telemetry["model_version"],
        "model_version": telemetry["model_version"],
        "method": telemetry["method"],
        "prompt_version": telemetry.get("prompt_version", "v1"),
        "tokens_in": telemetry["tokens_in"],
        "tokens_out": telemetry["tokens_out"],
        "latency_ms": telemetry["latency_ms"],
        "status": "success",
        "error": None,
        "created_at": utc_now(),
    })

    return serialize_doc(draft_doc)


# ---------------------------------------------------------------------------
# Feature 3: Block Explainer Diagnostics
# ---------------------------------------------------------------------------
@router.post("/release-requests/{release_request_id}/explain-block", response_model=dict)
async def explain_release_block(
    release_request_id: str,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Feature 3: Diagnostic explainer for blocked releases analyzing unmerged PRs and incomplete tasks.
    Admin gets 403. Scoped to PM / TL's project.
    """
    release = await db.release_requests.find_one({"release_request_id": release_request_id})
    if not release:
        raise HTTPException(status_code=404, detail=f"Release request '{release_request_id}' not found")

    project_id = release.get("project_id", "proj-gillo-core")
    if not current_user.all_projects and project_id != current_user.project_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: release does not belong to user project")

    task_ids = release.get("task_ids", [])
    tasks_cursor = db.task_snapshots.find({
        "task_id": {"$in": task_ids},
        "status": {"$ne": "done"}
    })
    uncompleted = await tasks_cursor.to_list(length=100)

    prs_cursor = db.pull_requests.find({"status": "open"})
    open_prs = await prs_cursor.to_list(length=100)

    draft_res = await AgileDraftService.generate_block_explanation_async(
        release_request_id=release_request_id,
        project_id=project_id,
        version=release.get("version", "v1.0.0"),
        uncompleted_tasks=uncompleted,
        open_prs=open_prs,
    )

    telemetry = draft_res["telemetry"]
    grounding = draft_res["grounding_report"]

    draft_doc = {
        "kind": "block_explanation",
        "owner_user_id": current_user.user_id,
        "project_id": project_id,
        "ref": {"release_request_id": release_request_id},
        "content": draft_res["content"],
        "data_sources": {"blocking_task_count": len(uncompleted), "open_pr_count": len(open_prs)},
        "grounding_report": grounding,
        "method": telemetry["method"],
        "model_version": telemetry["model_version"],
        "generated_by": current_user.user_id,
        "status": "generated",
        "created_at": utc_now(),
    }

    insert_res = await db.ai_drafts.insert_one(draft_doc)
    draft_doc["_id"] = insert_res.inserted_id

    await db.ai_runs.insert_one({
        "feature": "release_block_explanation",
        "trigger": "user",
        "user_id": current_user.user_id,
        "model": telemetry["model_version"],
        "model_version": telemetry["model_version"],
        "method": telemetry["method"],
        "prompt_version": telemetry.get("prompt_version", "v1"),
        "tokens_in": telemetry["tokens_in"],
        "tokens_out": telemetry["tokens_out"],
        "latency_ms": telemetry["latency_ms"],
        "status": "success",
        "error": None,
        "created_at": utc_now(),
    })

    return serialize_doc(draft_doc)
