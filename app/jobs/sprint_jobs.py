"""
Background Jobs for Gillo AI Service.
Includes:
1. Daily risk scanner across all active sprints (idempotent, safe to run multiple times).
2. Expiring suggestions past expires_at and superseding suggestions whose task is completed or moved.
3. Refreshing developer estimation velocity ratios dynamically from completed sprints.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import logging
import time

from app.config import settings
from app.ml.features import compute_model2_features
from app.ml.registry import predict_risk_with_fallback, predict_sprint_carryover_risks
from app.models.read_models import TaskSnapshot, SprintSnapshot

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def run_daily_risk_scan(db: Any) -> Dict[str, Any]:
    """
    Scans all active sprints for task slippage risk.
    Safe to run repeatedly; uses dedupe_key to prevent duplicate pending suggestions.
    Logs telemetry to ai_runs.
    """
    start_time = time.time()
    active_sprints_cursor = db.sprint_snapshots.find({"status": "active"})
    active_sprints = await active_sprints_cursor.to_list(length=50)

    total_tasks_scanned = 0
    total_warnings_generated = 0
    scanned_sprint_ids = []

    for sprint in active_sprints:
        sprint_id = sprint["sprint_id"]
        project_id = sprint.get("project_id", "default_proj")
        scanned_sprint_ids.append(sprint_id)

        start_d = sprint["start_date"]
        if isinstance(start_d, str):
            start_d = datetime.fromisoformat(start_d)
        end_d = sprint["end_date"]
        if isinstance(end_d, str):
            end_d = datetime.fromisoformat(end_d)

        sprint_snapshot = SprintSnapshot(
            sprint_id=sprint_id,
            project_id=project_id,
            name=sprint.get("name", sprint_id),
            start_date=start_d,
            end_date=end_d,
            status=sprint.get("status", "active")
        )

        # 1. Fetch tasks
        tasks_cursor = db.task_snapshots.find({"sprint_id": sprint_id})
        task_docs = await tasks_cursor.to_list(length=500)
        sprint_tasks = []
        for td in task_docs:
            c_at = td.get("created_at")
            if isinstance(c_at, str):
                c_at = datetime.fromisoformat(c_at)
            u_at = td.get("updated_at")
            if isinstance(u_at, str):
                u_at = datetime.fromisoformat(u_at)
            s_at = td.get("sprint_assigned_at")
            if isinstance(s_at, str):
                s_at = datetime.fromisoformat(s_at)
            sc_at = td.get("status_changed_at")
            if isinstance(sc_at, str):
                sc_at = datetime.fromisoformat(sc_at)

            sprint_tasks.append(TaskSnapshot(
                task_id=td["task_id"],
                project_id=td.get("project_id", project_id),
                sprint_id=sprint_id,
                title=td.get("title", ""),
                description=td.get("description", ""),
                task_type=td.get("task_type", "feature"),
                story_points=td.get("story_points"),
                status=td.get("status", "todo"),
                assignee_id=td.get("assignee_id"),
                stalled_days=int(td.get("stalled_days", 0)),
                dependencies=td.get("dependencies", []),
                created_at=c_at or utc_now(),
                updated_at=u_at or utc_now(),
                sprint_assigned_at=s_at,
                status_changed_at=sc_at
            ))

        total_tasks_scanned += len(sprint_tasks)

        scoring_result = predict_sprint_carryover_risks(
            tasks=sprint_tasks,
            sprint=sprint_snapshot,
            now=utc_now()
        )

        k = scoring_result["k_threshold"]
        n_open = scoring_result["n_open_tasks"]
        task_map = {t.task_id: t for t in sprint_tasks}

        for t_scored in scoring_result["scored_tasks"]:
            task_id = t_scored["task_id"]
            task_obj = task_map.get(task_id)
            if not task_obj:
                continue

            if t_scored["risk_tier"] == "high":
                existing = await db.ai_suggestions.find_one({
                    "sprint_id": sprint_id,
                    "task_id": task_id,
                    "type": "risk_warning",
                    "status": {"$in": ["pending", "dismissed"]}
                })
                if not existing:
                    dedupe_key = f"risk:{sprint_id}:{task_id}"
                    suggestion_doc = {
                        "project_id": project_id,
                        "sprint_id": sprint_id,
                        "task_id": task_id,
                        "concerns_user_id": task_obj.assignee_id,
                        "type": "risk_warning",
                        "action": None,
                        "audience_roles": ["PM", "TL"],
                        "severity": "high",
                        "confidence": "high" if t_scored["method"] == "lightgbm_taskove_v2" else "medium",
                        "confidence_note": "Within-sprint localized top-k risk prioritization." if t_scored["method"] == "lightgbm_taskove_v2" else "Deterministic rule baseline ranking.",
                        "status": "pending",
                        "method": t_scored["method"],
                        "model_version": t_scored["model_version"],
                        "generated_by": "system",
                        "warning": scoring_result.get("fallback_reason"),
                        "dedupe_key": dedupe_key,
                        "proposal": {
                            "message": (
                                f"Task '{task_obj.title}' ranked #{t_scored['risk_rank']} for carry-over risk "
                                f"in Sprint '{sprint_snapshot.name}' relative to {n_open} open sprint tasks "
                                f"(top-{k} focus). Review scope or blockers during standup."
                            )
                        },
                        "change_preview": {
                            "before": {"status": task_obj.status, "assignee_id": task_obj.assignee_id},
                            "after": {"status": "reassigned_or_mitigated"}
                        },
                        "reasoning_steps": [
                            {
                                "order": 1,
                                "description": t_scored["reason"],
                                "evidence": {
                                    "risk_rank": t_scored["risk_rank"],
                                    "risk_tier": t_scored["risk_tier"],
                                    "k_threshold": k,
                                    "top_contributing_factors": t_scored.get("top_contributing_factors", [])
                                }
                            }
                        ],
                        "data_sources": {"task_id": task_id, "sprint_id": sprint_id},
                        "apply_status": "none",
                        "created_at": utc_now(),
                        "comments": []
                    }
                    try:
                        await db.ai_suggestions.insert_one(suggestion_doc)
                        total_warnings_generated += 1
                    except Exception as e:
                        logger.info(f"Duplicate suggestion skipped by partial index: {e}")

    latency_ms = int((time.time() - start_time) * 1000)

    # Log telemetry record to ai_runs
    await db.ai_runs.insert_one({
        "feature": "scheduled_risk_scan",
        "trigger": "schedule",
        "user_id": None,
        "model": "model_risk",
        "model_version": "v1_synthetic",
        "method": "ml_model_synthetic",
        "prompt_version": None,
        "tokens_in": None,
        "tokens_out": None,
        "latency_ms": latency_ms,
        "status": "success",
        "created_at": utc_now()
    })

    return {
        "sprints_scanned": len(scanned_sprint_ids),
        "tasks_scanned": total_tasks_scanned,
        "warnings_generated": total_warnings_generated,
        "latency_ms": latency_ms
    }


async def expire_and_supersede_suggestions(db: Any) -> Tuple[int, int]:
    """
    1. Expires pending suggestions that have passed their expires_at date.
    2. Supersedes suggestions whose task was completed or moved out of the sprint.
    """
    # 1. Expire outdated suggestions
    now = utc_now()
    expire_res = await db.ai_suggestions.update_many(
        {"status": "pending", "expires_at": {"$lte": now}},
        {"$set": {"status": "expired", "updated_at": now}}
    )
    expired_count = expire_res.modified_count

    # 2. Supersede suggestions whose task is no longer pending/active
    pending_cursor = db.ai_suggestions.find({"status": "pending", "task_id": {"$ne": None}})
    superseded_count = 0

    async for sug in pending_cursor:
        task_id = sug.get("task_id")
        sprint_id = sug.get("sprint_id")
        task = await db.task_snapshots.find_one({"task_id": task_id})

        should_supersede = False
        supersede_reason = ""

        if not task:
            should_supersede = True
            supersede_reason = "Referenced task was deleted."
        elif task.get("status") == "done":
            should_supersede = True
            supersede_reason = "Referenced task was completed."
        elif sprint_id and task.get("sprint_id") != sprint_id:
            should_supersede = True
            supersede_reason = "Referenced task was moved to another sprint or backlog."

        if should_supersede:
            await db.ai_suggestions.update_one(
                {"_id": sug["_id"], "status": "pending"},
                {"$set": {
                    "status": "superseded",
                    "decision_reason": supersede_reason,
                    "updated_at": now
                }}
            )
            superseded_count += 1

    return expired_count, superseded_count


# ---------------------------------------------------------------------------
# Dynamic Developer Estimation Ratio Maintenance (Internal Only)
# ---------------------------------------------------------------------------
# NOTE: Developer ratios (velocity/estimation multipliers) are strictly INTERNAL
# operational calculations. They are refreshed periodically or on sprint close
# with Bayesian shrinkage solely to normalize historical task durations and model
# aggregate sprint capacity fit.
#
# PRIVACY & ETHICAL GUARANTEE:
# 1. Developer ratios are strictly internal: NEVER returned in any client API
#    response, telemetry payload, export, or response model.
# 2. Developer ratios are NEVER compared or ranked across developers.
#    The system enforces zero developer comparison and zero performance scoring.
# ---------------------------------------------------------------------------
async def refresh_developer_ratios(db: Any) -> int:
    """
    Finds developers with completed tasks and refreshes their internal dynamic estimation ratio.
    
    This function operates strictly in the background to update internal Bayesian priors:
    - Never exposes ratio values to clients or API responses.
    - Never ranks or compares developers against each other.
    - Used solely for nominal duration normalization in task baseline calculations.
    """
    cursor = db.task_snapshots.distinct("assignee_id", {"status": "done"})
    developer_ids = await cursor
    updated = 0

    from app.services.estimation_service import refresh_assignee_profile
    for uid in developer_ids:
        if uid and uid != "unassigned":
            try:
                await refresh_assignee_profile(user_id=uid, db=db)
                updated += 1
            except Exception as e:
                logger.warning(f"Failed to refresh estimation profile for {uid}: {e}")

    return updated
