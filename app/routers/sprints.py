from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import time
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import settings
from app.database import get_database
from app.models.estimates import (
    TaskEstimateModel,
    SprintFitCheckModel,
    UserCapacityLoad,
    BaselineInfo
)
from app.models.suggestions import (
    AISuggestionBase,
    ReasoningStep,
    ChangePreview
)
from app.models.read_models import TaskSnapshot, SprintSnapshot
from app.ml.features import compute_model1_features, compute_model2_features
from app.ml.registry import (
    predict_story_points_with_fallback,
    predict_effort_with_fallback,
    predict_risk_with_fallback,
)
from app.services.estimation_service import (
    compute_dynamic_assignee_ratio,
    find_similar_completed_tasks,
    compute_empirical_baseline_hours,
    compute_nominal_baseline_median
)
from app.middleware.auth import CurrentUser, get_current_user, require_roles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sprints", tags=["Sprint Intelligence (Features 1 & 2)"])

# PM and TL authorized for sprint planning operations
auth_pm_or_tl = require_roles(["PM", "TL"])


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def serialize_doc(doc: dict) -> dict:
    if not doc:
        return doc
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc


@router.post("/{sprint_id}/estimate", response_model=dict)
async def estimate_sprint_tasks(
    sprint_id: str,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Feature 1: Task Effort Estimation with Empirical Baselines & ML Fallback.
    - Dynamically computes developer estimation velocity ratios with Bayesian shrinkage.
    - Evaluates empirical quantiles from similar completed tasks across projects.
    - Runs active model from registry or deterministic rule baseline (never HTTP 500).
    - Idempotently marks existing active estimates as superseded.
    - Records honest, measured ai_runs telemetry.
    """
    start_time = time.time()
    sprint = await db.sprint_snapshots.find_one({"sprint_id": sprint_id})
    if not sprint:
        raise HTTPException(status_code=404, detail=f"Sprint '{sprint_id}' not found")

    project_id = sprint.get("project_id", "default_proj")
    if not current_user.all_projects and project_id != current_user.project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: sprint '{sprint_id}' belongs to project '{project_id}'"
        )

    # 1. Fetch all tasks for this sprint
    tasks_cursor = db.task_snapshots.find({"sprint_id": sprint_id})
    tasks = await tasks_cursor.to_list(length=200)
    if not tasks:
        return {
            "sprint_id": sprint_id,
            "tasks_estimated": 0,
            "total_low_hours": 0.0,
            "total_high_hours": 0.0,
            "estimates": []
        }

    estimates_created = []

    for t in tasks:
        task_id = t["task_id"]
        assignee_id = t.get("assignee_id")
        raw_sp = t.get("story_points")
        task_type = t.get("task_type", "feature")
        title = t.get("title", "")

        # 2. Compute developer ratio dynamically with shrinkage from completed work
        if assignee_id and assignee_id != "unassigned":
            user_ratio, sample_size = await compute_dynamic_assignee_ratio(user_id=assignee_id, db=db)
        else:
            user_ratio, sample_size = 1.0, 0

        # 3. Cold-start hierarchy for empirical baseline hours
        similar_hours, level, is_cold_start = await find_similar_completed_tasks(
            project_id=project_id,
            task_type=task_type,
            story_points=raw_sp,
            db=db,
            min_samples=settings.ESTIMATION_COLD_START_MIN_SAMPLES
        )

        low_base, mid_base, high_base, band_type = compute_empirical_baseline_hours(
            similar_hours=similar_hours,
            story_points=raw_sp,
            assignee_ratio=user_ratio,
            min_samples=settings.ESTIMATION_COLD_START_MIN_SAMPLES
        )

        nominal_median = compute_nominal_baseline_median(
            similar_hours=similar_hours,
            story_points=raw_sp,
            min_samples=settings.ESTIMATION_COLD_START_MIN_SAMPLES
        )

        # 4. Extract canonical features (pass nominal_median so user_ratio is not double-counted downstream)
        features = compute_model1_features(
            task=t,
            assignee_ratio=user_ratio,
            historical_median_hours=nominal_median
        )

        # 5. Execute Model 1 Story Point Estimation with out-of-project interval
        description = t.get("description", "")
        sp_pred = predict_story_points_with_fallback(
            title=title,
            description=description,
            task_type=task_type,
            db=db,
            user_id=current_user.user_id
        )

        # 6. Execute Model 1 Effort (Hours) Estimation (labeled as unvalidated heuristic)
        pred = predict_effort_with_fallback(
            title=title,
            task_type=task_type,
            story_points=features["story_points"] or sp_pred["story_points"],
            dependency_count=features["dependency_count"],
            historical_median=features["historical_median_hours"],
            assignee_ratio=features["assignee_user_ratio"],
            cold_start=is_cold_start
        )

        # 7. Idempotently supersede prior active estimates for this task
        await db.task_estimates.update_many(
            {"task_id": task_id, "status": "active"},
            {"$set": {"status": "superseded"}}
        )

        # 8. Persist new estimate with method & model version tags
        est_doc = {
            "task_id": task_id,
            "sprint_id": sprint_id,
            "project_id": project_id,
            "assignee_id": assignee_id,
            "task_type": task_type,
            "original_story_points": raw_sp,
            "baseline": {
                "median_hours": round(mid_base, 1),
                "sample_size": sample_size
            },
            "story_points": sp_pred["story_points"],
            "raw_story_points": sp_pred.get("raw_story_points"),
            "story_points_interval": sp_pred["interval"],
            "story_points_confidence_note": sp_pred["confidence_note"],
            "sp_method": sp_pred["method"],
            "sp_model_version": sp_pred["model_version"],
            "sp_fallback_reason": sp_pred.get("fallback_reason"),
            "low_hours": pred["low_hours"],
            "mid_hours": pred.get("mid_hours", round((pred["low_hours"] + pred["high_hours"]) / 2.0, 1)),
            "high_hours": pred["high_hours"],
            "hours_conversion_note": pred.get("hours_conversion_note", "unvalidated heuristic"),
            "confidence": pred["confidence"],
            "confidence_note": pred.get("confidence_note"),
            "reason": pred["reason"],
            "warning": pred.get("warning"),
            "method": pred["method"],
            "model_version": pred["model_version"],
            "generated_by": "system",
            "status": "active",
            "created_at": utc_now()
        }

        insert_res = await db.task_estimates.insert_one(est_doc)
        est_doc["_id"] = insert_res.inserted_id
        estimates_created.append(serialize_doc(est_doc))

    latency_ms = int((time.time() - start_time) * 1000)

    # 8. Record telemetry in ai_runs
    method_used = estimates_created[0]["method"] if estimates_created else "rule_baseline"
    model_version_used = estimates_created[0]["model_version"] if estimates_created else "v1"

    await db.ai_runs.insert_one({
        "feature": "task_effort_estimation",
        "trigger": "user",
        "user_id": current_user.user_id,
        "model": "model_est",
        "model_version": model_version_used,
        "method": method_used,
        "prompt_version": None,
        "tokens_in": None,
        "tokens_out": None,
        "latency_ms": latency_ms,
        "status": "success",
        "created_at": utc_now()
    })

    return {
        "sprint_id": sprint_id,
        "tasks_estimated": len(estimates_created),
        "total_low_hours": round(sum(e["low_hours"] for e in estimates_created), 1),
        "total_mid_hours": round(sum(e.get("mid_hours", (e["low_hours"] + e["high_hours"]) / 2.0) for e in estimates_created), 1),
        "total_high_hours": round(sum(e["high_hours"] for e in estimates_created), 1),
        "estimates": estimates_created
    }


@router.get("/{sprint_id}/fit-check", response_model=dict)
async def check_sprint_fit(
    sprint_id: str,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Feature 1: Aggregates predicted hours per user against capacity.
    Determines sprint verdict (fits, tight, over) and generates move/reassign proposals in ai_suggestions.
    Enforces that proposals never compare, score, or rank developers' skill or performance.
    """
    start_time = time.time()
    sprint = await db.sprint_snapshots.find_one({"sprint_id": sprint_id})
    if not sprint:
        raise HTTPException(status_code=404, detail=f"Sprint '{sprint_id}' not found")

    project_id = sprint.get("project_id", "default_proj")
    if not current_user.all_projects and project_id != current_user.project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: sprint '{sprint_id}' belongs to project '{project_id}'"
        )

    # 1. Fetch active task estimates for this sprint
    estimates_cursor = db.task_estimates.find({"sprint_id": sprint_id, "status": "active"})
    estimates = await estimates_cursor.to_list(length=300)

    # 2. Fetch user capacities for this sprint
    capacities_cursor = db.capacity.find({"sprint_id": sprint_id})
    user_caps = {c["user_id"]: float(c.get("capacity_hours", settings.DEFAULT_CAPACITY_HOURS)) for c in await capacities_cursor.to_list(length=100)}

    # 3. Aggregate load per user
    user_loads = {}
    tasks_by_user = {}

    for e in estimates:
        uid = e.get("assignee_id") or "unassigned"
        low_h = float(e.get("low_hours", 0.0))
        high_h = float(e.get("high_hours", 0.0))
        mid_h = float(e.get("mid_hours", round((low_h + high_h) / 2.0, 1)))
        if uid not in user_loads:
            user_loads[uid] = {
                "capacity_h": user_caps.get(uid, settings.DEFAULT_CAPACITY_HOURS if uid != "unassigned" else 0.0),
                "load_low": 0.0,
                "load_mid": 0.0,
                "load_high": 0.0
            }
            tasks_by_user[uid] = []
        user_loads[uid]["load_low"] += low_h
        user_loads[uid]["load_mid"] += mid_h
        user_loads[uid]["load_high"] += high_h
        tasks_by_user[uid].append(e)

    # Round aggregated numbers
    for uid in user_loads:
        user_loads[uid]["load_low"] = round(user_loads[uid]["load_low"], 1)
        user_loads[uid]["load_mid"] = round(user_loads[uid]["load_mid"], 1)
        user_loads[uid]["load_high"] = round(user_loads[uid]["load_high"], 1)

    # 4. Evaluate Sprint Verdict (fits, tight, over)
    verdict = "fits"
    overloaded_users = []
    underloaded_users = []

    for uid, stats in user_loads.items():
        if uid == "unassigned":
            continue
        cap = stats["capacity_h"]
        # 'over': expected mid-estimate load exceeds capacity threshold
        if stats["load_mid"] > (cap * settings.SPRINT_OVERLOAD_RATIO):
            verdict = "over"
            overloaded_users.append((uid, stats["load_mid"] - cap))
        # 'tight': high-estimate load exceeds capacity threshold, but expected mid-estimate fits within capacity
        elif stats["load_high"] > (cap * settings.SPRINT_TIGHT_RATIO):
            if verdict != "over":
                verdict = "tight"
            overloaded_users.append((uid, stats["load_high"] - cap))
        # underloaded: high-estimate load is comfortably below capacity
        elif stats["load_high"] < (cap * settings.SPRINT_UNDERLOAD_RATIO):
            underloaded_users.append((uid, cap - stats["load_high"]))

    # Sort overloaded users by excess workload descending
    overloaded_users.sort(key=lambda x: x[1], reverse=True)
    underloaded_users.sort(key=lambda x: x[1], reverse=True)

    # 5. If over or tight, generate actionable proposals in ai_suggestions
    generated_suggestion_ids = []
    if (verdict in ["over", "tight"]) and overloaded_users:
        overloaded_user, excess_h = overloaded_users[0]
        candidate_tasks = tasks_by_user.get(overloaded_user, [])

        # Sort candidates to find the most suitable task to rebalance (e.g. smallest task that helps relieve overload)
        candidate_tasks.sort(key=lambda x: float(x.get("high_hours", 0.0)))

        if candidate_tasks:
            target_task = candidate_tasks[0]
            task_id = target_task["task_id"]
            task_load = float(target_task.get("high_hours", 0.0))

            # Check if there is an underloaded user with sufficient available capacity
            target_user = None
            for uid, avail_h in underloaded_users:
                if avail_h >= task_load:
                    target_user = uid
                    break

            if target_user:
                # Deterministic Reassign Proposal: reassigns only based on spare capacity
                dedupe_key = f"fix:reassign:{sprint_id}:{task_id}"
                existing = await db.ai_suggestions.find_one({"dedupe_key": dedupe_key, "status": "pending"})
                if not existing:
                    proposal_doc = {
                        "project_id": project_id,
                        "sprint_id": sprint_id,
                        "task_id": task_id,
                        "concerns_user_id": overloaded_user,
                        "type": "sprint_fix",
                        "action": "reassign",
                        "audience_roles": ["PM", "TL"],
                        "severity": None,
                        "confidence": "high",
                        "confidence_note": "Rule-based reassign proposal based on deterministic capacity matching.",
                        "status": "pending",
                        "method": "rule_baseline",
                        "model_version": "deterministic_fit_v1",
                        "generated_by": "system",
                        "warning": None,
                        "dedupe_key": dedupe_key,
                        "proposal": {
                            "message": f"Reassign task '{task_id}' ({task_load:.1f}h) from {overloaded_user} to {target_user} to balance sprint capacity."
                        },
                        "change_preview": {
                            "before": {"assignee_id": overloaded_user},
                            "after": {"assignee_id": target_user}
                        },
                        "reasoning_steps": [
                            {
                                "order": 1,
                                "description": f"Assignee {overloaded_user} projected workload exceeds sprint capacity by +{excess_h:.1f}h.",
                                "evidence": {"assignee_overload_hours": round(excess_h, 1), "threshold": "> capacity"}
                            },
                            {
                                "order": 2,
                                "description": f"Target assignee {target_user} has available remaining capacity of {avail_h:.1f}h.",
                                "evidence": {"available_capacity_hours": round(avail_h, 1), "task_hours": task_load}
                            }
                        ],
                        "data_sources": {"sprint_id": sprint_id, "task_id": task_id},
                        "apply_status": "none",
                        "created_at": utc_now(),
                        "comments": []
                    }
                    res = await db.ai_suggestions.insert_one(proposal_doc)
                    generated_suggestion_ids.append(str(res.inserted_id))
            else:
                # No user has spare capacity -> propose moving task to backlog / next sprint
                dedupe_key = f"fix:move:{sprint_id}:{task_id}"
                existing = await db.ai_suggestions.find_one({"dedupe_key": dedupe_key, "status": "pending"})
                if not existing:
                    proposal_doc = {
                        "project_id": project_id,
                        "sprint_id": sprint_id,
                        "task_id": task_id,
                        "concerns_user_id": overloaded_user,
                        "type": "sprint_fix",
                        "action": "move",
                        "audience_roles": ["PM", "TL"],
                        "severity": None,
                        "confidence": "high",
                        "confidence_note": "Rule-based move proposal due to lack of team capacity.",
                        "status": "pending",
                        "method": "rule_baseline",
                        "model_version": "deterministic_fit_v1",
                        "generated_by": "system",
                        "warning": "Team capacity is fully allocated across all assignees.",
                        "dedupe_key": dedupe_key,
                        "proposal": {
                            "message": f"De-scope task '{task_id}' ({task_load:.1f}h) from Sprint '{sprint_id}' to backlog to resolve team overflow."
                        },
                        "change_preview": {
                            "before": {"sprint_id": sprint_id},
                            "after": {"sprint_id": None}
                        },
                        "reasoning_steps": [
                            {
                                "order": 1,
                                "description": f"Sprint workload is '{verdict}' and all developers are at or above target capacity limits.",
                                "evidence": {"sprint_verdict": verdict, "excess_workload_hours": round(excess_h, 1)}
                            }
                        ],
                        "data_sources": {"sprint_id": sprint_id, "task_id": task_id},
                        "apply_status": "none",
                        "created_at": utc_now(),
                        "comments": []
                    }
                    res = await db.ai_suggestions.insert_one(proposal_doc)
                    generated_suggestion_ids.append(str(res.inserted_id))

    latency_ms = int((time.time() - start_time) * 1000)

    # 6. Save Sprint Fit Check Record
    fit_record = {
        "sprint_id": sprint_id,
        "project_id": project_id,
        "per_user": user_loads,
        "verdict": verdict,
        "suggestion_ids": generated_suggestion_ids,
        "method": "rule_baseline",
        "model_version": "deterministic_fit_v1",
        "generated_by": "system",
        "created_at": utc_now()
    }
    fit_insert = await db.sprint_fit_checks.insert_one(fit_record)
    fit_record["_id"] = fit_insert.inserted_id

    # Record telemetry in ai_runs
    await db.ai_runs.insert_one({
        "feature": "sprint_fit_check",
        "trigger": "user",
        "user_id": current_user.user_id,
        "model": "capacity_fit_rules",
        "model_version": "v1.0",
        "method": "rule_baseline",
        "prompt_version": None,
        "tokens_in": None,
        "tokens_out": None,
        "latency_ms": latency_ms,
        "status": "success",
        "created_at": utc_now()
    })

    return serialize_doc(fit_record)


@router.post("/{sprint_id}/scan-risks", response_model=dict)
async def scan_sprint_risks(
    sprint_id: str,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Feature 2: Scans active sprint tasks for carry-over risk.
    - Integrates Model 2 Taskove (LightGBM + Isotonic Calibration) with honest fallback.
    - Uses localized within-sprint top-k policy: k = min(5, ceil(0.2 * n_open)).
    - Generates Inbox suggestions with relative ranking text (no raw probabilities returned).
    - Stores calibrated probabilities strictly in ai_runs for telemetry.
    - Deduplicates against pending or dismissed suggestions in this sprint.
    - Admin strictly blocked via auth_pm_or_tl dependency.
    """
    start_time = time.time()
    sprint_doc = await db.sprint_snapshots.find_one({"sprint_id": sprint_id})
    if not sprint_doc:
        raise HTTPException(status_code=404, detail=f"Sprint '{sprint_id}' not found")

    project_id = sprint_doc.get("project_id", "default_proj")
    if not current_user.all_projects and project_id != current_user.project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: sprint '{sprint_id}' belongs to project '{project_id}'"
        )

    # Construct SprintSnapshot
    start_d = sprint_doc["start_date"]
    if isinstance(start_d, str):
        start_d = datetime.fromisoformat(start_d.replace("Z", "+00:00"))
    end_d = sprint_doc["end_date"]
    if isinstance(end_d, str):
        end_d = datetime.fromisoformat(end_d.replace("Z", "+00:00"))

    sprint_snapshot = SprintSnapshot(
        sprint_id=sprint_id,
        project_id=project_id,
        name=sprint_doc.get("name", sprint_id),
        start_date=start_d,
        end_date=end_d,
        status=sprint_doc.get("status", "active")
    )

    # 1. Fetch all tasks in this sprint
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

    from app.ml.registry import predict_sprint_carryover_risks
    scoring_result = predict_sprint_carryover_risks(
        tasks=sprint_tasks,
        sprint=sprint_snapshot,
        now=utc_now()
    )

    warnings_created = []
    k = scoring_result["k_threshold"]
    n_open = scoring_result["n_open_tasks"]

    task_map = {t.task_id: t for t in sprint_tasks}

    for t_scored in scoring_result["scored_tasks"]:
        task_id = t_scored["task_id"]
        task_obj = task_map.get(task_id)
        if not task_obj:
            continue

        # Top-k policy: Only top-k tasks (risk_tier == 'high') generate an inbox suggestion
        if t_scored["risk_tier"] == "high":
            # Deduplication: do not re-suggest if pending or dismissed in this sprint
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
                    res = await db.ai_suggestions.insert_one(suggestion_doc)
                    suggestion_doc["_id"] = res.inserted_id
                    warnings_created.append(serialize_doc(suggestion_doc))
                except Exception as e:
                    logger.info(f"Duplicate suggestion skipped by partial index: {e}")

    latency_ms = int((time.time() - start_time) * 1000)

    # Telemetry: Record calibrated probabilities strictly in ai_runs
    telemetry_probs = {
        st["task_id"]: st.get("_telemetry_calibrated_prob")
        for st in scoring_result["scored_tasks"]
        if st.get("_telemetry_calibrated_prob") is not None
    }
    telemetry_contribs = {
        st["task_id"]: st.get("_telemetry_contributions")
        for st in scoring_result["scored_tasks"]
        if st.get("_telemetry_contributions") is not None
    }

    await db.ai_runs.insert_one({
        "feature": "sprint_risk_scan",
        "trigger": "user",
        "user_id": current_user.user_id,
        "model": "model2_sprint_carryover_risk_taskove" if scoring_result["method"] == "lightgbm_taskove_v2" else "rule_baseline",
        "model_version": scoring_result["model_version"],
        "method": scoring_result["method"],
        "calibrated_probabilities": telemetry_probs,
        "feature_contributions": telemetry_contribs,
        "latency_ms": latency_ms,
        "status": "success",
        "fallback_reason": scoring_result.get("fallback_reason"),
        "created_at": utc_now()
    })

    # Prepare user-facing task list: NEVER return raw probability percentage to client!
    client_scored = [
        {
            "task_id": st["task_id"],
            "title": st["title"],
            "assignee_id": st.get("assignee_id"),
            "risk_rank": st["risk_rank"],
            "risk_tier": st["risk_tier"],
            "method": st["method"],
            "model_version": st["model_version"],
            "top_contributing_factors": st.get("top_contributing_factors", []),
            "factor_note": st.get("factor_note"),
            "note": st.get("note") or st.get("factor_note"),
            "reason": st["reason"]
        }
        for st in scoring_result["scored_tasks"]
    ]

    return {
        "sprint_id": sprint_id,
        "active_tasks_scanned": n_open,
        "k_threshold": k,
        "method": scoring_result["method"],
        "model_version": scoring_result["model_version"],
        "fallback_reason": scoring_result.get("fallback_reason"),
        "explanation_note": scoring_result.get(
            "explanation_note",
            "These factors show what the model weighed on similar past sprints; they are hints, not confirmed causes."
        ),
        "sprint_context": scoring_result.get("sprint_context"),
        "new_risk_warnings_generated": len(warnings_created),
        "warnings": warnings_created,
        "scored_tasks": client_scored
    }
