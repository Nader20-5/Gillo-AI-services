"""
Estimation and Sprint Fit-Check Service for Gillo AI Service.
Handles:
1. Empirical task baseline estimation with cold-start hierarchy.
2. Dynamic estimation ratio calculation for developers with Bayesian shrinkage.
3. Capacity fit check and deterministic, skill-neutral reassignment/move proposals.
4. Method and telemetry attribution for all estimates.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

from app.config import settings
from app.ml.features import compute_model1_features
from app.ml.registry import predict_effort_with_fallback

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def compute_dynamic_assignee_ratio(
    user_id: str,
    db: Any,
    prior_mean: float = 1.0,
    prior_weight: float = 3.0
) -> Tuple[float, int]:
    """
    Computes developer estimation ratio dynamically from completed tasks:
        raw_ratio = actual_hours / max(estimated_hours, 0.5)
    Applies outlier clipping [0.50, 2.50] and Bayesian shrinkage toward 1.0 for small sample sizes:
        shrunk_ratio = (sum(ratios) + prior_weight * prior_mean) / (N + prior_weight)
    """
    # Find completed tasks assigned to this user
    cursor = db.task_snapshots.find({"assignee_id": user_id, "status": "done"})
    completed_tasks = await cursor.to_list(length=100)

    ratios = []
    for t in completed_tasks:
        task_id = t.get("task_id")
        # 1. Fetch actual hours from time_entries
        time_cursor = db.time_entries.find({"task_id": task_id, "user_id": user_id})
        time_docs = await time_cursor.to_list(length=50)
        actual_h = sum(float(td.get("hours", 0.0)) for td in time_docs)

        # 2. Fetch estimated hours from task_estimates
        est_doc = await db.task_estimates.find_one({"task_id": task_id})
        est_h = None
        if est_doc:
            est_h = float(est_doc.get("high_hours") or est_doc.get("low_hours") or 0.0)
        elif t.get("story_points"):
            est_h = float(t.get("story_points")) * 5.0

        if actual_h > 0 and est_h and est_h > 0:
            raw_r = actual_h / est_h
            # Clip outliers
            clipped_r = max(0.50, min(2.50, raw_r))
            ratios.append(clipped_r)

    n_samples = len(ratios)
    if n_samples == 0:
        return round(prior_mean, 2), 0

    # Bayesian shrinkage formula
    shrunk = (sum(ratios) + prior_weight * prior_mean) / (n_samples + prior_weight)
    return round(float(shrunk), 2), n_samples


async def refresh_assignee_profile(user_id: str, db: Any) -> Tuple[float, int]:
    """Idempotently refreshes ai_profiles with the developer's dynamic estimation ratio."""
    ratio, sample_size = await compute_dynamic_assignee_ratio(user_id=user_id, db=db)
    await db.ai_profiles.update_one(
        {"user_id": user_id},
        {"$set": {
            "estimation.ratio": ratio,
            "estimation.sample_size": sample_size,
            "updated_at": utc_now()
        }},
        upsert=True
    )
    return ratio, sample_size


async def find_similar_completed_tasks(
    project_id: str,
    task_type: str,
    story_points: Optional[float],
    db: Any,
    min_samples: int = 5
) -> Tuple[List[float], str, bool]:
    """
    Cold-start hierarchy to find similar completed task durations (hours):
    Level 1: Same project, same type, matching story points bucket
    Level 2: Same project, same type
    Level 3: Global (all projects), same type
    Level 4: Global cold-start fallback
    Returns: (list_of_hours, level_name, is_cold_start)
    """
    # Level 1: project + type + story_points
    if story_points is not None:
        query_l1 = {
            "project_id": project_id,
            "task_type": task_type,
            "status": "done",
            "story_points": {"$gte": story_points - 1.0, "$lte": story_points + 1.0}
        }
        l1_tasks = await db.task_snapshots.find(query_l1).to_list(length=50)
        hours = await _extract_logged_hours_for_tasks(l1_tasks, db)
        if len(hours) >= min_samples:
            return hours, "project_type_sp", False

    # Level 2: project + type
    query_l2 = {"project_id": project_id, "task_type": task_type, "status": "done"}
    l2_tasks = await db.task_snapshots.find(query_l2).to_list(length=50)
    hours = await _extract_logged_hours_for_tasks(l2_tasks, db)
    if len(hours) >= min_samples:
        return hours, "project_type", False

    # Level 3: global type across all projects
    query_l3 = {"task_type": task_type, "status": "done"}
    l3_tasks = await db.task_snapshots.find(query_l3).to_list(length=50)
    hours = await _extract_logged_hours_for_tasks(l3_tasks, db)
    if len(hours) >= min_samples:
        return hours, "global_type", True

    # Level 4: cold start
    return [], "global_cold_start", True


async def _extract_logged_hours_for_tasks(tasks: List[dict], db: Any) -> List[float]:
    """
    Extracts normalized nominal task hours from completed tasks:
    1. If actual time logged: nominal_h = total_actual_h / past_assignee_ratio.
       This normalizes out past developer speed so the baseline median reflects
       nominal task effort rather than raw actual hours.
    2. Fallback to story_points * 5.0 as nominal proxy when no time entries exist.
    """
    hours_list = []
    profile_cache: Dict[str, float] = {}
    for t in tasks:
        task_id = t.get("task_id")
        time_cursor = db.time_entries.find({"task_id": task_id})
        time_entries = await time_cursor.to_list(length=50)
        total_h = sum(float(te.get("hours", 0.0)) for te in time_entries)

        assignee_id = t.get("assignee_id")
        past_ratio = 1.0
        if assignee_id and assignee_id != "unassigned":
            if assignee_id not in profile_cache:
                prof = await db.ai_profiles.find_one({"user_id": assignee_id})
                past_ratio = prof.get("estimation", {}).get("ratio", 1.0) if prof else 1.0
                profile_cache[assignee_id] = past_ratio
            else:
                past_ratio = profile_cache[assignee_id]

        if total_h > 0:
            nominal_h = total_h / max(past_ratio, 0.30)
            hours_list.append(nominal_h)
        elif t.get("story_points"):
            # If no logged hours, use story points * 5.0 as nominal proxy
            hours_list.append(float(t.get("story_points")) * 5.0)
    return hours_list


def compute_nominal_baseline_median(
    similar_hours: List[float],
    story_points: Optional[float],
    min_samples: int = 5
) -> float:
    """
    Computes the unadjusted nominal baseline median hours (without developer ratio scaling).
    Used as the 'historical_median_hours' feature so downstream models/rules do not double-count
    the target developer's ratio.
    """
    if len(similar_hours) >= min_samples:
        return round(float(np.percentile(similar_hours, 50)), 1)
    sp = story_points if story_points is not None else 1.0
    return round(float(sp * 5.0), 1)


def compute_empirical_baseline_hours(
    similar_hours: List[float],
    story_points: Optional[float],
    assignee_ratio: float,
    min_samples: int = 5
) -> Tuple[float, float, float, str]:
    """
    Computes developer-adjusted low, mid, high hours:
    - If at least min_samples empirical observations exist: empirical quantiles (P25, P50, P75)
      scaled by the target developer's assignee_ratio.
    - Otherwise: percentage band (+/- 35%) around (story_points * 5.0) scaled by assignee_ratio.
    Returns: (low_hours, mid_hours, high_hours, band_type)
    """
    if len(similar_hours) >= min_samples:
        p25 = float(np.percentile(similar_hours, 25))
        p50 = float(np.percentile(similar_hours, 50))
        p75 = float(np.percentile(similar_hours, 75))

        low = max(0.5, round(p25 * assignee_ratio, 1))
        mid = max(low, round(p50 * assignee_ratio, 1))
        high = max(mid + 0.5, round(p75 * assignee_ratio, 1))
        return low, mid, high, "empirical_quantiles"
    else:
        nominal_mid = compute_nominal_baseline_median(similar_hours, story_points, min_samples)
        mid = max(0.5, round(nominal_mid * assignee_ratio, 1))
        spread = 0.35 if story_points is not None else 0.50
        low = max(0.5, round(mid * (1.0 - spread), 1))
        high = max(low + 0.5, round(mid * (1.0 + spread), 1))
        return low, mid, high, "percentage_band"
