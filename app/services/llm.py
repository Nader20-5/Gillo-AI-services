from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import time
from app.services.llm_provider import (
    generate_structured_draft,
    run_grounding_check,
    get_llm_provider,
    DeterministicTemplateProvider,
)
import logging

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgileDraftService:
    """
    AgileDraftService provides structured draft synthesis with honest telemetry.
    Delegates to LLM Provider with strict Pydantic output validation, 1x retry on schema failure,
    deterministic fallback, and entity grounding verification.
    """

    @staticmethod
    async def generate_standup_draft_async(
        user_id: str,
        project_id: str,
        user_name: str,
        recent_tasks: List[Dict[str, Any]],
        logged_hours: float,
        blockers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = {
            "user_id": user_id,
            "project_id": project_id,
            "user_name": user_name,
            "recent_tasks": recent_tasks,
            "logged_hours": logged_hours,
            "blockers": blockers,
        }
        content, telemetry = await generate_structured_draft("standup", context)
        # Run grounding check
        snapshot = {
            "task_ids": [t.get("task_id") for t in recent_tasks if t.get("task_id")],
            "tasks": recent_tasks,
        }
        grounding = run_grounding_check(content, snapshot)

        return {
            "content": content,
            "grounding_report": grounding,
            "telemetry": telemetry,
        }

    @staticmethod
    async def generate_retro_draft_async(
        sprint_id: str,
        project_id: str,
        sprint_name: str,
        tasks: List[Dict[str, Any]],
        slipped_tasks: List[Dict[str, Any]],
        fit_check: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = {
            "sprint_id": sprint_id,
            "project_id": project_id,
            "sprint_name": sprint_name,
            "tasks": tasks,
            "slipped_tasks": slipped_tasks,
            "fit_check": fit_check,
        }
        content, telemetry = await generate_structured_draft("retro", context)
        snapshot = {
            "task_ids": [t.get("task_id") for t in tasks if t.get("task_id")],
            "tasks": tasks,
        }
        grounding = run_grounding_check(content, snapshot)

        return {
            "content": content,
            "grounding_report": grounding,
            "telemetry": telemetry,
        }

    @staticmethod
    async def generate_release_draft_async(
        release_request_id: str,
        project_id: str,
        version: str,
        tasks: List[Dict[str, Any]],
        prs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = {
            "release_request_id": release_request_id,
            "project_id": project_id,
            "version": version,
            "tasks": tasks,
            "prs": prs,
        }
        content, telemetry = await generate_structured_draft("release_summary", context)
        snapshot = {
            "task_ids": [t.get("task_id") for t in tasks if t.get("task_id")],
            "tasks": tasks,
            "prs": prs,
        }
        grounding = run_grounding_check(content, snapshot)

        return {
            "content": content,
            "grounding_report": grounding,
            "telemetry": telemetry,
        }

    @staticmethod
    async def generate_block_explanation_async(
        release_request_id: str,
        project_id: str,
        version: str,
        uncompleted_tasks: List[Dict[str, Any]],
        open_prs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = {
            "release_request_id": release_request_id,
            "project_id": project_id,
            "version": version,
            "uncompleted_tasks": uncompleted_tasks,
            "open_prs": open_prs,
        }
        content, telemetry = await generate_structured_draft("block_explanation", context)
        snapshot = {
            "task_ids": [t.get("task_id") for t in uncompleted_tasks if t.get("task_id")],
            "tasks": uncompleted_tasks,
            "prs": open_prs,
        }
        grounding = run_grounding_check(content, snapshot)

        return {
            "content": content,
            "grounding_report": grounding,
            "telemetry": telemetry,
        }
