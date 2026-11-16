from typing import Dict, Any, List, Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)


async def get_my_tasks(user_id: str, project_id: str, db) -> List[Dict[str, Any]]:
    """Tool: Retrieves current active tasks assigned to the caller within project scope."""
    query: Dict[str, Any] = {"assignee_id": user_id}
    if project_id:
        query["project_id"] = project_id
    cursor = db.task_snapshots.find(query).limit(20)
    tasks = []
    async for doc in cursor:
        tasks.append({
            "task_id": doc.get("task_id"),
            "title": doc.get("title"),
            "status": doc.get("status"),
            "story_points": doc.get("story_points"),
            "stalled_days": doc.get("stalled_days", 0)
        })
    return tasks


async def get_sprint_status(sprint_id: Optional[str], project_id: str, db) -> Dict[str, Any]:
    """Tool: Retrieves status and task summary for a sprint within project scope."""
    query: Dict[str, Any] = {"sprint_id": sprint_id} if sprint_id else {"status": "active"}
    if project_id:
        query["project_id"] = project_id
    sprint = await db.sprint_snapshots.find_one(query)
    if not sprint:
        return {"error": "Sprint not found"}

    sid = sprint["sprint_id"]
    task_query: Dict[str, Any] = {"sprint_id": sid}
    if project_id:
        task_query["project_id"] = project_id
    cursor = db.task_snapshots.find(task_query)
    tasks = await cursor.to_list(length=100)

    done = sum(1 for t in tasks if t.get("status") == "done")
    blocked = sum(1 for t in tasks if t.get("status") == "blocked")

    return {
        "sprint_id": sid,
        "name": sprint.get("name"),
        "status": sprint.get("status"),
        "total_tasks": len(tasks),
        "completed_tasks": done,
        "blocked_tasks": blocked
    }


async def get_task_blockers(task_id: str, db) -> List[Dict[str, Any]]:
    """Tool: Retrieves active blockers for a specific task (privacy checked: no user mood)."""
    cursor = db.standup_blockers.find({"task_id": task_id, "status": "active"})
    blockers = []
    async for b in cursor:
        blockers.append({
            "blocker_id": b.get("blocker_id"),
            "description": b.get("description"),
            "reported_at": str(b.get("reported_at"))
        })
    return blockers


async def get_release_status(release_id: Optional[str], project_id: Optional[str] = None, db: Any = None) -> Dict[str, Any]:
    """Tool: Retrieves status of a release request within project scope."""
    query: Dict[str, Any] = {"release_request_id": release_id} if release_id else {"status": "blocked"}
    if project_id:
        query["project_id"] = project_id
    rel = await db.release_requests.find_one(query)
    if not rel:
        return {"error": "Release not found"}

    return {
        "release_id": rel.get("release_request_id"),
        "version": rel.get("version"),
        "status": rel.get("status"),
        "tasks_count": len(rel.get("task_ids", []))
    }
