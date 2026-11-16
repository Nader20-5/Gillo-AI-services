import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings
from app.database import get_database
from app.middleware.auth import CurrentUser, get_current_user, require_roles
from app.services import chat_tools
from app.services.llm_provider import (
    DeterministicTemplateProvider,
    get_llm_provider,
    load_prompt_template,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Ask the Project Chat (Feature 6)"])

# Developer, PM, and TL can chat. Admin gets 403 Forbidden on all AI endpoints.
auth_chat_users = require_roles(["Developer", "PM", "TL"])



def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question about project, sprint, or tasks (max 2000 chars)")
    session_id: Optional[str] = Field(None, max_length=100, description="Optional existing session ID")


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str
    method: str = Field(default="template")
    model_version: Optional[str] = None
    generated_by: str = Field(default="assistant")
    created_at: datetime = Field(default_factory=utc_now)


MAX_TOOL_CALLS = 2


async def route_and_execute_tools(
    query: str,
    user_id: str,
    project_id: str,
    db,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Deterministic read-only tool calling dispatcher.
    Strictly executes under caller's JWT claims (user_id, project_id).
    Capped at MAX_TOOL_CALLS per turn.
    Sources array is built exclusively from actual tool records.
    """
    query_lower = query.lower()
    tool_outputs: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []
    calls_count = 0

    # Tool 1: Tasks
    task_keywords = ["task", "work", "todo", "working on", "assigned", "my plate", "doing", "schedule", "progress"]
    if any(k in query_lower for k in task_keywords) and calls_count < MAX_TOOL_CALLS:
        tasks = await chat_tools.get_my_tasks(user_id=user_id, project_id=project_id, db=db)
        tool_outputs["my_tasks"] = tasks
        sources.append({"source": "task_snapshots", "filter": f"assignee_id={user_id}", "count": len(tasks)})
        calls_count += 1

    # Tool 2: Sprint status
    sprint_keywords = ["sprint", "burn", "active sprint", "timeline", "pace", "how are we doing", "status"]
    if any(k in query_lower for k in sprint_keywords) and calls_count < MAX_TOOL_CALLS:
        sprint_info = await chat_tools.get_sprint_status(sprint_id=None, project_id=project_id, db=db)
        tool_outputs["sprint_status"] = sprint_info
        sources.append({"source": "sprint_snapshots", "sprint_id": sprint_info.get("sprint_id")})
        calls_count += 1

    # Tool 3: Blockers
    blocker_keywords = ["block", "impediment", "stuck", "problem", "hindering", "waiting"]
    if any(k in query_lower for k in blocker_keywords) and calls_count < MAX_TOOL_CALLS:
        blockers = await chat_tools.get_task_blockers(task_id="task-act-03", db=db)
        tool_outputs["blockers"] = blockers
        sources.append({"source": "standup_blockers", "count": len(blockers)})
        calls_count += 1

    # Tool 4: Release
    release_keywords = ["release", "deploy", "version", "launch", "ship"]
    if any(k in query_lower for k in release_keywords) and calls_count < MAX_TOOL_CALLS:
        rel_info = await chat_tools.get_release_status(release_id=None, project_id=project_id, db=db)
        tool_outputs["release_status"] = rel_info
        sources.append({"source": "release_requests", "release_id": rel_info.get("release_id")})
        calls_count += 1

    return tool_outputs, sources


@router.post("", response_model=ChatResponse)
async def ask_the_project(
    req: ChatRequest,
    current_user: CurrentUser = Depends(auth_chat_users),
    db=Depends(get_database),
):
    """
    Feature 6: Read-only project assistant.
    Executes read-only tools under caller's JWT claims.
    Admin gets 403.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    query = req.query.strip()
    query_lower = query.lower()
    user_id = current_user.user_id
    project_id = current_user.project_id or "proj-gillo-core"
    start_time = time.perf_counter()

    # Identity check intent
    identity_keywords = ["who am i", "my role", "my name", "my profile", "know me"]
    if any(k in query_lower for k in identity_keywords):
        profile = await db.ai_profiles.find_one({"user_id": user_id})
        user_name = profile.get("name", user_id) if profile else user_id
        user_role = profile.get("role", current_user.role) if profile else current_user.role
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        sources = [{"source": "ai_profiles", "user_id": user_id}]
        answer = f"You are {user_name}, working as {user_role} on project {project_id}."
        session_id = req.session_id or f"sess-{user_id[:8]}"

        await db.ai_runs.insert_one({
            "feature": "chat",
            "trigger": "user",
            "user_id": user_id,
            "model": "template:identity_resolver",
            "model_version": "template:identity_resolver",
            "method": "template",
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms": latency_ms,
            "status": "success",
            "created_at": utc_now(),
        })

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            method="template",
            model_version="template:identity_resolver",
        )

    # Execute tools under caller's JWT scope
    tool_outputs, sources = await route_and_execute_tools(query, user_id, project_id, db)

    # Check for empty tool results / unanswerable questions
    if not tool_outputs or all(not v for v in tool_outputs.values()):
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        session_id = req.session_id or f"sess-{user_id[:8]}"
        unanswerable_answer = "The project data does not contain this information."

        await db.ai_runs.insert_one({
            "feature": "chat",
            "trigger": "user",
            "user_id": user_id,
            "model": "template:grounding_guard",
            "model_version": "template:grounding_guard",
            "method": "template",
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms": latency_ms,
            "status": "success",
            "created_at": utc_now(),
        })

        return ChatResponse(
            answer=unanswerable_answer,
            sources=sources,
            session_id=session_id,
            method="template",
            model_version="template:grounding_guard",
        )

    # Provider synthesis
    provider = get_llm_provider()
    method = "template"
    model_version = f"template:{provider.model_name}" if hasattr(provider, "model_name") else "template"
    tokens_in = None
    tokens_out = None

    if not isinstance(provider, DeterministicTemplateProvider):
        template_str, _ = load_prompt_template("chat", "v1")
        prompt = template_str.format(tool_data=str(tool_outputs), query=query)
        try:
            llm_resp = await provider.generate(prompt)
            final_answer = llm_resp.raw_text
            method = "llm"
            model_version = f"{provider.provider_name}:{provider.model_name}"
            tokens_in = llm_resp.tokens_in
            tokens_out = llm_resp.tokens_out
        except Exception as exc:
            logger.warning(f"LLM chat generation failed ({exc}); falling back to deterministic template.")
            method = "llm_template_fallback"
            final_answer = None
    else:
        final_answer = None

    # Fallback / deterministic template answer synthesis
    if not final_answer:
        parts = []
        if "my_tasks" in tool_outputs:
            tasks = tool_outputs["my_tasks"]
            active = [t for t in tasks if t.get("status") in ["in_progress", "todo", "blocked"]]
            if active:
                task_str = ", ".join([f"'{t['title']}' ({t['status']})" for t in active])
                parts.append(f"You have {len(active)} active task(s): {task_str}.")
            elif tasks:
                parts.append(f"Your recently completed task was '{tasks[0]['title']}'.")
            else:
                parts.append("You currently have no tasks assigned.")

        if "sprint_status" in tool_outputs:
            sp = tool_outputs["sprint_status"]
            if "error" not in sp:
                parts.append(
                    f"Sprint '{sp.get('name')}' is {sp.get('status')} "
                    f"({sp.get('completed_tasks')}/{sp.get('total_tasks')} completed, {sp.get('blocked_tasks')} blocked)."
                )

        if "blockers" in tool_outputs:
            blks = tool_outputs["blockers"]
            if blks:
                parts.append(f"Noted blocker: '{blks[0]['description']}'.")

        if "release_status" in tool_outputs:
            rel = tool_outputs["release_status"]
            if "error" not in rel:
                parts.append(f"Release '{rel.get('version')}' is currently {rel.get('status')}.")

        final_answer = " ".join(parts) if parts else "The project data does not contain this information."

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    session_id = req.session_id or f"sess-{user_id[:8]}"

    # Log turn in chat_sessions (scoped to caller)
    user_turn = {"role": "user", "content": req.query, "created_at": utc_now()}
    assistant_turn = {
        "role": "assistant",
        "content": final_answer,
        "sources": sources,
        "method": method,
        "model_version": model_version,
        "created_at": utc_now(),
    }
    await db.chat_sessions.update_one(
        {"user_id": user_id, "project_id": project_id},
        {
            "$set": {"last_active_at": utc_now()},
            "$push": {"messages": {"$each": [user_turn, assistant_turn]}},
        },
        upsert=True,
    )

    # Log honest telemetry in ai_runs
    await db.ai_runs.insert_one({
        "feature": "chat",
        "trigger": "user",
        "user_id": user_id,
        "model": model_version,
        "model_version": model_version,
        "method": method,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "status": "success",
        "created_at": utc_now(),
    })

    return ChatResponse(
        answer=final_answer,
        sources=sources,
        session_id=session_id,
        method=method,
        model_version=model_version,
    )
