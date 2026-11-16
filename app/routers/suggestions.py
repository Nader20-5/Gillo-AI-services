from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from bson import ObjectId
from app.database import get_database
from app.models.suggestions import (
    AISuggestionBase,
    SuggestionDecisionRequest,
    SuggestionCommentRequest,
    SuggestionComment,
    ApplyConfirmationRequest,
)
from app.middleware.auth import CurrentUser, get_current_user, require_roles
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/suggestions", tags=["AI Suggestions Inbox"])

# Only PM and TL are authorized for Inbox; Admin is strictly forbidden
auth_pm_or_tl = require_roles(["PM", "TL"])


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def serialize_doc(doc: dict) -> dict:
    if not doc:
        return doc
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc


@router.get("", response_model=List[dict])
async def list_suggestions(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    sprint_id: Optional[str] = Query(None, description="Filter by sprint ID"),
    status: Optional[str] = Query(None, description="Filter by status: pending, accepted, rejected, dismissed, expired, superseded"),
    type: Optional[str] = Query(None, description="Filter by type: sprint_fix, risk_warning"),
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    skip: int = Query(0, ge=0, description="Pagination skip offset"),
    limit: int = Query(50, ge=1, le=200, description="Pagination limit"),
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    List AI suggestions from the Inbox.
    Scoped to PM and TL on their accessible projects. Admin is strictly forbidden.
    """
    filter_query = {}

    # Scoping to caller's project
    if current_user.all_projects:
        if project_id:
            filter_query["project_id"] = project_id
    else:
        if project_id and project_id != current_user.project_id:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: cannot access suggestions for project '{project_id}'"
            )
        filter_query["project_id"] = current_user.project_id

    if sprint_id:
        filter_query["sprint_id"] = sprint_id
    if status:
        filter_query["status"] = status
    if type:
        filter_query["type"] = type
    if severity:
        filter_query["severity"] = severity

    ALLOWED_SORT_FIELDS = {"created_at", "severity", "status", "type", "confidence"}
    if sort_by not in ALLOWED_SORT_FIELDS:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid sort_by field '{sort_by}'. Allowed fields: {sorted(ALLOWED_SORT_FIELDS)}"
        )

    sort_direction = -1 if order.lower() == "desc" else 1

    cursor = db.ai_suggestions.find(filter_query).sort(sort_by, sort_direction).skip(skip).limit(limit)
    items = []
    async for doc in cursor:
        items.append(serialize_doc(doc))
    return items


@router.get("/{suggestion_id}", response_model=dict)
async def get_suggestion(
    suggestion_id: str,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """Get single AI suggestion with full reasoning and before/after previews."""
    try:
        oid = ObjectId(suggestion_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid suggestion ID format")

    doc = await db.ai_suggestions.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    # Project scoping check
    if not current_user.all_projects and doc.get("project_id") != current_user.project_id:
        raise HTTPException(status_code=http_status.HTTP_403_FORBIDDEN, detail="Access denied to this project's suggestion")

    return serialize_doc(doc)


@router.post("/{suggestion_id}/accept", response_model=dict)
async def accept_suggestion(
    suggestion_id: str,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Accept an actionable suggestion.
    Transition: pending -> accepted.
    Atomic conditional update returns 409 Conflict if already decided.
    """
    try:
        oid = ObjectId(suggestion_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid suggestion ID format")

    existing = await db.ai_suggestions.find_one({"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    if not current_user.all_projects and existing.get("project_id") != current_user.project_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied to this project's suggestion"
        )

    now = utc_now()
    update_payload = {
        "status": "accepted",
        "decided_by": current_user.user_id,
        "decided_at": now,
        "apply_status": "requested",
    }

    # Atomic transition: only allowed if status == 'pending'
    result = await db.ai_suggestions.update_one(
        {"_id": oid, "status": "pending"},
        {"$set": update_payload}
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"Suggestion already decided: current status is '{existing.get('status')}'."
        )

    # In Standalone Mode: simulate the accepted change directly on the Read Models & emit event
    doc = await db.ai_suggestions.find_one({"_id": oid})
    if settings.CORE_AGILE_MODE == "mock":
        change_preview = doc.get("change_preview", {}).get("after", {})
        task_id = doc.get("task_id")
        if task_id and change_preview:
            await db.task_snapshots.update_one(
                {"task_id": task_id},
                {"$set": {**change_preview, "updated_at": utc_now()}}
            )
        logger.info(f"[Standalone] Emitted AiSuggestionAccepted event for suggestion {suggestion_id}")

    return serialize_doc(doc)


@router.post("/{suggestion_id}/reject", response_model=dict)
async def reject_suggestion(
    suggestion_id: str,
    decision: SuggestionDecisionRequest,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Reject an actionable suggestion.
    Transition: pending -> rejected.
    Mandatory decision_reason (at least 5 characters) required.
    Atomic conditional update returns 409 Conflict if already decided.
    """
    if not decision.decision_reason or len(decision.decision_reason.strip()) < 5:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A non-empty 'decision_reason' of at least 5 characters is strictly required when rejecting a suggestion."
        )

    try:
        oid = ObjectId(suggestion_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid suggestion ID format")

    existing = await db.ai_suggestions.find_one({"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    if not current_user.all_projects and existing.get("project_id") != current_user.project_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied to this project's suggestion"
        )

    now = utc_now()
    update_payload = {
        "status": "rejected",
        "decision_reason": decision.decision_reason.strip(),
        "decided_by": current_user.user_id,
        "decided_at": now,
    }

    result = await db.ai_suggestions.update_one(
        {"_id": oid, "status": "pending"},
        {"$set": update_payload}
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"Suggestion already decided: current status is '{existing.get('status')}'."
        )

    updated_doc = await db.ai_suggestions.find_one({"_id": oid})
    return serialize_doc(updated_doc)


@router.post("/{suggestion_id}/dismiss", response_model=dict)
async def dismiss_suggestion(
    suggestion_id: str,
    decision: SuggestionDecisionRequest,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Dismiss a risk warning.
    Transition: pending -> dismissed.
    Mandatory decision_reason (at least 5 characters) required.
    Atomic conditional update returns 409 Conflict if already decided.
    """
    if not decision.decision_reason or len(decision.decision_reason.strip()) < 5:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A non-empty 'decision_reason' of at least 5 characters is strictly required when dismissing a suggestion."
        )

    try:
        oid = ObjectId(suggestion_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid suggestion ID format")

    existing = await db.ai_suggestions.find_one({"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    if not current_user.all_projects and existing.get("project_id") != current_user.project_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied to this project's suggestion"
        )

    now = utc_now()
    update_payload = {
        "status": "dismissed",
        "decision_reason": decision.decision_reason.strip(),
        "decided_by": current_user.user_id,
        "decided_at": now,
    }

    result = await db.ai_suggestions.update_one(
        {"_id": oid, "status": "pending"},
        {"$set": update_payload}
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"Suggestion already decided: current status is '{existing.get('status')}'."
        )

    updated_doc = await db.ai_suggestions.find_one({"_id": oid})
    return serialize_doc(updated_doc)


@router.post("/{suggestion_id}/apply", response_model=dict)
async def apply_confirmation(
    suggestion_id: str,
    apply_req: ApplyConfirmationRequest,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """
    Confirmation callback for suggestion application ('applied' or 'failed').
    Allowed only on accepted suggestions with apply_status == 'requested'.
    """
    if apply_req.status not in ("applied", "failed"):
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Apply confirmation status must be either 'applied' or 'failed'."
        )

    try:
        oid = ObjectId(suggestion_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid suggestion ID format")

    existing = await db.ai_suggestions.find_one({"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    if not current_user.all_projects and existing.get("project_id") != current_user.project_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied to this project's suggestion"
        )

    if existing.get("status") != "accepted":
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"Cannot confirm apply for suggestion in '{existing.get('status')}' status (must be 'accepted')."
        )
    if existing.get("apply_status") != "requested":
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=f"Apply status has already been settled (current apply_status is '{existing.get('apply_status')}')."
        )

    now = utc_now()
    update_payload = {
        "apply_status": apply_req.status,
        "applied_at": now,
        "apply_error": apply_req.error if apply_req.status == "failed" else None,
    }

    result = await db.ai_suggestions.update_one(
        {"_id": oid, "status": "accepted", "apply_status": "requested"},
        {"$set": update_payload}
    )

    updated_doc = await db.ai_suggestions.find_one({"_id": oid})
    return serialize_doc(updated_doc)


@router.post("/{suggestion_id}/comments", response_model=dict)
async def add_suggestion_comment(
    suggestion_id: str,
    comment_req: SuggestionCommentRequest,
    current_user: CurrentUser = Depends(auth_pm_or_tl),
    db=Depends(get_database),
):
    """Add a collaborative review comment to a suggestion."""
    if not comment_req.body or not comment_req.body.strip():
        raise HTTPException(status_code=422, detail="Comment body cannot be empty.")

    try:
        oid = ObjectId(suggestion_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid suggestion ID format")

    existing = await db.ai_suggestions.find_one({"_id": oid})
    if not existing:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    if not current_user.all_projects and existing.get("project_id") != current_user.project_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied to this project's suggestion"
        )

    comment = SuggestionComment(
        author_id=current_user.user_id,
        body=comment_req.body.strip(),
        created_at=utc_now()
    )

    result = await db.ai_suggestions.update_one(
        {"_id": oid},
        {"$push": {"comments": comment.model_dump()}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    updated_doc = await db.ai_suggestions.find_one({"_id": oid})
    return serialize_doc(updated_doc)
