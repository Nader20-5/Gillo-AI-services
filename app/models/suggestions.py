from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReasoningStep(BaseModel):
    order: int
    description: str
    evidence: Dict[str, Any] = Field(default_factory=dict, description="SHAP values and rule triggers")


class SuggestionComment(BaseModel):
    author_id: str
    body: str
    created_at: datetime = Field(default_factory=utc_now)


class ChangePreview(BaseModel):
    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)


class AISuggestionBase(BaseModel):
    project_id: str
    sprint_id: str
    task_id: Optional[str] = None
    concerns_user_id: Optional[str] = None
    type: str = Field(..., description="'sprint_fix' or 'risk_warning'")
    action: Optional[str] = Field(None, description="'move', 'split', 'reassign'")
    audience_roles: List[str] = Field(default=["PM", "TL"], description="PM, TL (never Admin)")
    severity: Optional[str] = Field(None, description="'low', 'medium', 'high' (risk only)")
    confidence: str = Field(default="medium", description="'low', 'medium', 'high'")
    confidence_note: Optional[str] = Field(None, description="Explanation of uncalibrated confidence assessment")
    status: str = Field(default="pending", description="'pending', 'accepted', 'rejected', 'dismissed', 'expired', 'superseded'")
    method: str = Field(default="rule_baseline", description="'ml_model', 'ml_model_synthetic', 'rule_baseline'")
    model_version: Optional[str] = None
    generated_by: str = Field(default="system")
    warning: Optional[str] = None
    proposal: Dict[str, Any] = Field(default_factory=dict)
    change_preview: ChangePreview = Field(default_factory=ChangePreview)
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    data_sources: Dict[str, Any] = Field(default_factory=dict)
    run_id: Optional[str] = None
    dedupe_key: str
    decided_by: Optional[str] = None
    decision_reason: Optional[str] = Field(None, description="Mandatory when rejected or dismissed")
    decided_at: Optional[datetime] = None
    apply_status: str = Field(default="none", description="'none', 'requested', 'applied', 'failed'")
    apply_error: Optional[str] = None
    applied_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utc_now)
    comments: List[SuggestionComment] = Field(default_factory=list)


class SuggestionDecisionRequest(BaseModel):
    decision_reason: Optional[str] = Field(None, max_length=2000, description="Required if rejecting or dismissing (min 5, max 2000 chars)")


class ApplyConfirmationRequest(BaseModel):
    status: str = Field(..., description="'applied' or 'failed'")
    error: Optional[str] = Field(None, max_length=1000, description="Optional error message if apply failed")


class SuggestionCommentRequest(BaseModel):
    body: str = Field(..., min_length=1, max_length=2000, description="Comment body (max 2000 chars)")

