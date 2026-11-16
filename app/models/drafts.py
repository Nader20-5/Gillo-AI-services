from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AIDraft(BaseModel):
    kind: str = Field(..., description="'standup', 'retro', 'release_summary', 'release_notes', 'rollback_plan', 'block_explanation'")
    owner_user_id: str
    project_id: str
    ref: Dict[str, Any] = Field(default_factory=dict, description="e.g. task_ids, sprint_id, release_request_id")
    content: Dict[str, Any] = Field(default_factory=dict)
    data_sources: Dict[str, Any] = Field(default_factory=dict)
    grounding_report: Optional[Dict[str, Any]] = None
    status: str = Field(default="generated", description="'generated', 'edited', 'submitted', 'discarded'")
    method: str = Field(default="template", description="'llm', 'llm_template_fallback', 'template'")
    model_version: Optional[str] = None
    generated_by: str = Field(default="system")
    run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', 'system'")
    content: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    method: Optional[str] = Field(None, description="'llm', 'tool_call', 'template'")
    model_version: Optional[str] = None
    generated_by: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class ChatSession(BaseModel):
    user_id: str
    project_id: str
    last_active_at: datetime = Field(default_factory=utc_now)
    messages: List[ChatMessage] = Field(default_factory=list)


class AIRun(BaseModel):
    feature: str
    trigger: str = Field(..., description="'schedule', 'user', 'event'")
    user_id: Optional[str] = None
    model: str = Field(..., description="Name of the model or subsystem invoked")
    model_version: Optional[str] = None
    method: str = Field(..., description="'ml_model', 'ml_model_synthetic', 'rule_baseline', 'llm', 'llm_template_fallback', 'template'")
    prompt_version: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: Optional[int] = None
    status: str = "success"  # 'success', 'failed'
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)

