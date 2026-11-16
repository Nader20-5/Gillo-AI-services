from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BaselineInfo(BaseModel):
    median_hours: float = 0.0
    user_ratio: float = 1.0
    sample_size: int = 0


class DevOverride(BaseModel):
    hours: float
    at: datetime = Field(default_factory=utc_now)


class TaskEstimateModel(BaseModel):
    task_id: str
    sprint_id: str
    project_id: str
    assignee_id: Optional[str] = None
    baseline: BaselineInfo = Field(default_factory=BaselineInfo)
    low_hours: float
    mid_hours: Optional[float] = None
    high_hours: float
    confidence: str = Field(default="medium", description="'low', 'medium', 'high'")
    confidence_note: Optional[str] = Field(None, description="Explanation of uncalibrated confidence assessment")
    reason: str = ""
    warning: Optional[str] = None
    method: str = Field(default="ml_model", description="'ml_model', 'ml_model_synthetic', 'rule_baseline'")
    model_version: Optional[str] = None
    generated_by: str = Field(default="system")
    dev_override: Optional[DevOverride] = None
    actual_hours: Optional[float] = None
    status: str = Field(default="active", description="'active' or 'superseded'")
    run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class UserCapacityLoad(BaseModel):
    capacity_h: float
    load_low: float
    load_mid: Optional[float] = None
    load_high: float


class SprintFitCheckModel(BaseModel):
    sprint_id: str
    project_id: str
    per_user: Dict[str, UserCapacityLoad] = Field(default_factory=dict)
    verdict: str = Field(..., description="'fits', 'tight', 'over'")
    suggestion_ids: List[str] = Field(default_factory=list)
    method: str = Field(default="rule_baseline", description="'rule_baseline'")
    model_version: Optional[str] = None
    generated_by: str = Field(default="system")
    run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)

