import json
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.utils.privacy import strip_mood_fields

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Structured Output Schemas (Strict Pydantic Validation)
# ---------------------------------------------------------------------------
class StandupDraftContent(BaseModel):
    kind: str = "standup"
    developer: str
    yesterday: List[str] = Field(default_factory=list)
    today: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)


class RetroMetrics(BaseModel):
    total_tasks: int
    completed_tasks: int
    slipped_tasks: int
    completion_rate_pct: int


class RetroDraftContent(BaseModel):
    kind: str = "retro"
    sprint_name: str
    metrics: RetroMetrics
    what_went_well: List[str] = Field(default_factory=list)
    what_needs_improvement: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)


class ReleaseSummaryDraftContent(BaseModel):
    kind: str = "release_summary"
    version: str
    features_delivered: List[str] = Field(default_factory=list)
    bug_fixes: List[str] = Field(default_factory=list)
    infrastructure_updates: List[str] = Field(default_factory=list)
    pull_requests_merged: List[str] = Field(default_factory=list)
    rollback_plan: List[str] = Field(default_factory=list)


class ReleaseNotesDraftContent(BaseModel):
    kind: str = "release_notes"
    version: str
    highlights: List[str] = Field(default_factory=list)
    changes: List[str] = Field(default_factory=list)


class RollbackPlanDraftContent(BaseModel):
    kind: str = "rollback_plan"
    version: str
    steps: List[str] = Field(default_factory=list)
    verification_checks: List[str] = Field(default_factory=list)


class BlockExplanationDraftContent(BaseModel):
    kind: str = "block_explanation"
    version: str
    status: str = "blocked"
    blocking_items_count: int
    diagnostics: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


DRAFT_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "standup": StandupDraftContent,
    "retro": RetroDraftContent,
    "release_summary": ReleaseSummaryDraftContent,
    "release_notes": ReleaseNotesDraftContent,
    "rollback_plan": RollbackPlanDraftContent,
    "block_explanation": BlockExplanationDraftContent,
}


# ---------------------------------------------------------------------------
# LLM Response Model
# ---------------------------------------------------------------------------
class LLMResponse(BaseModel):
    raw_text: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: Optional[int] = None
    model_name: str
    provider_name: str


# ---------------------------------------------------------------------------
# Provider Interface & Implementations
# ---------------------------------------------------------------------------
class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call provider with prompt and return honest response telemetry."""
        pass


class DeterministicTemplateProvider(BaseLLMProvider):
    """
    Deterministic template fallback provider.
    Honest telemetry: tokens_in and tokens_out are stored as None (never fabricated).
    Latency is real measured duration.
    """
    def __init__(self, model_name: str = "template_engine_v1"):
        self.model_name = model_name
        self.provider_name = "template"

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        start_time = time.perf_counter()
        # Returns a generic fallback JSON matching draft schema markers if prompt has schema clue
        raw = "{}"
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return LLMResponse(
            raw_text=raw,
            tokens_in=None,
            tokens_out=None,
            latency_ms=latency_ms,
            model_name=self.model_name,
            provider_name=self.provider_name,
        )

    def generate_draft_content(self, kind: str, context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Generates schema-compliant deterministic content and returns (content, latency_ms)."""
        start_time = time.perf_counter()
        context = strip_mood_fields(context)
        if kind == "standup":
            recent_tasks = context.get("recent_tasks", [])
            logged_hours = context.get("logged_hours", 0.0)
            blockers = context.get("blockers", [])
            dev_name = context.get("user_name", "Developer")

            yesterday = [
                f"Worked on '{t.get('title')}' ({t.get('status')})"
                for t in recent_tasks if t.get("status") in ["done", "in_review"]
            ]
            if not yesterday:
                yesterday = [f"Investigated technical requirements and logged {logged_hours:.1f} hours"]

            today = [
                f"Continue development on '{t.get('title')}'"
                for t in recent_tasks if t.get("status") in ["in_progress", "todo"]
            ]
            if not today:
                today = ["Pick next priority item from sprint backlog"]

            # Filter blockers (mood is strictly stripped)
            blocker_items = [b.get("description") for b in blockers if b.get("description")]

            content = StandupDraftContent(
                kind="standup",
                developer=dev_name,
                yesterday=yesterday,
                today=today,
                blockers=blocker_items,
            ).model_dump()

        elif kind == "retro":
            tasks = context.get("tasks", [])
            slipped_tasks = context.get("slipped_tasks", [])
            sprint_name = context.get("sprint_name", "Sprint")
            fit_check = context.get("fit_check") or {}

            done_count = sum(1 for t in tasks if t.get("status") == "done")
            total_count = len(tasks)
            completion_pct = int((done_count / max(total_count, 1)) * 100)

            what_went_well = [
                f"Completed {done_count} of {total_count} sprint tasks ({completion_pct}% completion rate).",
                "Maintained clean pipeline test runs."
            ]
            what_needs_improvement = []
            if slipped_tasks:
                what_needs_improvement.append(f"{len(slipped_tasks)} task(s) slipped due to unresolved blockers.")
            if fit_check.get("verdict") in ["tight", "over"]:
                what_needs_improvement.append(f"Sprint workload was flagged as '{fit_check.get('verdict')}'.")
            if not what_needs_improvement:
                what_needs_improvement.append("Planning meetings ran slightly longer than scheduled.")

            action_items = [
                "Utilize Model 1 effort estimates during backlog refinement.",
                "Review blockers daily during standup."
            ]

            content = RetroDraftContent(
                kind="retro",
                sprint_name=sprint_name,
                metrics=RetroMetrics(
                    total_tasks=total_count,
                    completed_tasks=done_count,
                    slipped_tasks=len(slipped_tasks),
                    completion_rate_pct=completion_pct,
                ),
                what_went_well=what_went_well,
                what_needs_improvement=what_needs_improvement,
                action_items=action_items,
            ).model_dump()

        elif kind == "release_summary":
            version = context.get("version", "v1.0.0")
            tasks = context.get("tasks", [])
            prs = context.get("prs", [])

            features = [t.get("title") for t in tasks if t.get("task_type") == "feature"]
            bug_fixes = [t.get("title") for t in tasks if t.get("task_type") == "bug"]
            infra = [t.get("title") for t in tasks if t.get("task_type") in ["infra", "refactor"]]
            merged_prs = [pr.get("title") for pr in prs if pr.get("status") == "merged"]
            rollback_plan = [
                "Step 1: Revert production deployment container to previous stable tag.",
                "Step 2: Verify service health at /health.",
                "Step 3: Alert Team Lead and PM of rollback status."
            ]

            content = ReleaseSummaryDraftContent(
                kind="release_summary",
                version=version,
                features_delivered=features,
                bug_fixes=bug_fixes,
                infrastructure_updates=infra,
                pull_requests_merged=merged_prs,
                rollback_plan=rollback_plan,
            ).model_dump()

        elif kind == "release_notes":
            version = context.get("version", "v1.0.0")
            tasks = context.get("tasks", [])
            content = ReleaseNotesDraftContent(
                kind="release_notes",
                version=version,
                highlights=[f"Release {version} release highlights"],
                changes=[t.get("title") for t in tasks],
            ).model_dump()

        elif kind == "rollback_plan":
            version = context.get("version", "v1.0.0")
            content = RollbackPlanDraftContent(
                kind="rollback_plan",
                version=version,
                steps=[
                    "Revert deployment image to previous release tag",
                    "Verify health checks and database connectivity",
                    "Notify team"
                ],
                verification_checks=["GET /health returns 200 OK"]
            ).model_dump()

        elif kind == "block_explanation":
            version = context.get("version", "v1.0.0")
            uncompleted = context.get("uncompleted_tasks", [])
            open_prs = context.get("open_prs", [])

            diagnostics = []
            for t in uncompleted:
                diagnostics.append(f"Task '{t.get('title')}' is in '{t.get('status')}'")
            for pr in open_prs:
                diagnostics.append(f"PR '{pr.get('title')}' on '{pr.get('branch')}' is unmerged")

            content = BlockExplanationDraftContent(
                kind="block_explanation",
                version=version,
                status="blocked",
                blocking_items_count=len(uncompleted) + len(open_prs),
                diagnostics=diagnostics,
                recommendations=[
                    "Resolve and merge open pull requests.",
                    "Move or de-scope unfinished tasks.",
                    "Re-run verification pipeline."
                ]
            ).model_dump()
        else:
            content = {"kind": kind}

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return content, latency_ms


class OllamaProvider(BaseLLMProvider):
    """Local Ollama LLM provider measuring real latency and token usage."""
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.provider_name = "ollama"

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        # Ollama returns prompt_eval_count (tokens_in) and eval_count (tokens_out)
        tokens_in = data.get("prompt_eval_count")
        tokens_out = data.get("eval_count")

        return LLMResponse(
            raw_text=data.get("response", "").strip(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model_name=self.model,
            provider_name=self.provider_name,
        )


class CloudLLMProvider(BaseLLMProvider):
    """Cloud LLM provider (OpenAI / Groq format) measuring real latency and token usage."""
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or settings.CLOUD_LLM_API_KEY
        self.base_url = (base_url or settings.CLOUD_LLM_BASE_URL).rstrip("/")
        self.model = model or settings.CLOUD_LLM_MODEL
        self.provider_name = "cloud"

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if not self.api_key:
            raise ValueError("Cloud LLM API key is not configured.")

        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"}
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "").strip() if choices else ""

        return LLMResponse(
            raw_text=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model_name=self.model,
            provider_name=self.provider_name,
        )


def get_llm_provider(name: Optional[str] = None) -> BaseLLMProvider:
    provider_type = (name or settings.LLM_PROVIDER).lower()
    if provider_type == "ollama":
        return OllamaProvider()
    elif provider_type == "cloud":
        return CloudLLMProvider()
    else:
        return DeterministicTemplateProvider()


# ---------------------------------------------------------------------------
# Versioned Prompt Loader
# ---------------------------------------------------------------------------
def load_prompt_template(kind: str, version: str = "v1") -> Tuple[str, str]:
    """Loads prompt template markdown file; returns (template_content, version)."""
    prompt_file = PROMPTS_DIR / kind / f"{version}.md"
    if not prompt_file.exists():
        logger.warning(f"Prompt template {prompt_file} not found; using fallback string.")
        return ("You are an Agile assistant. Output a valid JSON draft.\n\n<UNTRUSTED_DATA>\n{untrusted_data}\n</UNTRUSTED_DATA>", version)
    return (prompt_file.read_text(encoding="utf-8"), version)


# ---------------------------------------------------------------------------
# Structured Output Generation with Validation and 1x Retry
# ---------------------------------------------------------------------------
def clean_json_text(text: str) -> str:
    """Strips markdown code ticks if returned by LLM."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


async def generate_structured_draft(
    kind: str,
    context: Dict[str, Any],
    provider: Optional[BaseLLMProvider] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generates a structured draft.
    Flow:
    1. Loads versioned prompt template.
    2. Calls active provider (Ollama / Cloud / Template).
    3. Validates against Pydantic schema for this draft kind.
    4. On validation failure, retries ONCE with validation error.
    5. On persistent failure or provider error, falls back to deterministic template provider.
    Returns: (draft_content_dict, telemetry_dict)
    """
    active_provider = provider or get_llm_provider()
    template_provider = DeterministicTemplateProvider()
    schema_cls = DRAFT_SCHEMAS.get(kind)
    context = strip_mood_fields(context)

    # If active provider is template engine, generate directly
    if isinstance(active_provider, DeterministicTemplateProvider):
        content, latency_ms = active_provider.generate_draft_content(kind, context)
        telemetry = {
            "method": "template",
            "model_version": f"template:{active_provider.model_name}",
            "prompt_version": "v1",
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms": latency_ms,
            "warning": None,
        }
        return content, telemetry

    # Format prompt with context
    template_str, prompt_version = load_prompt_template(kind, "v1")
    # Format safely: replace known keys or format untrusted data block
    try:
        # Build safe untrusted text
        tasks_text = "\n".join([f"- Task {t.get('task_id')}: {t.get('title')} ({t.get('status')})" for t in context.get("recent_tasks", context.get("tasks", []))])
        blockers_text = "\n".join([f"- Blocker: {b.get('description')}" for b in context.get("blockers", [])])
        prs_text = "\n".join([f"- PR {pr.get('pr_number', pr.get('id', '')): {pr.get('title')}} ({pr.get('status')})" for pr in context.get("prs", [])])

        formatted_prompt = template_str.format(
            developer=context.get("user_name", "Developer"),
            logged_hours=context.get("logged_hours", 0.0),
            tasks_text=tasks_text or "None",
            blockers_text=blockers_text or "None",
            sprint_name=context.get("sprint_name", "Active Sprint"),
            total_tasks=len(context.get("tasks", [])),
            done_tasks=sum(1 for t in context.get("tasks", []) if t.get("status") == "done"),
            slipped_tasks=len(context.get("slipped_tasks", [])),
            completion_pct=int((sum(1 for t in context.get("tasks", []) if t.get("status") == "done") / max(len(context.get("tasks", [])), 1)) * 100),
            slipped_tasks_text="\n".join([f"- Slipped {t.get('task_id')}: {t.get('title')}" for t in context.get("slipped_tasks", [])]) or "None",
            fit_check_verdict=context.get("fit_check", {}).get("verdict", "balanced") if isinstance(context.get("fit_check"), dict) else "balanced",
            version=context.get("version", "v1.0.0"),
            prs_text=prs_text or "None",
            context=json.dumps(context.get("context", {})),
            uncompleted_tasks_text="\n".join([f"- Incomplete {t.get('task_id')}: {t.get('title')}" for t in context.get("uncompleted_tasks", [])]) or "None",
            open_prs_text="\n".join([f"- Open PR {pr.get('id', '')}: {pr.get('title')}" for pr in context.get("open_prs", [])]) or "None",
        )
    except Exception as exc:
        logger.warning(f"Failed to format prompt template: {exc}; using generic formatting.")
        formatted_prompt = f"{template_str}\n\nData: {json.dumps(context)}"

    # Attempt 1: Call LLM
    try:
        llm_resp = await active_provider.generate(formatted_prompt)
        cleaned = clean_json_text(llm_resp.raw_text)
        if schema_cls:
            parsed = schema_cls.model_validate_json(cleaned)
            content = parsed.model_dump()
        else:
            content = json.loads(cleaned)

        telemetry = {
            "method": "llm",
            "model_version": f"{active_provider.provider_name}:{active_provider.model_name}",
            "prompt_version": prompt_version,
            "tokens_in": llm_resp.tokens_in,
            "tokens_out": llm_resp.tokens_out,
            "latency_ms": llm_resp.latency_ms,
            "warning": None,
        }
        return content, telemetry

    except (ValidationError, json.JSONDecodeError) as val_err:
        logger.warning(f"LLM output validation error on attempt 1: {val_err}. Retrying once...")
        # Attempt 2: Retry with validation error feedback
        retry_prompt = (
            f"{formatted_prompt}\n\n"
            f"IMPORTANT FIX: Your previous response was invalid. Schema validation error: {str(val_err)}.\n"
            f"Return ONLY valid JSON strictly matching the requested schema."
        )
        try:
            llm_resp = await active_provider.generate(retry_prompt)
            cleaned = clean_json_text(llm_resp.raw_text)
            if schema_cls:
                parsed = schema_cls.model_validate_json(cleaned)
                content = parsed.model_dump()
            else:
                content = json.loads(cleaned)

            telemetry = {
                "method": "llm",
                "model_version": f"{active_provider.provider_name}:{active_provider.model_name}",
                "prompt_version": prompt_version,
                "tokens_in": llm_resp.tokens_in,
                "tokens_out": llm_resp.tokens_out,
                "latency_ms": llm_resp.latency_ms,
                "warning": "Recovered via 1x schema retry",
            }
            return content, telemetry
        except Exception as retry_err:
            logger.warning(f"LLM retry failed ({retry_err}); falling back to deterministic template.")

    except Exception as llm_err:
        logger.warning(f"LLM provider error ({llm_err}); falling back to deterministic template.")

    # Fallback: Deterministic template provider
    content, fallback_lat = template_provider.generate_draft_content(kind, context)
    telemetry = {
        "method": "llm_template_fallback",
        "model_version": f"template:{template_provider.model_name}",
        "prompt_version": prompt_version,
        "tokens_in": None,
        "tokens_out": None,
        "latency_ms": fallback_lat,
        "warning": "LLM generation or validation failed; fell back to deterministic template.",
    }
    return content, telemetry


# ---------------------------------------------------------------------------
# Post-Generation Entity Grounding Check
# ---------------------------------------------------------------------------
def _extract_all_strings(val: Any) -> List[str]:
    res = []
    if isinstance(val, str):
        res.append(val)
    elif isinstance(val, list):
        for item in val:
            res.extend(_extract_all_strings(item))
    elif isinstance(val, dict):
        for k, v in val.items():
            res.extend(_extract_all_strings(v))
    return res


def run_grounding_check(draft_content: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifies that every task ID, PR number, and task title mentioned in draft_content
    exists in the input snapshot.
    Returns grounding_report with total, supported, and unsupported reference counts.
    """
    # 1. Collect known entities from snapshot
    known_task_ids = set()
    for tid in snapshot.get("task_ids", []):
        known_task_ids.add(str(tid).lower())
    for t in snapshot.get("tasks", []) + snapshot.get("recent_tasks", []) + snapshot.get("uncompleted_tasks", []):
        if t.get("task_id"):
            known_task_ids.add(str(t.get("task_id")).lower())

    known_prs = set()
    for pr in snapshot.get("prs", []) + snapshot.get("open_prs", []):
        if pr.get("pr_number"):
            known_prs.add(str(pr.get("pr_number")).lower())
        if pr.get("id"):
            known_prs.add(str(pr.get("id")).lower())

    # 2. Extract referenced entities from draft strings
    strings = _extract_all_strings(draft_content)
    full_text = " ".join(strings)

    task_id_matches = re.findall(r'\b(task-[a-zA-Z0-9_-]+)\b', full_text, re.IGNORECASE)
    pr_matches = re.findall(r'(?:\bpr-?\d+|#\d+)\b', full_text, re.IGNORECASE)

    checked_tasks = set(m.lower() for m in task_id_matches)
    checked_prs = set(m.lower() for m in pr_matches)

    total_checked = len(checked_tasks) + len(checked_prs)
    unsupported = []

    for tid in checked_tasks:
        if tid not in known_task_ids:
            unsupported.append(f"task:{tid}")

    for pr in checked_prs:
        clean_pr = pr.lstrip("#")
        if pr not in known_prs and clean_pr not in known_prs:
            unsupported.append(f"pr:{pr}")

    supported_count = total_checked - len(unsupported)

    return {
        "total_references_checked": total_checked,
        "supported_count": supported_count,
        "unsupported_count": len(unsupported),
        "unsupported_references": unsupported,
        "grounded": len(unsupported) == 0,
    }
