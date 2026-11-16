# Model 3 Architecture & Endpoints Inventory

**Target System:** Taskove (Gillo) AI Service - Model 3 (LLM Drafts & Agile Chat)  
**Evaluation Scope:** Read-only architectural inventory of draft endpoints, schemas, prompts, and providers.

---

## 1. User-Facing Draft & Assistant Endpoints

| Endpoint | HTTP Method | Target Feature | Auth Roles | Primary Input | Kind |
|---|---|---|---|---|---|
| `/standup/draft` | `POST` | Personal Standup Draft | Developer, PM, TL | Assignee tasks, logged hours, active blockers | `standup` |
| `/sprints/{sprint_id}/retro/draft` | `POST` | Sprint Retrospective Draft | PM, TL | Sprint tasks, slipped tasks, fit check | `retro` |
| `/release-requests/{release_request_id}/draft` | `POST` | Release Summary & Changelog | PM, TL | Release tasks, associated pull requests | `release_summary` |
| `/release-requests/{release_request_id}/explain-block` | `POST` | Release Block Explainer | PM, TL | Uncompleted tasks, open pull requests | `block_explanation` |
| `/chat` | `POST` | Ask the Project Chat | Developer, PM, TL | Natural language query, session history | `chat` |

*(Note: Admin role is strictly forbidden with HTTP 403 on all AI endpoints).*

---

## 2. Pydantic Output Schemas

Defined in `app/services/llm_provider.py` and `app/routers/chat.py`:

1. **`StandupDraftContent`**:
   - `kind`: Literal `"standup"`
   - `developer`: String (developer display name)
   - `yesterday`: List[str] (items completed or in review)
   - `today`: List[str] (planned in-progress or todo items)
   - `blockers`: List[str] (active blockers; mood fields strictly stripped)

2. **`RetroDraftContent`**:
   - `kind`: Literal `"retro"`
   - `sprint_name`: String
   - `metrics`: `RetroMetrics` object:
     - `total_tasks`: int
     - `completed_tasks`: int
     - `slipped_tasks`: int
     - `completion_rate_pct`: int
   - `what_went_well`: List[str]
   - `what_needs_improvement`: List[str]
   - `action_items`: List[str]

3. **`ReleaseSummaryDraftContent`**:
   - `kind`: Literal `"release_summary"`
   - `version`: String
   - `features_delivered`: List[str]
   - `bug_fixes`: List[str]
   - `infrastructure_updates`: List[str]
   - `pull_requests_merged`: List[str]
   - `rollback_plan`: List[str]

4. **`ReleaseNotesDraftContent`**:
   - `kind`: Literal `"release_notes"`
   - `version`: String
   - `highlights`: List[str]
   - `changes`: List[str]

5. **`RollbackPlanDraftContent`**:
   - `kind`: Literal `"rollback_plan"`
   - `version`: String
   - `steps`: List[str]
   - `verification_checks`: List[str]

6. **`BlockExplanationDraftContent`**:
   - `kind`: Literal `"block_explanation"`
   - `version`: String
   - `status`: Literal `"blocked"`
   - `blocking_items_count`: int
   - `diagnostics`: List[str]
   - `recommendations`: List[str]

7. **`ChatResponse`**:
   - `answer`: String (grounded natural-language synthesis)
   - `sources`: List[Dict[str, Any]] (strict provenance from database tools)
   - `session_id`: String
   - `method`: String (`"template"`, `"llm"`, `"llm_template_fallback"`)
   - `model_version`: Optional[str]
   - `generated_by`: String (`"assistant"`)
   - `created_at`: datetime

---

## 3. Versioned Prompt Files

All prompts reside in `prompts/<kind>/<version>.md`:

- `prompts/standup/v1.md`: Instructs generation of standup JSON within `<UNTRUSTED_DATA>` security boundary.
- `prompts/retro/v1.md`: Enforces accurate calculation and reflection of sprint metrics (`total_tasks`, `completed_tasks`, `slipped_tasks`).
- `prompts/release_summary/v1.md`: Synthesizes release categories and rollback steps from merged PRs and completed tasks.
- `prompts/release_notes/v1.md`: Formats highlights and changes for end-user documentation.
- `prompts/rollback_plan/v1.md`: Generates verified rollback procedure and health checks.
- `prompts/block_explanation/v1.md`: Diagnoses specific uncompleted tasks and unmerged PRs blocking a release.
- `prompts/chat/v1.md`: Synthesizes chat responses strictly over `<TOOL_DATA>` blocks without hallucinating missing project data.

---

## 4. Provider Interface & Implementations

Defined in `app/services/llm_provider.py`:

- **`BaseLLMProvider` (ABC):**
  Abstract method: `generate(prompt: str, system_prompt: Optional[str] = None) -> LLMResponse`
  Returns `LLMResponse(raw_text, tokens_in, tokens_out, latency_ms, model_name, provider_name)`.

- **`DeterministicTemplateProvider`:**
  Zero-token fallback provider (`tokens_in=None, tokens_out=None`). Produces deterministic, grounded drafts using template logic and measured latency.

- **`OllamaProvider`:**
  Calls local Ollama daemon at `settings.OLLAMA_BASE_URL` (`POST /api/generate` with `format="json"`). Captures provider-reported `prompt_eval_count` and `eval_count`.

- **`CloudLLMProvider`:**
  Calls OpenAI/Groq compatible cloud endpoints (`POST /chat/completions` with `response_format={"type": "json_object"}`). Captures `prompt_tokens` and `completion_tokens`.

- **`generate_structured_draft()` Orchestrator:**
  Loads versioned prompt template &rarr; Calls active provider &rarr; Pydantic validation &rarr; On validation failure: executes 1&times; retry with error feedback &rarr; On persistent failure: gracefully falls back to `DeterministicTemplateProvider`.

---

## 5. Grounding Checker

Implemented in `run_grounding_check(draft_content: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]`:
- Extracts known task IDs (`task-*`), PR numbers (`pr-*`, `#\d+`), and assignee names from the input snapshot.
- Regex-scans all string values in the generated draft.
- Validates that every entity referenced exists in the known set.
- Produces a report: `{total_references_checked, supported_count, unsupported_count, unsupported_references, grounded: bool}`.
