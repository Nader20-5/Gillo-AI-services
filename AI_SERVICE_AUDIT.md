# Comprehensive Architecture & Reliability Audit: Gillo AI Service (STEP 0)

**Audit Date:** October 20, 2026  
**Audited Artifacts:** FastAPI routes, Pydantic schemas, ML service modules, database layers, test suite, and environment configurations.  
**Baseline Review:** Built on evidence compiled in `review_evidence_pack.md`.

---

## 1. Component Audit (Routers, Endpoints, Services, Jobs, Collections)

| Layer / Component | File & Line Reference | Classification | Evidence & Operational Reality |
| :--- | :--- | :--- | :--- |
| **System Health** (`GET /health`) | [`app/main.py:46-63`](file:///d:/Gillo%20AI%20Service/app/main.py#L46-L63) | **Real** | Pings MongoDB client and returns service environment, port, and health status. |
| **Sprint Estimation** (`POST /sprints/{id}/estimate`) | [`app/routers/sprints.py:53-205`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L53-L205) | **Real / Integrated (TAWOS Model 1 v2_taskove)** | Integrates real Model 1 LightGBM regressor (`v2_taskove`) trained on TAWOS dataset (Epic-excluded, N=29,330). Ordered input contract `['title', 'description', 'type']`. Canonical type mapping (feature->Story, bug->Bug, docs->Documentation, refactor->Improvement, infra/test/backend/frontend/qa->Technical task, ml->Task; unknown->Task). Language guard (>30% non-Latin letters falls back). Continuous predictions snapped to agile Fibonacci scale `{0.5, 1, 2, 3, 5, 8, 13, 21}`. Honest out-of-project conformal interval calibrated with `c_cold_start = 0.1147` (approximate range; out-of-project 50% coverage ranged 42%-65% across 5 cold-start folds, pooled 50.8% by construction of the constant). Strict no-hours-conversion policy (weak SP-to-time link $r \sim 0.06 - 0.30$; legacy hours tagged as unvalidated heuristic and removed from proposals). Fallback chain guarantees zero HTTP 500 errors. Logs honest `ai_runs` telemetry. |
| **Sprint Fit Check** (`GET /sprints/{id}/fit-check`) | [`app/routers/sprints.py:144-277`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L144-L277) | **Partial / Heuristic** | Compares low/high ranges against capacity. Classifies `fits`, `tight`, `over` via simple comparisons. Reassign proposal logic picks `candidate_tasks[-1]` with a hardcoded `0.89` confidence. No `ai_runs` logged. |
| **Early Risk Scanner** (`POST /sprints/{id}/scan-risks`) | [`app/routers/sprints.py:434-590`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L434-L590) | **Real / Integrated (TAWOS Model 2)** | Integrates real Model 2 LightGBM + Isotonic Calibrator (`v2_taskove_features`) trained on 92,547 TAWOS instances. 23-feature contract strictly enforced via `FEATURE_CONTRACT.md`. Localized within-sprint top-k policy: $k = \min(5, \lceil 0.2 \times n_{\text{open}} \rceil)$. Deduplicates against pending/dismissed suggestions. Honest limits: base rate ~74%, lift ~1.18x, within-sprint AUC ~0.70, not validated on Taskove data. Falls back to deterministic rule baseline if ineligible ($f \notin [0.2, 0.8]$, open tasks < 10, inactive sprint) or missing timestamps. Logs honest `ai_runs` telemetry with calibrated probabilities. |
| **Suggestions List** (`GET /suggestions`) | [`app/routers/suggestions.py:36-73`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L36-L73) | **Partial** | Basic query on `ai_suggestions`. Missing pagination, sorting controls, and robust project boundary scoping. |
| **Suggestion Details** (`GET /suggestions/{id}`) | [`app/routers/suggestions.py:75-96`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L75-L96) | **Real** | Fetches single suggestion by `_id`. |
| **Accept Suggestion** (`POST /suggestions/{id}/accept`) | [`app/routers/suggestions.py:99-146`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L99-L146) | **Partial** | Checks status in memory, updates to `accepted`. Lacks conditional atomic update (`status == "pending"`), does not return 409 Conflict on race conditions, and does not emit Kafka events. |
| **Reject Suggestion** (`POST /suggestions/{id}/reject`) | [`app/routers/suggestions.py:148-184`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L148-L184) | **Partial** | Checks `decision_reason`, sets `rejected`. Lacks conditional atomic update, 409 Conflict handling, and minimum reason length enforcement. |
| **Dismiss Suggestion** (`POST /suggestions/{id}/dismiss`) | [`app/routers/suggestions.py:186-222`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L186-L222) | **Partial** | Checks `decision_reason`, sets `dismissed`. Lacks conditional atomic update and 409 Conflict handling. |
| **Suggestion Comment** (`POST /suggestions/{id}/comments`) | [`app/routers/suggestions.py:224-253`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L224-L253) | **Real** | Appends comment document to `comments` array. |
| **Standup Draft** (`POST /standup/draft`) | [`app/routers/drafts.py:33-102`](file:///d:/Gillo%20AI%20Service/app/routers/drafts.py#L33-L102) | **Fake / Stub** | Calls `AgileDraftService.generate_standup_draft` which is static string formatting. Injects fabricated tokens (`120` in, `85` out), fabricated latency (`0 ms`), and writes `model: "qwen_agile_lora"` to `ai_runs` even though no LoRA model exists. |
| **Retro Draft** (`POST /sprints/{id}/retro/draft`) | [`app/routers/drafts.py:107-168`](file:///d:/Gillo%20AI%20Service/app/routers/drafts.py#L107-L168) | **Fake / Stub** | Static string formatting. Injects fabricated tokens (`250` in, `190` out), fabricated latency, and false `qwen_agile_lora` model tag into `ai_runs`. |
| **Release Summary** (`POST /release-requests/{id}/draft`) | [`app/routers/drafts.py:173-230`](file:///d:/Gillo%20AI%20Service/app/routers/drafts.py#L173-L230) | **Fake / Stub** | Static string formatting. Injects fabricated tokens (`320` in, `220` out) and false `qwen_agile_lora` model tag into `ai_runs`. |
| **Block Explanation** (`POST /release-requests/{id}/explain-block`) | [`app/routers/drafts.py:235-296`](file:///d:/Gillo%20AI%20Service/app/routers/drafts.py#L235-L296) | **Fake / Stub** | Static string formatting. Injects fabricated tokens (`280` in, `170` out) and false `qwen_agile_lora` model tag into `ai_runs`. |
| **Ask the Project Chat** (`POST /chat`) | [`app/routers/chat.py:55-211`](file:///d:/Gillo%20AI%20Service/app/routers/chat.py#L55-L211) | **Partial / Stub** | Regex keyword routing with optional zero-shot Ollama query or string fallback. Hardcoded sources. No role guardrail (Admin is not blocked). Does not log to `ai_runs`. |
| **Estimator Service** (`EffortEstimatorModel`) | [`app/services/estimator.py:1-67`](file:///d:/Gillo%20AI%20Service/app/services/estimator.py#L1-L67) | **Fake / Synthetic** | Wraps `model_est.pkl` (synthetic training data). Heuristic confidence formula `1.0 - (rel_spread * 0.35)`. Lacks fallback mechanism. |
| **Risk Service** (`RiskPredictorModel`) | [`app/services/risk.py:1-108`](file:///d:/Gillo%20AI%20Service/app/services/risk.py#L1-L108) | **Fake / Synthetic** | Wraps `model_risk.pkl` (synthetic training data). Expects 6 synthetic features. Hardcoded severity classification. Lacks fallback mechanism. |
| **LLM Draft Service** (`AgileDraftService`) | [`app/services/llm.py:1-199`](file:///d:/Gillo%20AI%20Service/app/services/llm.py#L1-L199) | **Fake / Stub** | Pure procedural Python string formatting. Contains hardcoded token constants and measures RAM dictionary creation time. |
| **Chat Tools Service** (`chat_tools.py`) | [`app/services/chat_tools.py:1-79`](file:///d:/Gillo%20AI%20Service/app/services/chat_tools.py#L1-L79) | **Real** | Read-only asynchronous MongoDB queries into `task_snapshots`, `sprint_snapshots`, `standup_blockers`, and `release_requests`. |
| **Background Jobs & Schedulers** | `None` | **Stub / Missing** | Completely absent. Architecture planned daily risk scan, suggestion expiration, and estimation ratio refresh, but none are implemented. |
| **Kafka Consumers & Event Workers** | `None` | **Stub / Missing** | Completely absent. No Kafka event ingestion, dead-letter queues, or consumer groups implemented. |
| **Collection: `ai_suggestions`** | [`app/database.py:45-59`](file:///d:/Gillo%20AI%20Service/app/database.py#L45-L59) | **Real** | Stores actionable suggestions. Has compound indexes and unique partial index on `dedupe_key` for `pending` status. |
| **Collection: `task_estimates`** | [`app/database.py:62-69`](file:///d:/Gillo%20AI%20Service/app/database.py#L62-L69) | **Real** | Stores task estimates. Indexes on `task_id` and `sprint_id`. |
| **Collection: `sprint_fit_checks`** | [`app/models/estimates.py:44-52`](file:///d:/Gillo%20AI%20Service/app/models/estimates.py#L44-L52) | **Real** | Stores sprint capacity fit records. |
| **Collection: `ai_drafts`** | [`app/models/drafts.py:10-20`](file:///d:/Gillo%20AI%20Service/app/models/drafts.py#L10-L20) | **Real** | Stores generated standup, retro, and release drafts. |
| **Collection: `chat_sessions`** | [`app/database.py:72-76`](file:///d:/Gillo%20AI%20Service/app/database.py#L72-L76) | **Real** | Stores multi-turn chat sessions with 7-day TTL index on `last_active_at`. |
| **Collection: `ai_runs`** | [`app/models/drafts.py:36-48`](file:///d:/Gillo%20AI%20Service/app/models/drafts.py#L36-L48) | **Fake Telemetry** | Contains fabricated tokens (`120/85`, `250/190`, etc.), fabricated `latency_ms`, and fake model names (`"qwen_agile_lora"`). |
| **Collection: `processed_events`** | [`app/database.py:79-83`](file:///d:/Gillo%20AI%20Service/app/database.py#L79-L83) | **Stub** | Unique index exists for `event_id`, but no consumer ever writes to it. |
| **Collections: Read Models** (`task_snapshots`, `time_entries`, etc.) | [`app/models/read_models.py`](file:///d:/Gillo%20AI%20Service/app/models/read_models.py) | **Real** | Collections populated by `seed_data.py` to serve as the read model store. |

---

## 2. Hardcoded Numbers, Defaults, and Heuristics Audit

1. **`story_points` Default `1.0`:**
   - [`app/routers/sprints.py:70`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L70): `story_points = float(t.get("story_points") or 1.0)`
   - [`app/routers/sprints.py:314`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L314): `story_points = float(t.get("story_points", 1.0))`
   - [`app/models/read_models.py:17`](file:///d:/Gillo%20AI%20Service/app/models/read_models.py#L17): `story_points: Optional[float] = 1.0`
   - *Impact:* Silently masks missing story points and generates normal-looking estimates without flagging uncertainty.
2. **Assignee User Ratio Default `1.0` & `1.05`:**
   - [`app/routers/sprints.py:76`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L76): `user_ratio = 1.0`
   - [`app/routers/chat.py:79`](file:///d:/Gillo%20AI%20Service/app/routers/chat.py#L79): `user_ratio = profile.get("estimation", {}).get("ratio", 1.05)`
   - *Impact:* Static seeded values; never updated from actual completed developer velocity.
3. **Historical Median Hours Multiplier `story_points * 5.0`:**
   - [`app/routers/sprints.py:91`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L91): `medians = [float(h.get("story_points", 1.0)) * 5.0 for h in hist_tasks]`
   - [`app/routers/sprints.py:94`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L94): `historical_median = story_points * 5.0`
   - [`app/routers/sprints.py:328`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L328): `est_hours = latest_est.get("high_hours", story_points * 5.0)`
   - *Impact:* Pure hardcoded heuristic assuming 5.0 hours per story point.
4. **Hardcoded Capacity Ceiling `60.0h / 40.0h`:**
   - [`app/routers/sprints.py:176`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L176): `user_caps.get(uid, 60.0 if uid != "unassigned" else 0.0)`
   - [`app/routers/sprints.py:305`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L305): `float(c.get("capacity_hours", 60.0))`
   - *Impact:* Assumes every developer has 60.0 hours per sprint; no actual team schedule integration.
5. **Sprint Fit Heuristic `cap * 0.70`:**
   - [`app/routers/sprints.py:206`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L206): `elif stats["load_high"] < (cap * 0.70): underloaded_users.append(...)`
   - *Impact:* Hardcoded 70% threshold for underload determination.
6. **Arbitrary Reassign Selection `candidate_tasks[-1]`:**
   - [`app/routers/sprints.py:218`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L218): `target_task = candidate_tasks[-1]`
   - *Impact:* Picks the last task in list regardless of priority, dependencies, or size.
7. **Hardcoded Suggestion Confidence `0.89`:**
   - [`app/routers/sprints.py:234`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L234): `"confidence": 0.89`
   - *Impact:* Fabricated static confidence number.
8. **Heuristic Assignee Load Ratio in Risk Scanner:**
   - [`app/routers/sprints.py:335`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L335): `assignee_load_ratio = 1.25 if cap > 0 and logged_hours > 20 else 0.95`
   - *Impact:* Fabricated ternary logic replacing a real capacity calculation.
9. **Heuristic Confidence Formula in Model 1:**
   - [`app/services/estimator.py:43-44`](file:///d:/Gillo%20AI%20Service/app/services/estimator.py#L43-L44): `confidence = float(np.clip(1.0 - (rel_spread * 0.35), 0.55, 0.95))`
   - *Impact:* Fabricated linear scaling claiming confidence between 55% and 95%.
10. **Fabricated Token Counts in LLM Drafts:**
    - Standup: `tokens_in: 120, tokens_out: 85` ([`app/services/llm.py:57-58`](file:///d:/Gillo%20AI%20Service/app/services/llm.py#L57-L58))
    - Retro: `tokens_in: 250, tokens_out: 190` ([`app/services/llm.py:119-120`](file:///d:/Gillo%20AI%20Service/app/services/llm.py#L119-L120))
    - Release Summary: `tokens_in: 320, tokens_out: 220` ([`app/services/llm.py:157-158`](file:///d:/Gillo%20AI%20Service/app/services/llm.py#L157-L158))
    - Block Explanation: `tokens_in: 280, tokens_out: 170` ([`app/services/llm.py:196-197`](file:///d:/Gillo%20AI%20Service/app/services/llm.py#L196-L197))
    - *Impact:* Fabricated telemetry persisted directly into `ai_runs`.
11. **Fabricated Latency Measurement:**
    - [`app/services/llm.py:53, 115, 153, 192`](file:///d:/Gillo%20AI%20Service/app/services/llm.py#L53): `latency_ms = int((time.time() - start_time) * 1000)`
    - *Impact:* Times in-memory dictionary allocation (0 ms), falsely logged as AI inference time.
12. **Fabricated Model Name in `ai_runs`:**
    - [`app/routers/drafts.py:93, 159, 221, 288`](file:///d:/Gillo%20AI%20Service/app/routers/drafts.py#L93): `"model": "qwen_agile_lora"`
    - *Impact:* Misleads stakeholders into believing a fine-tuned LoRA model generated the drafts.

---

## 3. Role-Based Access Control (RBAC) Audit

| Endpoint | Declared Auth Dependency | Allowed Roles | Admin Status | Vulnerability / Deficiency |
| :--- | :--- | :--- | :--- | :--- |
| `GET /health` | None | Anyone (Public) | Allowed | None (Standard public ping). |
| `POST /sprints/{id}/estimate` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Properly guarded. |
| `GET /sprints/{id}/fit-check` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Properly guarded. |
| `POST /sprints/{id}/scan-risks` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Properly guarded. |
| `GET /suggestions` | `get_current_user` | All except Admin | **403 Forbidden** | Does not restrict developers to their assigned projects. |
| `GET /suggestions/{id}` | `get_current_user` | All except Admin | **403 Forbidden** | Lacks project-level authorization check. |
| `POST /suggestions/{id}/accept` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Lacks project boundary validation. |
| `POST /suggestions/{id}/reject` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Lacks project boundary validation. |
| `POST /suggestions/{id}/dismiss` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Lacks project boundary validation. |
| `POST /suggestions/{id}/comments`| `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Developers cannot comment on suggestions concerning them. |
| `POST /standup/draft` | `get_current_user` | All except Admin | **403 Forbidden** | Allows any caller to generate standup as long as they provide their user ID. |
| `POST /sprints/{id}/retro/draft` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Properly guarded. |
| `POST /release-requests/{id}/draft` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Developers cannot draft release requests. |
| `POST /release-requests/{id}/explain-block` | `require_roles(["PM", "TL"])` | PM, TL | **403 Forbidden** | Properly guarded. |
| `POST /chat` | `get_current_user` | **All (including Admin)** | **ALLOWED (BUG)** | **Admin is NOT blocked from /chat.** Missing RBAC check. |

---

## 4. Privacy & Mood Field Audit

Every repository location referencing "mood":
1. [`tests/test_guardrails.py:6-19`](file:///d:/Gillo%20AI%20Service/tests/test_guardrails.py#L6-L19): Unit test asserting that `mood` and `user_mood` are not keys in `StandupBlocker.model_dump()`.
2. [`app/models/read_models.py:50-53`](file:///d:/Gillo%20AI%20Service/app/models/read_models.py#L50-L53): Model docstring: *"CRITICAL PRIVACY GUARDRAIL: User mood is strictly excluded from this model and database collection."*
3. [`app/services/llm.py:42`](file:///d:/Gillo%20AI%20Service/app/services/llm.py#L42): Comment: `# Blockers (Privacy check: user mood is NOT included)`.
4. [`app/services/chat_tools.py:48`](file:///d:/Gillo%20AI%20Service/app/services/chat_tools.py#L48): Docstring: `(privacy checked: no user mood)`.
5. [`app/routers/drafts.py:60`](file:///d:/Gillo%20AI%20Service/app/routers/drafts.py#L60): Comment: `# 3. Fetch user blockers (Privacy check: user mood is NOT in DB or output)`.
6. [`scripts/seed_data.py:8, 193, 335`](file:///d:/Gillo%20AI%20Service/scripts/seed_data.py#L8): Console print statements asserting mood is not stored.
7. [`AI plan and arch.html:280, 319, 421, 430`](file:///d:/Gillo%20AI%20Service/AI%20plan%20and%20arch.html#L280): Architecture text asserting Kafka consumer strips mood.

**VULNERABILITY IDENTIFIED:**
The guardrail currently exists **only in docstrings, comments, and a passive unit test**. There is no schema-level Pydantic validator (`@model_validator`) that strips or rejects incoming `mood` fields if present in an external JSON payload, and no consumer worker exists to strip them from event streams.

---

## 5. Test Suite Execution Results

- **Command:** `pytest -v`
- **Result:** **2 Passed, 2 Failed** in 130.37 seconds.
  - `tests/test_guardrails.py::test_privacy_guardrail_no_mood_in_standup_blocker`: **PASSED**
  - `tests/test_guardrails.py::test_human_in_the_loop_mandatory_decision_reason`: **PASSED**
  - `tests/test_drafts_and_chat.py::test_drafts_and_chat_lifecycle`: **FAILED** (`ServerSelectionTimeoutError: localhost:27017: [WinError 10061] target machine actively refused it`)
  - `tests/test_sprint_intelligence.py::test_sprint_intelligence_and_risk_scanner_pipeline`: **FAILED** (`ServerSelectionTimeoutError: localhost:27017`)
- **Diagnosis:** The test suite directly attempts to connect to an external live MongoDB server on port 27017. When MongoDB is not running locally, the tests hang for 30 seconds per query before timing out. The suite lacks in-memory database isolation (`mongomock-motor`).

---

## 6. Implementation Action Plan (Steps 1 – 8)

### Phase 1: Core ML Registry & Feature Pipeline (Steps 1 & 2)
1. **Create `app/ml/registry.py`:**
   - Define model registry managing paths, metadata loading, feature list validation, and SHA-256 integrity checks.
   - Establish fallback chains: active model $\rightarrow$ deterministic rule baseline (never HTTP 500).
   - If metadata specifies `trained_on: "synthetic"`, label method as `ml_model_synthetic` and attach an explicit warning.
2. **Create `app/ml/features.py`:**
   - Single canonical feature extraction engine computing strictly prediction-time features from read models.
   - Replaces all ad-hoc heuristics in routers.
3. **Migrate existing synthetic artifacts into versioned folders:**
   - `models/artifacts/model_est/v1_synthetic/` (model_est.pkl + metadata.json)
   - `models/artifacts/model_risk/v1_synthetic/` (model_risk.pkl + metadata.json)

### Phase 2: Honest Telemetry & Schema Grounding (Step 1)
1. **Fix `app/models/drafts.py` (`AIRun` schema):**
   - Add fields: `method` (enum: `"ml_model"`, `"ml_model_synthetic"`, `"rule_baseline"`, `"llm"`, `"llm_template_fallback"`, `"template"`), `model_version`, `generated_by`.
   - Make `tokens_in`, `tokens_out`, and `latency_ms` optional (`None` when not measured).
2. **Update Domain Schemas:**
   - Add `method`, `model_version`, and `generated_by` to `TaskEstimateModel`, `SprintFitCheckModel`, `AISuggestionBase`, and `AIDraft`.
   - Remove hardcoded constants from all services.

### Phase 3: Robust Sprint Estimation & Capacity Fit (Step 3)
1. **Refactor `app/services/estimator.py` & `app/routers/sprints.py`:**
   - Implement empirical baseline quantile estimator (cold-start hierarchy: similar tasks $\rightarrow$ project-level $\rightarrow$ global).
   - Dynamic calculation of `ai_profiles.estimation.ratio` from completed tasks with shrinkage and outlier clipping.
   - Flag missing story points with low confidence rather than silently defaulting to `1.0`.
   - Replace arbitrary task picking in reassignments with deterministic, priority-aware rules.
   - Idempotent estimation (supersedes older active estimates).

### Phase 4: Risk Classifier & Atomic Inbox State Machine (Step 4)
1. **Refactor `app/services/risk.py` & `app/routers/sprints.py`:**
   - Use `features.py` for risk features; compute assignee load from actual capacity and active estimates.
   - Convert contributions into structured reasoning steps.
   - Unique partial index deduplication on `dedupe_key`.
2. **Harden Inbox State Machine (`app/routers/suggestions.py`):**
   - Atomic conditional updates (`status == "pending"`). Return HTTP 409 Conflict if already decided.
   - Require `decision_reason` with minimum length 5 characters on reject/dismiss.
   - Implement `POST /suggestions/{id}/apply` confirmation handler.
   - Add sorting, pagination, and project-scoping to `GET /suggestions`.
3. **Background Jobs (`app/jobs/`):**
   - Implement scheduled risk scan job, suggestion expiration job, and developer ratio refresh job.

### Phase 5: LLM Architecture & Anti-Hallucination Guardrails (Step 5)
1. **Create `app/services/llm_provider.py`:**
   - Abstract provider interface with Ollama, Cloud, and Template implementations.
   - Versioned prompt templates in `prompts/<name>/v1.md`.
2. **Structured Output & Grounding Checks:**
   - Strict Pydantic output validation with 1x retry on validation failure before template fallback.
   - Post-generation entity grounding check (verifies task IDs, titles, PR numbers, and assignees against input snapshot).
   - Delimit untrusted user inputs with safety boundaries to prevent prompt injection.
3. **Refactor `/chat` (`app/routers/chat.py`):**
   - Replace regex branches with read-only tool calling. Block Admin role.

### Phase 6: Privacy, Event Consumers & Test Isolation (Steps 6 & 7)
1. **Central Auth & Privacy:**
   - Enforce RBAC across all endpoints (Admin 403 everywhere on AI).
   - Add schema-level Pydantic validators that reject/strip `mood` fields.
   - Unit test preventing developer ranking/scoring phrases.
2. **Kafka Event Consumer (`app/events/`):**
   - Idempotency via `processed_events`, dead-letter collection for malformed payloads.
3. **Pytest In-Memory Fixtures (`conftest.py`):**
   - Integrate `mongomock-motor` for hermetic test execution with zero external server dependencies.
   - Expand test coverage to 100% of critical paths.

### Phase 7: Documentation & Transparency (Step 8)
1. Update `README.md`, `AI plan and arch.html`, and create `AI_SERVICE_CHANGES.md`.
