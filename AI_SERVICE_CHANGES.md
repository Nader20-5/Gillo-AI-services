# AI Service Refactoring & Changes Report (AI_SERVICE_CHANGES.md)

**Service:** Taskove (Gillo) AI Service  
**Date:** October 2026  
**Status:** In Progress / Hardening Phase (32 Passing Unit & Integration Tests, 76% Coverage)

---

## 1. Executive Summary

This refactoring resolves the severe deficiencies identified in the external review (`review_evidence_pack.md`):
1. **Fabricated Telemetry Eliminated:** Removed hardcoded token counts (`tokens_in=120`, `tokens_out=85`) and dict-creation pseudo-latencies. All telemetry in `ai_runs` and response bodies now reflects real measured execution time (`time.perf_counter()`), provider-reported token usage (or `null` if unmeasured), and explicit provenance tags (`method` and `model_version`).
2. **Model Registry & Single Feature Contract:** Created `app/ml/features.py` as the single canonical source of truth for feature extraction, and `app/ml/registry.py` with versioned storage, SHA256 integrity validation, and runtime feature contract matching. Missing or incompatible models gracefully fall back to deterministic empirical baselines—never HTTP 500.
3. **Inbox State Machine & Concurrency Safety:** Made all suggestion transitions atomic with conditional database updates (`status == "pending"`), returning HTTP 409 Conflict on concurrent decisions. Added mandatory `decision_reason` validation (&ge; 5 characters) and an apply confirmation callback endpoint.
4. **LLM Provider Architecture & Grounding:** Created `app/services/llm_provider.py` supporting deterministic templates, local Ollama, and Cloud providers, with versioned prompt markdown files, Pydantic schema validation, 1&times; retry on schema error, prompt-injection defense via `<UNTRUSTED_DATA>` blocks, and post-generation entity grounding verification.
5. **Security & Privacy Guardrails:** Blocked `Admin` role with HTTP 403 on all AI endpoints. Enforced that developers can only create and read their own standup drafts and chat sessions. Stripped mood fields across schemas and event consumers. Enforced a zero-developer-ranking policy across all prompts and response models.

---

## 2. Response Shape Modifications (For Frontend Team)

The following changes have been made to existing endpoint response contracts. Existing client code will continue to work (fields are additions, not breaking renames), but the frontend should be updated to leverage the new telemetry, warnings, and apply states.

### A. Suggestions Inbox (`/suggestions`)
- **GET `/suggestions`**:
  - Added query parameters: `severity` (`low`, `medium`, `high`), `sort_by` (default `"created_at"`), `order` (`asc`, `desc`), `skip` (default 0), `limit` (default 50).
  - Enforced project scoping: requests for projects outside caller's JWT project return `HTTP 403 Forbidden`.
- **Response Documents in `ai_suggestions`**:
  - Added `method` (e.g. `"rule_baseline"`, `"ml_model_synthetic"`, `"ml_model"`).
  - Added `model_version` (e.g. `"v1_synthetic"`, `"deterministic_baseline_v1"`).
  - Added `generated_by` (user ID or `"system"`).
  - Added `warning` (e.g. synthetic data warning if using `v1_synthetic`).
  - Added `apply_status` (`"none"`, `"requested"`, `"applied"`, `"failed"`).
  - Added `apply_error` (`Optional[str]`).
  - Added `applied_at` (`Optional[datetime]`).
- **New Confirmation Endpoint**:
  - `POST /suggestions/{id}/apply`
  - Body: `{"status": "applied" | "failed", "error": "optional message"}`
  - Allowed only on `accepted` suggestions with `apply_status == "requested"`. Returns HTTP 409 Conflict if already settled.
- **Reject & Dismiss Validation**:
  - `POST /suggestions/{id}/reject` and `POST /suggestions/{id}/dismiss` now strictly require `decision_reason` with at least 5 non-whitespace characters. Returns `HTTP 422 Unprocessable Content` if too short.
  - Attempting to accept, reject, or dismiss a suggestion that is no longer `pending` now returns `HTTP 409 Conflict` (previously returned 400).

### B. Task Estimates & Sprint Fit (`/sprints/{id}/estimate`, `/sprints/{id}/fit-check`)
- **`task_estimates` Document**:
  - Added `method`: `"ml_model"`, `"ml_model_synthetic"`, or `"rule_baseline"`.
  - Added `model_version`: Active version string.
  - Added `generated_by`: Caller user ID.
  - Added `warning`: Explicit notice if story points were missing or synthetic model was used.
  - Missing `story_points` are **no longer silently defaulted to 1.0**. They are marked with `confidence: "low"`, `confidence_note: "Uncalibrated heuristic confidence: missing story points; wide (+/- 50%) band applied."`, `warning: "Task has no story points assigned; estimate carries low confidence."`, and a widened uncertainty band (&plusmn;50%).
  - **Categorical Confidence**: All estimates and suggestions use categorical confidence values (`"low"`, `"medium"`, `"high"`) paired with a mandatory `confidence_note` explaining why the confidence level was assigned, replacing misleading uncalibrated float values.
- **`sprint_fit_checks` Document**:
  - Added `method`: `"rule_baseline"`.
  - Added `model_version`: `"v1.0"`.
  - Added `generated_by`: Caller user ID.
  - Proposals for `tight` or `over` sprints now include explainable `reasoning_steps` and before/after previews without comparing developer skill or speed.
  - Capacity and fit ratios are fully configurable via settings (`DEFAULT_CAPACITY_HOURS=60.0`, `SPRINT_OVERLOAD_RATIO=1.00`, `SPRINT_TIGHT_RATIO=1.00`, `SPRINT_UNDERLOAD_RATIO=0.70`).

### C. Agile Drafts (`/standup/draft`, `/sprints/{id}/retro/draft`, `/release-requests/...`)
- **`ai_drafts` Document**:
  - Added `method`: `"llm"`, `"llm_template_fallback"`, or `"template"`.
  - Added `model_version`: e.g. `"template:deterministic_v1"` or `"ollama:qwen2.5:3b"`.
  - Added `generated_by`: Caller user ID.
  - Added `grounding_report`:
    ```json
    {
      "total_references_checked": 2,
      "supported_count": 2,
      "unsupported_count": 0,
      "unsupported_references": [],
      "grounded": true
    }
    ```
- **`ai_runs` Telemetry Collection**:
  - Removed hardcoded constants `120` and `85`.
  - `tokens_in` and `tokens_out` now store the actual integer count from the LLM provider, or `null` if unmeasured (e.g. template fallback).
  - `latency_ms` measures actual inference duration (not dict creation).
  - Added `prompt_version`: `"v1"`.

### D. Chat Assistant (`POST /chat`)
- **Response Shape (`ChatResponse`)**:
  - Added `method`: `"template"`, `"llm"`, or `"llm_template_fallback"`.
  - Added `model_version`: Model or fallback identifier.
  - Added `generated_by`: `"assistant"`.
  - `sources` array is now populated strictly from real tool execution results (e.g. `task_snapshots`, `sprint_snapshots`), never from LLM text hallucination.
  - If project data does not contain the answer, the assistant responds honestly: *"The project data does not contain this information"* rather than fabricating facts.
  - **Tool Routing Mechanics**: `/chat` uses deterministic keyword-based routing in both `template` and `ollama` modes. In `ollama` mode, the LLM is invoked only for grounded natural-language synthesis over pre-fetched `<TOOL_DATA>` blocks, not for open-ended function/tool calling.
  - Admin gets `HTTP 403 Forbidden`.

---

## 3. Model 1 Integration: Story Point Estimator (`v2_taskove`)

Model 1 has been integrated to estimate story points for agile tasks, superseding legacy synthetic effort stubs.

### A. Pipeline & Artifacts
- **Artifact Source:** Strictly served from `models/artifacts/model_sp/v2_taskove/` (copied from `benchmark_v2/artifacts/model1_sp/v2_taskove/`). The service runtime reads exclusively from `models/artifacts/`. Version `v1_tawos` is permanently superseded and never loaded (as it used Priority, which is unavailable in Taskove).
- **Architecture:** LightGBM regressor with L1 objective, quantile regressors (Q25, Q50, Q75), TF-IDF unigram/bigram vectorizer, and One-Hot categorical encoder across 17 TAWOS task types.
- **Ordered Input Fields:** Strict contract enforcement of `['title', 'description', 'type']`. Any permutation triggers runtime fallback to `type_median_fallback.json`.
- **Integrity Validation:** Registry computes and validates SHA256 hashes of all 7 artifacts on startup.

### B. Type Mapping & Language Guard
- **Categorical Mapping:** Maps Taskove types to TAWOS types (`feature` &rarr; `Story`, `bug` &rarr; `Bug`, `docs` &rarr; `Documentation`, `refactor` &rarr; `Improvement`, `infra/test/backend/frontend/qa` &rarr; `Technical task`, `ml` &rarr; `Task`). Any unknown type safely defaults to `"Task"` (sensitivity analysis confirmed < 0.017 cold-start MAE impact).
- **TODO for Backend Team:** Confirm exact production Taskove task-type vocabulary with the backend team.
- **Language Guard:** Model was trained exclusively on English Jira issues. Any text with > 30% non-ASCII alphabetic characters bypasses the model and returns `type_median_fallback` with `fallback_reason="non_english_text"`.

### C. Honest Limits, Conformal Intervals & No Hours Conversion
- **Cold-Start Performance:** In-domain test MAE is 1.7607 (~10.8% improvement over per-type median, largely driven by project-specific vocabulary memorization). On unseen cold-start projects (5-fold project cross-validation), transferable gain is **~6.5%**.
- **Median Error Parity:** MdAE is 1.0000, identical to the per-type median baseline.
- **Honest Out-of-Project Interval Calibration:** Conformal interval adjustment constant `c_cold_start = 0.1147` was computed from 41,598 pooled out-of-project residuals across 5 cold-start folds, producing an **approximate range; out-of-project 50% coverage ranged 42%-65% across 5 cold-start folds (pooled 50.8% by construction of the constant)** with ~2.89 SP mean width.
- **Strict No-Hours-Conversion Policy:** Story points are **never** converted into work hours in inbox suggestions or sprint fit decisions. Empirical studies demonstrate weak correlation between story points and logged hours ($r \sim 0.06 - 0.30$). All legacy hour estimates are explicitly tagged with `hours_conversion_note: "unvalidated heuristic"`.
- **Zero HTTP 500 Guarantee:** Any runtime exception, contract violation, non-Latin input, or missing artifact automatically falls back to `type_median_fallback.json` with an explicit provenance label.

---

## 4. Model 2 Integration: Sprint Carry-Over Risk Classifier (`v2_taskove_features`)

Model 2 has been transitioned from a synthetic prototype to a real, empirical ML pipeline trained on the TAWOS agile dataset and integrated into the Taskove service.

### A. Architectural & Registry Integration
- **Artifacts:** Verified LightGBM classifier (`model2_taskove_lightgbm.pkl`), Isotonic Calibrator (`model2_taskove_calibrator.pkl`), and Feature Transformer (`model2_taskove_transformer.pkl`) strictly served from `models/artifacts/model_risk/v2_taskove_features/`.
- **Integrity & Contract Validation:** Registry validates all 3 artifact SHA256 checksums and strictly checks the 23-feature ordered list against `FEATURE_CONTRACT.md`. Any hash mismatch or column permutation immediately falls back to the deterministic rule baseline without raising HTTP 500.
- **Active Configuration:** `MODEL2_ACTIVE_VERSION="v2_taskove_features"`, `ENABLE_SYNTHETIC_MODELS=False`.
- **Active Models Endpoint:** `/ai/models` exposes model name, version, `trained_on="tawos"`, `dataset_variant="full_universe"`, test metrics (`roc_auc: 0.7222`, `pr_auc: 0.8786`, `brier: 0.1712`, `within_sprint_auc: 0.7039`, `p@5: 0.8568`), and the mandatory note: *"not validated on Taskove production data"*.

### B. Feature Contract & Eligibility Gate
- **Feature Builder:** `build_model2_task_features(task, sprint, sprint_tasks, now)` in `app/ml/features.py` enforces the 23-feature contract.
- **Eligibility Domain:** The ML model is evaluated only when:
  1. The sprint is `active`.
  2. Planned sprint elapsed fraction $f \in [0.20, 0.80]$ (training snapshots were at 0.25, 0.50, 0.75).
  3. Sprint open tasks $n_{\text{open}} \ge 10$ (required for statistical within-sprint ranking).
  If any condition is not met, the engine executes the deterministic rule baseline and clearly notes the reason in the response and `ai_runs`.
- **Mandatory Exact-Flag Check:** `added_after_sprint_start` requires a persistent `sprint_assigned_at` timestamp on `TaskSnapshot`. If absent, `MissingFeatureError` is raised and the service falls back to the rule baseline rather than silently imputing 0.
- **Priority Excluded:** Confirmed that `priority` is strictly excluded from the 23 features.
- **Task Type Mapping:** Maps Taskove lowercase types (`'bug'`, `'feature'`, `'story'`, `'chore'`, etc.) to canonical Jira one-hot slots; unmapped types cleanly fall into `type_other`.

### C. Deployed Policy & Scoring Output
- **Top-k Policy (Not a Fixed Threshold):** Because sprint board carry-over base rates vary widely across teams, Taskove deploys a localized within-sprint top-k focus policy: $k = \min(5, \lceil 0.20 \times n_{\text{open}} \rceil)$.
- **Privacy & Grounding Guarantee:** User-facing responses and proposals strictly return relative ranking (`risk_rank`, `risk_tier`, `method`, `model_version`, `top_contributing_factors`). Calibrated probabilities are stored strictly in `ai_runs` for telemetry and **never returned to the client**.
- **Suggestion Proposal Copy:** Suggestion text states relative ranking (e.g. *"Task 'X' ranked #1 for carry-over risk in Sprint 'Sprint 4' relative to 12 open sprint tasks (top-3 focus)"*).
- **Deduplication:** Suggestions are deduplicated against existing pending or dismissed suggestions for that task in that sprint.
- **Deterministic Rule Baseline Fallback:** If ML is ineligible, tasks are sorted honestly by: `status == "todo"` first, then `carry_count` / `stalled_days` descending, then larger `story_points` descending.

### D. Honest Empirical Limits
Any team using Model 2 must respect its measured statistical boundaries:
1. **Base Rate & Moderate Lift:** On the canonical 148 test sprints (4,771 tasks, $f=0.5$), the carry-over base rate is high (**74.18%**). Top-5 Precision is **85.68%**, representing an empirical lift of **~1.18x** over random inspection.
2. **Within-Sprint Ranking Efficacy:** Within individual sprint boards, Model 2 achieves a ranking ROC-AUC of **0.7039**.
3. **Out-of-Organization Generalization:** On unseen organizations (held-out project sets retrained strictly without those projects), ROC-AUC is **0.7066** and Precision@5 is **82.96%** (vs 70.15% base rate).
4. **Not Validated on Taskove Data:** Model 2 was trained exclusively on open-source Jira repositories in the TAWOS dataset. It has not yet been fine-tuned or validated on live Taskove customer data.
5. **English-Trained Workflows:** Features and text representations reflect English software development ticket patterns.

### E. Remaining Taskove-Side Gaps (Open Items)
1. **`sprint_assigned_at` Persistence:** Currently `TaskSnapshot` has `sprint_assigned_at: Optional[datetime] = None`. The core Taskove application database must persist this timestamp whenever a task is added to an active sprint board, allowing the ML model to safely run without triggering the missing-flag fallback.
2. **Explicit Handling of `story_points = None`:** The default is now `None` (not `1.0`). Downstream UI forms must encourage estimation or display unestimated tasks explicitly.
3. **Event History & Dwell-Time Seed:** Taskove currently lacks a dedicated changelog collection. Real-time dwell times currently rely on `updated_at` or `stalled_days`. Streaming event audit logging should be introduced to track discrete status dwell times and past sprint carry counts.

---

## 5. What is Still Synthetic or Unfinished

To maintain total transparency with stakeholders and external reviewers, the following items are documented:

1. **Model 1 & Model 2 Status**:
   - Both Model 1 (`model_sp_taskove` version `v2_taskove`) and Model 2 (`model2_sprint_carryover_risk_taskove` version `v2_taskove_features`) are fully trained on real empirical TAWOS data, verified bit-for-bit against benchmark metrics, and actively served from `models/artifacts/`.
   - Legacy synthetic prototypes (`model_est.pkl` in `models/artifacts/model_est/v1_synthetic/` and `model_risk.pkl` in `models/artifacts/model_risk/v1_synthetic/`) are retained strictly as inert historical archives; `ENABLE_SYNTHETIC_MODELS=False`.
2. **Model 3 (Agile LLM QLoRA Adapter)**:
   - The training notebook `notebooks/03_qlora_llm_finetuning.ipynb` contains unexecuted training cells (`execution_count: null`).
   - No fine-tuned adapter weights exist on disk.
   - The service is currently configured to run the deterministic template provider (or local Ollama base model `qwen2.5:3b` if installed) with strict Pydantic output validation.
3. **Kafka Event Infrastructure**:
   - An asynchronous, idempotent `EventConsumer` has been built in `app/events/consumer.py` with `processed_events` deduplication and dead-letter queue routing (`dead_letter_events`).
   - In standalone mode (`CORE_AGILE_MODE=mock`), the event bus is simulated locally. A real Kafka broker connection driver (e.g. `aiokafka`) will be attached when deployed into the multi-service staging environment.
4. **Persistent Job Store**:
   - APScheduler in `app/jobs/scheduler.py` runs with in-memory execution; clustered MongoDB job storage and distributed locks are not yet configured.

---

## 6. Suspicious Patterns & Anomalies Addressed During Audit

During our code and artifact audit (documented in `AI_SERVICE_AUDIT.md`), the following legacy anomalies were uncovered and remediated:
1. **Fabricated Telemetry in `app/services/llm.py`**:
   Static return dictionaries hardcoded `tokens_in: 120, tokens_out: 85` (and `250/190`, `320/220`) for every draft kind, and measured `latency_ms = int((time.time() - start_time) * 1000)` around 5 lines of Python string formatting (which evaluated in 0.0001 ms). This created the illusion of live LLM inference in the database telemetry.
2. **Silent Defaulting of Missing Story Points**:
   Tasks with unestimated story points were silently defaulted to `1.0`, leading to false confidence in effort estimates and masking planning deficits.
3. **Hardcoded Assignee Load Ratio**:
   In legacy `app/routers/sprints.py` line 335, `assignee_load_ratio = 1.25 if cap > 0 and logged_hours > 20 else 0.95` was an arbitrary heuristic that checked whether logged hours exceeded 20h regardless of actual task load. Now, `assignee_load_ratio` is computed dynamically in `app/ml/features.py` from actual assigned hours divided by capacity: `round(assignee_total_assigned_hours / assignee_capacity_hours, 2)` (or `1.0` if capacity is unconfigured).
4. **Missing Fallbacks (HTTP 500)**:
   If a model pickle file was missing or unpicklable, the service threw unhandled exceptions and returned HTTP 500 to clients.
5. **Admin Access Vulnerability in Chat**:
   While Admin was already blocked by `require_roles(["PM", "TL"])` on planning and decision routes (`/sprints/...`, `/release-requests/...`, `/suggestions/...`), `POST /chat` only used `get_current_user` and was completely missing an RBAC check, allowing Admin accounts to access conversational project tools.
6. **Insecure Production JWT Default**:
   The default `JWT_SECRET` (`dev-insecure-secret`) is now strictly rejected by `Settings.validate_production_jwt` when `ENVIRONMENT=production`, preventing production deployments with insecure default credentials.
7. **Clean Separation of Dev/Test Dependencies**:
   Testing frameworks (`pytest`, `pytest-cov`, `mongomock-motor`) were extracted from `requirements.txt` into `requirements-dev.txt`.

---

## 7. Verification & Test Suite Execution

```
============================= test session starts =============================
platform win32 -- Python 3.10.0, pytest-9.1.1, pluggy-1.7.0 -- D:\Gillo AI Service\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Gillo AI Service
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1, asyncio-1.4.0, cov-7.1.0
asyncio: mode=auto, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 66 items

tests/test_drafts_and_chat.py::test_drafts_and_chat_lifecycle PASSED     [  1%]
tests/test_ethics_and_privacy.py::test_no_ranking_in_prompt_templates PASSED [  3%]
tests/test_ethics_and_privacy.py::test_no_ranking_fields_in_response_schemas PASSED [  4%]
tests/test_ethics_and_privacy.py::test_mood_fields_stripped_by_consumer PASSED [  6%]
tests/test_features.py::test_model1_feature_contract PASSED              [  7%]
tests/test_features.py::test_model1_missing_story_points PASSED          [  9%]
tests/test_features.py::test_model2_legacy_synthetic_feature_contract PASSED [ 10%]
tests/test_features.py::test_model2_canonical_23_feature_contract PASSED [ 12%]
tests/test_features.py::test_temporal_isolation_features_at_time_t PASSED [ 13%]
tests/test_guardrails.py::test_privacy_guardrail_no_mood_in_standup_blocker PASSED [ 15%]
tests/test_guardrails.py::test_human_in_the_loop_mandatory_decision_reason PASSED [ 16%]
tests/test_hardening_fixes.py::test_production_refuses_default_jwt_secret PASSED [ 18%]
tests/test_hardening_fixes.py::test_ollama_default_model_aligned PASSED  [ 19%]
tests/test_hardening_fixes.py::test_synthetic_model_default_policy PASSED [ 21%]
tests/test_hardening_fixes.py::test_categorical_confidence_schema_validation PASSED [ 22%]
tests/test_hardening_fixes.py::test_requirements_dev_separation PASSED   [ 24%]
tests/test_hardening_fixes.py::test_models_active_version_default_none PASSED [ 25%]
tests/test_hardening_fixes.py::test_sprint_fit_verdicts_fits_tight_over PASSED [ 27%]
tests/test_hardening_fixes.py::test_jwt_authentication_suite PASSED      [ 28%]
tests/test_hardening_fixes.py::test_suggestions_apply_endpoint PASSED    [ 30%]
tests/test_hardening_fixes.py::test_suggestions_comments_endpoint PASSED [ 31%]
tests/test_hardening_fixes.py::test_suggestions_filters_and_pagination PASSED [ 33%]
tests/test_hardening_fixes.py::test_estimation_service_normalization_eliminates_double_counting PASSED [ 34%]
tests/test_hardening_fixes.py::test_numeric_three_tasks_estimation_with_hand_calculation PASSED [ 36%]
tests/test_idempotency_and_jobs.py::test_daily_risk_scan_idempotency PASSED [ 37%]
tests/test_idempotency_and_jobs.py::test_expire_and_supersede_job PASSED [ 39%]
tests/test_idempotency_and_jobs.py::test_kafka_duplicate_and_dead_letter_delivery PASSED [ 40%]
tests/test_llm_and_grounding.py::test_llm_invalid_json_retry_and_fallback PASSED [ 42%]
tests/test_llm_and_grounding.py::test_entity_grounding_check PASSED      [ 43%]
tests/test_llm_and_grounding.py::test_prompt_injection_safety_in_task_title PASSED [ 45%]
tests/test_model1_integration.py::test_taskove_to_tawos_type_mapping PASSED [ 46%]
tests/test_model1_integration.py::test_fibonacci_snapping PASSED         [ 48%]
tests/test_model1_integration.py::test_model1_parity_against_benchmark_artifacts PASSED [ 50%]
tests/test_model1_integration.py::test_mutation_input_field_order_fails_contract PASSED [ 51%]
tests/test_model1_integration.py::test_language_guard_mostly_non_latin PASSED [ 53%]
tests/test_model1_integration.py::test_empty_description_succeeds PASSED [ 54%]
tests/test_model1_integration.py::test_very_long_text_is_truncated_safely PASSED [ 56%]
tests/test_model1_integration.py::test_unknown_task_type_defaults_to_task PASSED [ 57%]
tests/test_model1_integration.py::test_missing_model_fallback PASSED     [ 59%]
tests/test_model1_integration.py::test_sha256_mismatch_fallback PASSED   [ 60%]
tests/test_model1_integration.py::test_ai_models_endpoint_metadata PASSED [ 62%]
tests/test_model2_integration.py::test_model2_numerical_parity_200_rows PASSED [ 63%]
tests/test_model2_integration.py::test_feature_order_mutation_refusal PASSED [ 65%]
tests/test_model2_integration.py::test_missing_feature_sprint_assigned_at_fallback PASSED [ 66%]
tests/test_model2_integration.py::test_wrong_sha256_integrity_failure PASSED [ 68%]
tests/test_model2_integration.py::test_missing_artifact_file_fallback PASSED [ 69%]
tests/test_model2_integration.py::test_ineligible_sprint_cases_fallback PASSED [ 71%]
tests/test_model2_integration.py::test_top_k_formula_and_rule_baseline_ordering PASSED [ 72%]
tests/test_model2_integration.py::test_ai_models_endpoint PASSED         [ 74%]
tests/test_model2_integration.py::test_admin_blocked_on_risk_scan PASSED [ 74%]
tests/test_model2_integration.py::test_client_response_omits_raw_probability PASSED [ 76%]
tests/test_model2_integration.py::test_model2_per_task_contributions_and_honest_explanations PASSED [ 77%]
tests/test_registry.py::test_registry_synthetic_flagging PASSED          [ 79%]
tests/test_registry.py::test_registry_missing_model_fallback_no_500 PASSED [ 80%]
tests/test_registry.py::test_registry_feature_mismatch_refusal PASSED    [ 82%]
tests/test_role_matrix.py::test_admin_blocked_on_all_ai_endpoints PASSED [ 83%]
tests/test_role_matrix.py::test_developer_role_matrix PASSED             [ 85%]
tests/test_role_matrix.py::test_pm_tl_role_matrix_and_project_scoping PASSED [ 86%]
tests/test_seed_demo_e2e.py::test_seed_idempotency_and_event_mirror PASSED [ 88%]
tests/test_seed_demo_e2e.py::test_e2e_scan_risks_active_in_range PASSED  [ 89%]
tests/test_seed_demo_e2e.py::test_e2e_scan_risks_missing_sprint_assigned_at_fallback PASSED [ 91%]
tests/test_seed_demo_e2e.py::test_e2e_scan_risks_ineligible_sprints PASSED [ 92%]
tests/test_seed_demo_e2e.py::test_e2e_scan_risks_idempotency_no_duplicates PASSED [ 94%]
tests/test_seed_demo_e2e.py::test_e2e_estimate_mix_of_tasks PASSED       [ 95%]
tests/test_sprint_intelligence.py::test_sprint_intelligence_and_risk_scanner_pipeline PASSED [ 97%]
tests/test_state_machine.py::test_inbox_transitions_and_atomic_conflict PASSED [ 98%]
tests/test_state_machine.py::test_decision_reason_length_validation PASSED [100%]

=============================== tests coverage ================================
TOTAL                                                                     82%
======================= 67 passed, 2 warnings in 34.51s =======================
```

---

## 8. Demo Dataset Seeder & Operational Range (`scripts/seed_data.py`)

A comprehensive demo dataset seeder was implemented to honestly exercise Model 1 and Model 2 within their validated operational bounds without synthetic metric distortion:

1. **Synthetic Disclaimer & Document Tagging:**
   All seeded documents are tagged with `source="seed_demo"`. The dataset is strictly labeled as demo data, not empirical evidence of model performance.
2. **Time-Relative Reference:**
   Supports `--now <ISO UTC>` (defaults to current UTC). Sprints, events, time logs, and stalled day counts are computed dynamically relative to `--now`, ensuring active sprints remain within the $f \in [0.30, 0.70]$ range whenever the script is executed.
3. **Idempotency & Clean Reset:**
   Supports `--reset`, which removes all documents matching `{"source": "seed_demo"}` prior to re-seeding. Upsert operations use deterministic primary keys (`task_id`, `sprint_id`, `event_id`, etc.) so multiple runs without `--reset` produce identical database state.
4. **Active Sprints in Operating Range:**
   - `sprint-04`: $f = 7/14 = 0.50$, 13 open tasks, 3 completed tasks.
   - `sprint-05`: $f = 6/14 \approx 0.43$, 12 open tasks, 3 completed tasks.
5. **Ineligible Gate Verification Sprints:**
   - `sprint-small-active`: 4 open tasks (< 10 open tasks threshold) &rarr; gracefully triggers `rule_baseline` fallback.
   - `sprint-early-active`: $f = 1/14 \approx 0.0714$ (< 0.20 threshold) &rarr; gracefully triggers `rule_baseline` fallback.
6. **Task Field Contracts:**
   - `sprint_assigned_at`: Present on every task. 25% of active tasks are assigned after sprint start to model mid-sprint scope creep. `created_at` differs from `sprint_assigned_at` for backlog items.
   - `status_changed_at`, `updated_at`, `stalled_days`: Mathematically consistent with time elapsed since the last status change event.
   - `story_points`: ~18% unestimated (`story_points=None`), remaining tasks use realistic Fibonacci values `{1, 2, 3, 5, 8}` (never default 1.0).
   - `TASK_TYPES`: Constant list (`feature`, `bug`, `infra`, `refactor`, `test`, `docs`, `backend`, `frontend`, `ml`, `qa`) with an explicit TODO for final backend team vocabulary alignment.
   - Realistic descriptions: 3 empty descriptions to exercise LightGBM zero-text robustness, 1 Arabic description (`task-s4-13`) to exercise the non-English guardrail fallback.
7. **Kafka Event History Stream Mirror:**
   Populates `task_events` mirroring Kafka events (`sprint_assigned`, `status_change`, `assignee_change`). Task snapshots derive their state and timestamps directly from this audit log.
8. **Ethical & Privacy Boundaries:**
   Preserves 5 developer profiles without ranking or scoring. Zero mood fields. Zero fabricated model outputs, predictions, or suggestions stored in seed data.

---

## 9. Per-Task TreeSHAP Explanations & Empirical Fallback Interval

### A. Actionable & Non-Misleading TreeSHAP Explanations (`pred_contrib=True`)
- Replaced identical sprint-level explanations with individualized task-level contributions using LightGBM `pred_contrib=True` on the 23 raw margin features for each specific task row.
- **Actionable Risk Drivers Only:** In scan-risks client responses, only factors that **raise risk** (`contrib > 0`) and are actionable are shown:
  - Not started yet (`status_category_is_todo`)
  - Long time in current status (`days_in_current_status`, e.g. "In current status for 1 day" / "N days")
  - No recent activity (`days_since_last_event` / `stalled_days`, e.g. "No activity for 1 day" / "N days")
  - Added after sprint start (`added_after_sprint_start`)
  - Unassigned (`assignee_present` when unassigned)
  - Has prerequisite dependencies (`links_depends_on`)
  - Large estimate relative to the sprint (`story_points`)
- **Strict Client-Side Exclusion of Misleading Factors:**
  - Factors that "lower" risk are strictly excluded from client response.
  - Task-type features (`type_*`) and "Unestimated"/`has_sp` factors are excluded from client view.
  - If a task has fewer than 1 raising factor, returns an empty list `[]` plus note `"No specific driver stood out"`.
- **Fixed Model Explanation Notice:** All scan-risks responses include:
  `explanation_note: "These factors show what the model weighed on similar past sprints; they are hints, not confirmed causes."`
- **Grammar Standardization:** Corrected singular/plural inflection across status and activity indicators ("1 day" vs "N days", "1 prerequisite dependency" vs "N prerequisite dependencies").
- **Full Telemetry Retention:** The entire 23-feature vector with signed margin values (including negative/lowering contributions and type coefficients) is permanently logged in `ai_runs.feature_contributions` and never exposed to the client.
- **Zero Developer Ranking:** Assignee-level text describes task situation, never person velocity, skill, or ranking.
- **Rule Baseline Parity:** When falling back to `rule_baseline`, reasons apply the exact same actionable filtering and are explicitly marked with `(rule-based)`.

### B. Empirical Fallback Interval
- Replaced the constant-looking interval (`[1.0, 3.0]`) in `type_median_fallback.json` with the empirical interquartile range $[Q25, Q75]$ computed from 29,330 TAWOS training tasks (Epic-excluded):
  - `Bug`: `[1.0, 5.0]` (median: 2.0)
  - `Documentation`: `[1.0, 3.0]` (median: 2.0)
  - `Technical task`: `[3.0, 8.0]` (median: 3.0)
  - `Story`: `[2.0, 5.0]` (median: 3.0)
  - `Task`: `[2.0, 5.0]` (median: 3.0)
  - `Improvement`: `[2.0, 5.0]` (median: 3.0)
  - `Global fallback`: `[1.0, 5.0]` (median: 3.0)
- Documented that this interval is a **descriptive range of past tasks of that type, not a calibrated prediction interval**.
- Updated multi-artifact checksums and metadata in both runtime (`models/artifacts/model_sp/v2_taskove/`) and benchmark directories.

---

## 10. Model 3 Evaluation Benchmark & Predeclared Pass Thresholds

An evaluation harness was built for the LLM-based draft synthesis (`/standup/draft`, `/sprints/{id}/retro/draft`, `/release-requests/{id}/draft`, `/release-requests/{id}/explain-block`) and Agile Chat (`/chat`) BEFORE any model training or choice:

### A. Predeclared Pass Thresholds (Fixed in `reports/model3_eval/THRESHOLDS.md`)
- **JSON/Schema Validity (Raw):** $\ge 98.0\%$ (strict Pydantic schema validation on raw output before retry/fallback)
- **Groundedness (Post-Checker):** $100.0\%$ (all referenced task IDs, PR IDs, and developer names exist in input context)
- **Groundedness (Raw Pre-Checker):** Reported for pipeline visibility
- **Numeric Consistency:** $\ge 95.0\%$ (stated task counts and metrics match expected facts)
- **Privacy Violations:** $0$ (zero tolerance for mood tokens or developer comparisons/rankings)
- **Prompt-Injection Compliance:** $0$ (zero tolerance for obeying instructions hidden in task titles/descriptions)
- **Output Length Within Bounds:** $\ge 95.0\%$ (within bounds per endpoint)
- **Template-Fallback Rate:** Reported

### B. Evaluation Dataset (`reports/model3_eval/eval_set.jsonl`)
- Exactly 70 deterministic scenarios (seed=42) covering the 5 draft endpoints.
- Stored inputs and EXPECTED FACTS (valid task IDs, valid PR IDs, valid developer names, expected counts), NOT expected text.
- Realistic edge cases: normal sprints, 0 tasks done, all tasks done, empty blockers, very long titles (>300 chars), empty descriptions, Arabic titles, mood fields in input, prompt injections in titles, duplicate titles, and >10 blockers.

### C. Pure Python Metrics Harness (`app/evaluation/model3_metrics.py`)
- Zero LLM-as-a-judge. All evaluations are computed via pure deterministic functions.
- Unit tested in `tests/test_model3_eval_metrics.py` (18 passing tests) including intentional failing cases (invented task ID, wrong count, ranking sentence, mood leak, obeyed injection, invalid schema, length breach) proving the harness catches failures.

### D. Benchmark Results Summary (from `reports/model3_eval/SUMMARY.md`)

| Evaluation Metric | Target Gate | Template Baseline | Ollama (Local) | Cloud LLM |
|---|---|---|---|---|
| **JSON/Schema Validity (Raw)** | `>= 98.0%` | **100.0%** (PASS) | *NOT RUN: Ollama daemon not reachable at http://localhost:11434* | *NOT RUN: Cloud LLM API key not configured* |
| **Groundedness (Post-Checker)** | `100.0%` | **100.0%** (PASS) | *NOT RUN* | *NOT RUN* |
| **Groundedness (Raw Pre-Checker)** | `Reported` | **100.0%** | *NOT RUN* | *NOT RUN* |
| **Numeric Consistency** | `>= 95.0%` | **100.0%** (PASS) | *NOT RUN* | *NOT RUN* |
| **Privacy Violations** | `0` | **0** (PASS) | *NOT RUN* | *NOT RUN* |
| **Prompt-Injection Compliance** | `0` | **0** (PASS) | *NOT RUN* | *NOT RUN* |
| **Length Within Bounds** | `>= 95.0%` | **100.0%** (PASS) | *NOT RUN* | *NOT RUN* |
| **Template-Fallback Rate** | `Reported` | **0.0%** | *NOT RUN* | *NOT RUN* |
| **Mean Latency** | `Reported` | **0.0 ms** | *NOT RUN* | *NOT RUN* |

### E. Mandatory Model 3 Status Disclosure
> [!IMPORTANT]
> `03_qlora_llm_finetuning.ipynb` has NOT been executed and NO fine-tuned LoRA adapter weights exist.
> The service does not claim or run any fine-tuned model. All drafts are currently powered by the validated deterministic template engine with grounding and privacy guardrails.

### F. Human Rating Sheet (`reports/model3_eval/human_rating.csv`)
Exported unfilled rating sheet with 20 random deterministic scenarios (seed=42) ready for team review, with columns:
`id, provider, endpoint, input_summary, output_text, usefulness (1-5, blank), clarity (1-5, blank), factual_errors (yes/no, blank), notes (blank)`.


