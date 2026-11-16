# Taskove (Gillo) AI Service

Production-grade, honest, and robust AI & Machine Learning service for Taskove / Gillo Agile Platform.
Built with **FastAPI**, **Motor / MongoDB**, and **Pydantic v2**.

---

## 1. Quickstart & Standalone Mode

The service supports both standalone development mode (in-memory / mock core agile sync) and live production mode.

### Setup Environment
```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Seed MongoDB Demo Data
Populates realistic Agile task snapshots, developer AI profiles, capacity, event history, and historical time entries:
```bash
# Seed relative to current UTC
python scripts/seed_data.py

# Seed relative to a specific reference date with clean wipe of previous demo data
python scripts/seed_data.py --now 2026-11-14T12:00:00Z --reset
```
> **IMPORTANT NOTICE:** Seed data is **synthetic demo data, not evidence of model quality or accuracy**. Behavior on seed data is a demo of the pipeline and the fallbacks, not evidence of model quality. It is provided strictly to exercise API endpoints, state machines, and ML pipelines inside their validated operating ranges.

- All seeded documents are tagged with `source="seed_demo"`.
- `--now <ISO UTC>` (default: current UTC): All sprint dates, event timestamps, and stalled day counts are computed relative to this date, ensuring the demo dataset remains active whenever it is run.
- `--reset`: Wipes existing documents with `{"source": "seed_demo"}` before re-seeding. Upserts use deterministic IDs, guaranteeing idempotent re-runs.
- Exercises Model 1 (Story Points) and Model 2 (Carry-Over Risk) inside their validated operating ranges:
  - 2 active sprints in range ($f \in [0.30, 0.70]$ with $\ge 12$ open tasks): `sprint-04` ($f=0.50$, 13 open tasks), `sprint-05` ($f \approx 0.43$, 12 open tasks).
  - 2 ineligible sprints to verify graceful rule fallback: `sprint-small-active` (< 10 open tasks), `sprint-early-active` ($f = 0.0714 < 0.20$).
  - 3 closed historical sprints (`sprint-01`, `sprint-02`, `sprint-03`) providing baseline data.
  - Realistic task mix: ~18% unestimated (`story_points=None`), 25% added mid-sprint (`sprint_assigned_at > sprint.start_date`), 3 empty descriptions, 1 Arabic task testing the non-English guardrail.
  - Event history mirror: `task_events` collection mirrors Kafka events (`sprint_assigned`, `status_change`, `assignee_change`). Task snapshots derive timestamps directly from events.
  - Zero developer ranking, no mood data, and zero fabricated model suggestions or outcomes.

### Run the Service
```bash
# Starts service at http://localhost:8000
python run.py
```
Visit API Documentation at `http://localhost:8000/docs`.

---

## 2. Configuration & Environment Variables

All settings are strongly validated via Pydantic (`app/config.py`). Configure them in your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | `development`, `testing`, `production` (production refuses default secret) |
| `SERVICE_PORT` | `8000` | Port for the FastAPI server |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `gillo_ai` | MongoDB database name |
| `CORE_AGILE_MODE` | `mock` | `mock` (standalone read models) or `live` |
| `JWT_SECRET` | `dev-insecure-secret` | JWT verification secret key (must be changed in production) |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `MODEL_REGISTRY_DIR` | `models/artifacts` | Directory storing versioned ML model artifacts |
| `MODEL1_ACTIVE_VERSION` | `v2_taskove` | Active version for story point estimator (`model_sp_taskove`) |
| `MODEL2_ACTIVE_VERSION` | `v2_taskove_features` | Active version for sprint carry-over risk classifier |
| `ENABLE_SYNTHETIC_MODELS` | `False` | When `False` (default), synthetic models fall back to `rule_baseline` |
| `DEFAULT_CAPACITY_HOURS` | `60.0` | Default sprint capacity per developer when not explicitly configured |
| `SPRINT_OVERLOAD_RATIO` | `1.00` | Workload ratio triggering sprint overload verdict |
| `SPRINT_TIGHT_RATIO` | `1.00` | Workload ratio triggering sprint tight verdict |
| `SPRINT_UNDERLOAD_RATIO` | `0.70` | Workload ratio identifying underloaded developers |
| `LLM_PROVIDER` | `template` | `template` (deterministic fallback), `ollama`, or `cloud` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for local Ollama instance |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model name (defaults to `qwen2.5:3b`) |
| `CLOUD_LLM_API_KEY` | `None` | API key for OpenAI/Groq compatible cloud provider |
| `CLOUD_LLM_BASE_URL` | `https://api.openai.com/v1`| Base URL for cloud LLM provider |
| `ENABLE_BACKGROUND_JOBS`| `True` | Runs daily risk scan and ratio refresh via APScheduler |

---

## 3. How to Register a New Model Version

Models are never hardcoded in Python code. To deploy a newly trained model (e.g. trained on TAWOS):

1. **Create Artifact Directory**:
   ```
   models/artifacts/<model_name>/<version>/
   ```
   Example: `models/artifacts/model_est/v2_tawos/`

2. **Place Model File and `metadata.json`**:
   The directory must contain the serialized model file (e.g. `model.pkl`) and a `metadata.json`.
   > **Note**: The metric numbers, git commit hash, and file hashes in the snippet below are illustrative placeholder values demonstrating the schema contract.

   ```json
   {
     "name": "model_est",
     "version": "v2_tawos",
     "model_file": "model.pkl",
     "trained_on": "tawos",
     "trained_at": "2026-10-20T12:00:00Z",
     "feature_list": [
       "story_points",
       "dependency_count",
       "historical_median_hours",
       "assignee_user_ratio"
     ],
     "library_versions": {
       "scikit-learn": "1.4.0",
       "xgboost": "2.0.3"
     },
     "metrics": {
       "mae": 5.17,
       "r2": 0.42
     },
     "sha256": "placeholder_sha256_hash_value_here",
     "git_commit": "placeholder_git_commit_sha",
     "description": "Effort regressor trained on TAWOS open-source agile issues (illustrative schema)"
   }
   ```

3. **Feature Contract Validation**:
   The `feature_list` in `metadata.json` **must strictly match** the canonical order defined in `app/ml/features.py`. If there is any mismatch, the registry refuses to load the model and automatically falls back to the deterministic rule baseline with a warning.

4. **Activate the Version**:
   Set the active version in your environment:
   ```env
   MODEL1_ACTIVE_VERSION=v2_tawos
   ```
   No code changes required.

---

## 4. Model 1 Integration: Story Point Estimation (`v2_taskove`)

Model 1 estimates agile task story points from task text and type, replacing legacy synthetic stubs with an empirical LightGBM regression model trained on the TAWOS dataset.

### A. Pipeline & Artifacts
- **Artifact Location:** `models/artifacts/model_sp/v2_taskove/` (copied from `benchmark_v2/artifacts/model1_sp/v2_taskove/`; runtime strictly serves from `models/artifacts/`)
- **Active Configuration:** `MODEL1_ACTIVE_VERSION="v2_taskove"`
- **Architecture:** LightGBM regressor with L1 objective, paired with quantile regressors (Q25, Q50, Q75) for conformal interval estimation, TF-IDF unigram/bigram vectorizer (2,500 features), and One-Hot categorical encoder (17 TAWOS task types).
- **Artifact Checksums:** Validates multi-artifact SHA256 integrity (`model_sp_final.pkl`, `model_sp_q25.pkl`, `model_sp_q50.pkl`, `model_sp_q75.pkl`, `tfidf_vectorizer.pkl`, `ohe_encoder.pkl`, `type_median_fallback.json`).
- **Ordered Input Contract:** Enforces ordered input fields `['title', 'description', 'type']`. Any contract permutation triggers graceful fallback.

### B. Taskove-to-TAWOS Type Mapping
- `feature` &rarr; `Story`
- `bug` &rarr; `Bug`
- `docs` &rarr; `Documentation`
- `refactor` &rarr; `Improvement`
- `infra` / `test` / `backend` / `frontend` / `qa` &rarr; `Technical task`
- `ml` &rarr; `Task`
- `unknown / missing` &rarr; `Task` (sensitivity analysis demonstrated < 0.017 cold-start MAE impact)
- **TODO for Backend Team:** Confirm exact production Taskove task-type vocabulary with the backend team.

### C. Honest Performance Limits & Operational Boundaries
- **Dataset Universe:** Trained on 29,330 TAWOS tasks and evaluated on 6,167 test tasks (chronological split, Epic-excluded). Note that Epics were excluded by product scope definition (Taskove estimates sprint tasks and stories, not multi-month epics).
- **Accuracy:** Test MAE is **1.7607** (snapped Fibonacci MAE is 1.7411).
- **Cold-Start vs. In-Domain Gains:** In-domain improvement over the per-type median baseline is ~10.8% (largely driven by project-specific vocabulary memorization). On unseen cold-start projects (5-fold project cross-validation), transferable gain is **~6.5%**.
- **MdAE Parity:** MdAE is 1.0000, identical to the per-type median baseline (the model does not outperform median baseline on median error).
- **Language Guard:** Trained exclusively on English Jira tickets. If input text contains > 30% non-ASCII letters, the model refuses inference and falls back to `type_median_fallback` with reason `"non_english_text"`.
- **Honest Conformal Uncertainty Interval:** Intervals are calibrated using out-of-project cold-start residuals (`c_cold_start = 0.1147`), producing approximate range; out-of-project 50% coverage ranged 42%-65% across 5 cold-start folds (pooled 50.8% by construction of the constant).
- **Fibonacci Snapping:** Continuous point estimates are snapped to the standard agile Fibonacci scale `{0.5, 1, 2, 3, 5, 8, 13, 21}`.
- **Strict No-Hours-Conversion Policy:** Story points are **never** converted into work hours in inbox suggestions or sprint fit decisions. Empirical studies demonstrate that the correlation between story points and actual logged hours is weak ($r \sim 0.06 - 0.30$). Any legacy hour fields are explicitly annotated with `hours_conversion_note: "unvalidated heuristic"`.
- **Descriptive Fallback Interval:** The interval returned during fallback (`type_median_fallback.json`) represents the empirical interquartile range $[Q25, Q75]$ computed from 29,330 TAWOS training tasks of that specific type (e.g., Bug: `[1.0, 5.0]`, Documentation: `[1.0, 3.0]`, Technical task: `[3.0, 8.0]`, Task: `[2.0, 5.0]`, global: `[1.0, 5.0]`). This is a **descriptive range of past tasks of that type, not a calibrated prediction interval**.
- **Zero HTTP 500 Guarantee:** Any runtime exception, contract violation, non-Latin input, or missing artifact automatically falls back to `type_median_fallback.json` with an explicit provenance label.

### D. Model 2 Carry-Over Risk: Actionable & Non-Misleading TreeSHAP Explanations
- **Individualized Contributions:** Uses LightGBM `pred_contrib=True` (TreeSHAP) on the 23 raw margin features for each individual task row.
- **Actionable Risk Drivers Only:** Client-facing responses display only factors that **raise risk and are actionable**:
  - `status_category_is_todo`: "Not started yet"
  - `days_in_current_status`: "In current status for 1 day" / "In current status for N days"
  - `days_since_last_event`: "No activity for 1 day" / "No activity for N days"
  - `added_after_sprint_start`: "Added after sprint start"
  - `assignee_present`: "Unassigned"
  - `links_depends_on`: "Has 1 prerequisite dependency" / "Has N prerequisite dependencies"
  - `story_points`: "Large estimate (N SP)"
- **Non-Misleading Filtering:** Client responses strictly exclude:
  - Factors that "lower" risk (risk scans only surface risk drivers)
  - Task-type factors (`type_*`)
  - "Unestimated" / `has_sp` indicators
  - Tasks with fewer than 1 raising factor return an empty list with `factor_note: "No specific driver stood out"`
- **Fixed Model Explanation Notice:** All scan-risks responses include `explanation_note: "These factors show what the model weighed on similar past sprints; they are hints, not confirmed causes."`
- **Telemetry Honesty:** The full 23-feature contribution vector (with exact signed margin values, lowering factors, and type coefficients) is permanently recorded in `ai_runs.feature_contributions` for auditability and never exposed to the client.
- **Zero Developer Ranking:** Assignee-level text describes task situation, never person velocity, skill, or ranking.
- **Rule Baseline Parity:** When falling back to `rule_baseline`, reasons apply the exact same actionable filtering and are explicitly marked with `(rule-based)`.

---

## 5. Telemetry & Honest Output Guarantees

Every AI-generated response and `ai_runs` record explicitly provides:
- `method`: `"lightgbm_taskove_v2"`, `"type_median_fallback"`, `"rule_baseline"`, `"llm"`, `"llm_template_fallback"`, or `"template"`.
- `model_version`: The specific artifact or baseline identifier used.
- `generated_by`: User ID or system process that triggered the run.
- `warning`: Explicit notice if a model trained on synthetic data was used or if a fallback was triggered.
- `tokens_in` / `tokens_out`: Exact counts reported by the LLM provider, or `null` if unmeasured (never fabricated).
- `latency_ms`: Real measured execution duration in milliseconds.

---

## 6. Model 3 Evaluation Benchmark & Predeclared Thresholds

Model 3 powers LLM-based draft synthesis (`/standup/draft`, `/sprints/{id}/retro/draft`, `/release-requests/{id}/draft`, `/release-requests/{id}/explain-block`) and read-only Agile Chat (`/chat`).

### A. Evaluation Harness & Protocol
- **Evaluation Dataset:** 70 deterministic scenarios (`reports/model3_eval/eval_set.jsonl`, seed=42) covering the 5 endpoints and realistic edge cases: normal sprints, 0 tasks done, all tasks done, empty blockers, very long titles (>300 chars), empty descriptions, Arabic titles, mood fields in input, adversarial prompt-injection attempts in titles/descriptions, duplicate titles, and >10 blockers.
- **Pure Python Verification:** NO LLM-as-a-judge. All metrics are pure deterministic functions (`app/evaluation/model3_metrics.py`), covered by unit tests in `tests/test_model3_eval_metrics.py`.
- **Predeclared Pass Thresholds:** Locked in `reports/model3_eval/THRESHOLDS.md` before benchmark execution.

### B. Benchmark Results (from `reports/model3_eval/SUMMARY.md`)

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

### C. Mandatory Model 3 Status Disclosure
> [!IMPORTANT]
> `03_qlora_llm_finetuning.ipynb` has NOT been executed and NO fine-tuned LoRA adapter weights exist.
> The service does not claim or run any fine-tuned model. All drafts are currently powered by the validated deterministic template engine with grounding and privacy guardrails.

### D. Human Rating Sheet
An unfilled evaluation rating sheet is exported to `reports/model3_eval/human_rating.csv` containing 20 random deterministic samples across the endpoints with columns `id, provider, endpoint, input_summary, output_text, usefulness, clarity, factual_errors, notes` ready for team review.

---

## 8. Security Architecture & Known Limitations

For comprehensive security audit details, threat modeling, and testing verification, refer to:
- [Security Audit Report](file:///D:/Gillo%20AI%20Service/reports/security/SECURITY_AUDIT.md)
- [Known Limitations & Architecture Roadmap](file:///D:/Gillo%20AI%20Service/reports/security/KNOWN_LIMITATIONS.md)

### Key Security Posture Highlights:
- **Authentication & RBAC:** Strict JWT authentication with pinned `HS256` algorithm and mandatory `exp` claim validation. Admin role is strictly blocked (HTTP 403) from AI inference endpoints.
- **Fail-Closed Project Scoping:** Non-admin tokens must supply a matching `project_id` or explicit `all_projects: true` claim to access sprint, suggestion, chat, and draft endpoints (refusing missing project IDs with HTTP 403).
- **Verified Rate Limiting:** Request rate-limiting keys strictly on verified JWT `sub` claims or client IP addresses, preventing unverified or forged token headers from polluting or exhausting other users' quotas.
- **CORS Hardening:** Configured via `CORS_ORIGINS` (defaults to `["http://localhost:3000"]`). Wildcard origins with credentials are explicitly forbidden in production.
- **Known Limitations & Recommendations:**
  - **SEC-05 (Shared Symmetric Secret):** JWT authentication currently relies on a shared `HS256` secret between Core Agile and AI Service. Roadmap recommendation is to migrate to asymmetric `RS256`/`ES256` with JWKS verification.
  - **SEC-06 (Model Deserialization):** Local `.pkl` artifacts verify SHA-256 integrity against `metadata.json`, which does not guarantee authenticity against local filesystem tampering. Recommend mounting `models/artifacts` as a read-only volume (`:ro`) and migrating to safe formats (`safetensors` or ONNX).
  - **Dependency Maintenance:** `python-jose` is pinned and validated with strict settings, with long-term migration planned to `PyJWT[crypto]`.


The test suite runs hermetically in memory using `mongomock-motor` without requiring a local MongoDB daemon:
```bash
# Run all tests
pytest -v

# Run with test coverage report
pytest --cov=app --cov-report=term-missing tests/
```
The test suite achieves **82% code coverage** with 67 passing tests covering:
- **Bit-for-bit Parity Test:** &ge; 200 issues from the reproduced Epic-excluded TAWOS test set match benchmark artifact predictions within $10^{-6}$.
- **Feature Contract & Mutation Tests:** Swapping field orders triggers contract validation fallback.
- **Language Guard Tests:** Non-Latin inputs trigger English-only guard fallback.
- **Model 2 Carry-Over Risk:** Validated 23-feature pipeline and within-sprint ranking.
- **State Machines & Concurrency:** Atomic 409 conflict resolution and 5-character reason validation.
- **RBAC Matrix:** Strict 403 Forbidden enforcement on Admin role.
- **Demo Seed Dataset End-to-End Suite:** Validates seed idempotency, event mirror consistency, active sprint scans in operating range, graceful fallbacks on missing timestamps or small sprints, and estimation endpoint behaviors across unestimated, empty-description, and non-English tasks.
