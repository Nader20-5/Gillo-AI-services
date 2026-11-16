# Independent Review Evidence Pack: Taskove (Gillo) AI Service

**Audit Date:** October 20, 2026  
**Audit Scope:** Read-only inspection of repository codebase, trained model artifacts, evaluation logs, and TAWOS Jira benchmark data.  
**Policy:** Strict read-only verification. Zero modifications made. No defensive framing.

---

## A. Data & Split (`scripts/benchmark_tawos.py`)

### 1. Task Table Construction Pipeline
The task table is constructed by reading `tawos_csv/sprint.csv`, `tawos_csv/issue_link.csv`, and streaming `tawos_csv/issue.csv` in chunks of 50,000.

**Exact Code Excerpt ([scripts/benchmark_tawos.py:41-72](file:///d:/Gillo%20AI%20Service/scripts/benchmark_tawos.py#L41-L72)):**
```python
    # 1. Load Sprints
    sprints = pd.read_csv(os.path.join(TAWOS_DIR, "sprint.csv"))
    sprints = sprints[sprints["State"] == "CLOSED"].dropna(subset=["Start_Date", "End_Date"])
    sprints["Start_Date"] = pd.to_datetime(sprints["Start_Date"], errors="coerce")
    sprints["End_Date"] = pd.to_datetime(sprints["End_Date"], errors="coerce")
    sprints = sprints.dropna(subset=["Start_Date", "End_Date"])
    sprints["duration_days"] = (sprints["End_Date"] - sprints["Start_Date"]).dt.total_seconds() / 86400.0
    valid_sprints = sprints[(sprints["duration_days"] >= 5) & (sprints["duration_days"] <= 35)].set_index("ID")

    print(f"Loaded {len(valid_sprints)} valid closed sprints.")

    # 2. Load Issue Links (for dependency count)
    links_df = pd.read_csv(os.path.join(TAWOS_DIR, "issue_link.csv"), usecols=["Issue_ID"])
    dep_counts = links_df["Issue_ID"].value_counts().to_dict()

    # 3. Stream issues matching valid sprints
    sprint_ids_set = set(valid_sprints.index)
    cols = [
        "ID", "Issue_Key", "Title", "Type", "Priority", "Status",
        "Creation_Date", "Resolution_Date", "Story_Point",
        "Resolution_Time_Minutes", "Sprint_ID", "Project_ID"
    ]

    matched_chunks = []
    for chunk in pd.read_csv(os.path.join(TAWOS_DIR, "issue.csv"), usecols=cols, chunksize=50000):
        sub = chunk[chunk["Sprint_ID"].isin(sprint_ids_set)].copy()
        if len(sub) > 0:
            matched_chunks.append(sub)

    df = pd.concat(matched_chunks, ignore_index=True)
```

**Step-by-Step Row Counts and Drops:**
1. **Raw Sprints in `sprint.csv`:** 4,594 sprints.
   - Filter `State == "CLOSED"`: 4,507 retained (87 non-closed dropped).
   - Filter `dropna(Start_Date, End_Date)`: 4,506 retained (1 dropped).
   - Filter duration between 5 and 35 days: **4,344 valid closed sprints** retained (162 dropped).
2. **Raw Issues in `issue.csv`:** 458,232 issues.
   - Filter `Sprint_ID.isin(valid_sprints)`: **40,225 matched issues** retained (**418,007 issues dropped** because they were backlog items never assigned to a sprint, or belonged to open/deleted/invalid sprints).
3. **Model 1 Cycle Time Filter ([scripts/benchmark_tawos.py:224](file:///d:/Gillo%20AI%20Service/scripts/benchmark_tawos.py#L224)):**
   - Filter `Resolution_Time_Minutes > 0`: **38,231 resolved tasks** retained (**1,994 tasks dropped** due to non-positive or missing resolution time).

---

### 2. Exact Split Logic
The split sorts the 40,225 matched rows chronologically by `Sprint_Start` date, and divides by row index at 75%.

**Exact Code Excerpt ([scripts/benchmark_tawos.py:114-118](file:///d:/Gillo%20AI%20Service/scripts/benchmark_tawos.py#L114-L118)):**
```python
    # Chronological Temporal Split by Sprint_Start
    df = df.sort_values("Sprint_Start").reset_index(drop=True)
    split_idx = int(0.75 * len(df))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
```

- **Split Unit:** Row index after sorting by `Sprint_Start` (not a strict calendar date boundary or discrete sprint boundary).
- **Train rows:** 30,168 rows (75.0%).
- **Test rows:** 10,057 rows (25.0%).
- **Cutoff Date:** `2018-03-14 16:46:44`.
- **Train Date Range (`Sprint_Start`):** `2012-02-28 00:49:00` to `2018-03-14 16:46:44`.
- **Test Date Range (`Sprint_Start`):** `2018-03-14 16:46:44` to `2020-10-12 15:34:00`.
- **Temporal Contamination / Overlap Flaw:** 
  1. Because the split divides by row count rather than by discrete sprint IDs, **Sprint ID 3224 appears in BOTH train and test splits**.
  2. Training sprint end dates extend up to `2018-04-02 01:25:00`, which is 19 days *after* the earliest test sprint started (`2018-03-14 16:46:44`). Sprints in train were still active after test sprints commenced.

---

### 3. Issue ID Overlap Check
- **Code checked:** `set(train_df["ID"]) & set(test_df["ID"])`
- **Result:** **NO issue IDs appear in both train and test.**
- **Overlap count:** `0`. Every row has a distinct issue ID.

---

### 4. Issue Status Filter Audit
- **Is there an issue status filter?** **NO.** There is no filter on `Status` or `Resolution` in `benchmark_tawos.py`.
- **Distribution of Statuses in Matched Dataset (40,225 rows):**
  - `Closed`: 20,334 (50.55%)
  - `Done`: 11,419 (28.39%)
  - `Resolved`: 3,794 (9.43%)
  - `Complete`: 3,272 (8.13%)
  - `To Do`: 269 (0.67%)
  - `Open`: 256 (0.64%)
  - `Won't Fix`: 248 (0.62%)
  - `In Progress`: 143 (0.36%)
  - `Invalid`: 121 (0.30%)
  - `Accepted`: 73 (0.18%)
  - `Backlog`: 43 (0.11%)
  - `Reopened`: 35 (0.09%)
  - `In Review`: 35 (0.09%)
  - Other statuses (Gathering Impact, Reviewed, etc.): 184 (0.45%)
- **Are rejected, duplicate, cancelled, or won't-fix issues excluded?** **KEPT.** 248 `Won't Fix` issues, 121 `Invalid` issues, and 269 `To Do` issues were retained and fed into Model 1 and Model 2.

---

### 5. Target Variable Computation
The target variable is `cycle_time_hours`.

**Exact Code Excerpt ([scripts/benchmark_tawos.py:224-225](file:///d:/Gillo%20AI%20Service/scripts/benchmark_tawos.py#L224-L225)):**
```python
    resolved_df = df[df["Resolution_Time_Minutes"] > 0].copy()
    resolved_df["cycle_time_hours"] = np.clip(resolved_df["Resolution_Time_Minutes"] / 60.0, 0.5, 720.0)
```

- **Calculation:** Pre-computed column `Resolution_Time_Minutes` from `issue.csv` divided by `60.0`, clipped between `0.5` hours and `720.0` hours (30 calendar days).
- **Underlying Meaning in TAWOS:** In TAWOS, `Resolution_Time_Minutes` is computed as `(Resolution_Date - Creation_Date)`. 
- **Does it use work logs?** **NO.** `worklog.csv` is never loaded or referenced.
- **Critical Target Defect:** Because the target is `Resolution_Date - Creation_Date`, it measures **backlog wait time plus active work time**, NOT sprint effort. In the test set, **4,185 out of 9,210 tasks (45.4%) hit the 720.0 hour ceiling**, creating severe artificial distortion.

---

## B. Feature Leakage Audit

### 1. Comprehensive Feature Audit (Model 1 Benchmark)

| Feature Name | Exact Code / Formula | Agile Availability | Aggregation Window | Leaky? | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Sprint_Duration` | `(Sprint_End - Sprint_Start).total_seconds() / 86400.0` | At sprint planning | Per individual sprint | **No** | Planned sprint length is known before development begins. |
| `backlog_age_days` | `clip((Sprint_Start - Creation_Date).days, 0, 365)` | At sprint planning (for pre-existing tasks) | Per individual task | **Yes (Construct Leak)** | 1) 15,827 tasks were created *after* sprint start (clipped to 0). 2) More critically, the target is `(Resolution - Creation) = (Sprint_Start - Creation) + (Resolution - Sprint_Start)`. The feature is literally a sub-component of the target variable. |
| `dependency_count` | `links_df["Issue_ID"].value_counts()` from `issue_link.csv` | Ongoing / Post-sprint | Entire dataset (all time) | **Yes** | Aggregates all links in `issue_link.csv` with zero temporal filter. Links created mid-sprint or post-sprint are leaked into planning time. |
| `Story_Point_Clean` | `df["Story_Point"].fillna(sp_median)` | At estimation / sprint planning | Full dataset (`train + test`) | **Questionable** | `sp_median` was calculated over the *entire* DataFrame (train and test combined = 3.0), leaking test distribution into training. Furthermore, Jira `Story_Point` records final values, which can change mid-sprint. |
| `type_*` (5 types) | `(df["Type"] == t).astype(float)` | At task creation | Per individual task | **No** | Task type (Task, Bug, Sub-task, Story, Improvement) is fixed metadata. |
| `prio_*` (5 priorities) | `(df["Priority"] == p).astype(float)` | At task triage | Per individual task | **No** | Static issue priority field. |

---

### 2. Specific Feature Audits

1. **`assignee_completion_rate`:**
   - **Status in TAWOS Benchmark:** **DOES NOT EXIST.** Not used in `scripts/benchmark_tawos.py`.
   - **Status in Synthetic / Runtime Service:** In `scripts/train_model1.py` and `app/routers/sprints.py`, it is named `assignee_user_ratio`. It is read directly as a static scalar from `ai_profiles.estimation.ratio` (e.g., hardcoded to 1.05 for user Nader, 0.95 for Layla). It is **never dynamically calculated** over past sprints or historical windows in the codebase.
2. **`sprint_velocity`:**
   - **Status:** **DOES NOT EXIST AS AN ML FEATURE.** It is never computed or fed into Model 1 or Model 2 in either the benchmark or synthetic models. It only appears as hardcoded string text in `scripts/prepare_ml_datasets.py` (`"Velocity: 42 points"`) and in chat responses.
3. **`sprint_planned_points`:**
   - **Status:** **DOES NOT EXIST AS A DISTINCT SNAPSHOT.** In `benchmark_tawos.py`, the model uses `Story_Point_Clean`, which is the final `Story_Point` column in `issue.csv`. No baseline point snapshot captured at sprint planning is used.
4. **`dependencies_count`:**
   - **Status:** **LEAKY.** Evaluated on all links in `tawos_csv/issue_link.csv` without timestamp checking. In real agile development, dependencies and blocker links are frequently discovered and logged mid-sprint or during testing.
5. **`team_capacity`:**
   - **Status in TAWOS Benchmark:** **DOES NOT EXIST.**
   - **Status in Runtime Service:** Read from `db.capacity` collection in `app/routers/sprints.py:165`. It is a static configuration (`capacity_hours = 60.0` for developers, `40.0` for TLs seeded via `scripts/seed_data.py`), not a calculated empirical velocity.

---

## C. Model 1 (Task Effort Estimation)

### 1. Saved Artifact on Disk
- **File Path:** [`models/artifacts/model_est.pkl`](file:///d:/Gillo%20AI%20Service/models/artifacts/model_est.pkl)
- **File Size:** 1,671,192 bytes (1.67 MB)
- **File Timestamp:** October 16, 2026
- **Architecture & Hyperparameters:** `EffortEstimatorModel` bundle containing `TfidfVectorizer(max_features=32)` and three `XGBRegressor` quantile models:
  ```python
  XGBRegressor(objective="reg:quantileerror", quantile_alpha=[0.25, 0.50, 0.75],
               n_estimators=250, learning_rate=0.04, max_depth=5,
               tree_method="hist", subsample=0.8, colsample_bytree=0.75,
               reg_alpha=0.5, reg_lambda=1.5, random_state=42)
  ```
- **CRITICAL FINDING:** `model_est.pkl` was trained on **100% SYNTHETIC DATA** generated by `scripts/prepare_ml_datasets.py`. It has never ingested a single row of the TAWOS dataset. `scripts/benchmark_tawos.py` evaluates an in-memory XGBoost model but **never serializes it to disk**.

---

### 2. Real Test-Set Metrics on TAWOS Split
Evaluated strictly on the 9,210 resolved tasks in the TAWOS test split:
- **$R^2$ Score:** `0.3591` (on raw hours)
- **Mean Absolute Error (MAE):** `140.11 hours`
- **Median Absolute Error (MdAE):** `72.57 hours`
- **Root Mean Squared Error (RMSE):** `221.80 hours`
- **Pred(25) (within 25% of actual):** `45.93%`
- **Pred(50) (within 50% of actual):** `61.18%`
- **Interval Coverage [$P_{25}, P_{75}$]:** `86.3%`

---

### 3. Comparison Against Baselines (Same Test Set, $N = 9,210$)

| Model / Baseline | $R^2$ Score | MAE (hours) | MdAE (hours) | Pred(25) | Pred(50) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1:** Global Training Median (548.9h) | `-0.0217` | `252.14` | `214.52` | `11.41%` | `62.66%` |
| **Baseline 2:** Issue Type Training Median | `-0.2120` | `248.77` | `238.67` | `39.87%` | `59.65%` |
| **Baseline 3:** Linear Regression on Story Points alone | `-0.0108` | `259.32` | `284.11` | `11.56%` | `66.97%` |
| **Baseline 3b:** Story Points $\times$ Training Avg Hours/Pt | `-5.1334` | `393.25` | `208.25` | `27.34%` | `44.66%` |
| **Model 1: XGBoost Quantile Regressor ($P_{50}$)** | **`0.3591`** | **`140.11`** | **`72.57`** | **`45.93%`** | **`61.18%`** |

**Does Model 1 beat Baseline 2?**
- **Yes.** Model 1 achieves an MAE of 140.11h compared to Baseline 2's 248.77h (a **108.66 hour reduction**, or 43.7% error decrease). MdAE drops from 238.67h to 72.57h (a **166.10 hour reduction**, or 69.6% error decrease).

---

### 4. Ablation: Removing Leaky & Non-Planning Features
- Baseline Model 1 Full $R^2$: `0.3591`
- Model 1 without `dependency_count`: $R^2 = \mathbf{0.3437}$
- Model 1 without `dependency_count` and without `Sprint_Duration`: $R^2 = \mathbf{0.3539}$
- **Model 1 with ONLY Static Planning Features (Story Points + Issue Type + Priority), removing `backlog_age_days`:** $R^2 = \mathbf{-0.1003}$ (MAE = 264.83h)

**Conclusion:** Model 1's entire predictive capability ($R^2 > 0$) comes from `backlog_age_days` (`Sprint_Start - Creation_Date`). Because the target is `Resolution_Date - Creation_Date`, the model is merely learning that tasks that spent months waiting in the backlog have longer total cycle times. When limited to true sprint planning features, **the model has negative $R^2$ and cannot beat a simple mean**.

---

### 5. Serving Audit (`app/services/estimator.py`)
- **Loaded Artifact:** Loaded at [app/services/estimator.py:61-65](file:///d:/Gillo%20AI%20Service/app/services/estimator.py#L61-L65):
  ```python
  model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "artifacts", "model_est.pkl")
  with open(model_path, "rb") as f:
      _model_instance = pickle.load(f)
  ```
  It loads the **synthetic-trained pickle**, which expects `title`, `task_type`, `story_points`, `dependency_count`, `historical_median`, and `assignee_ratio`.
- **Runtime Defaults for Missing Features:**
  - `story_points`: defaults to `1.0` if missing ([app/routers/sprints.py:70](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L70)).
  - `assignee_ratio`: defaults to `1.0` if assignee has no profile record ([app/routers/sprints.py:76](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L76)).
  - `historical_median`: falls back to `story_points * 5.0` if no completed historical tasks exist ([app/routers/sprints.py:94](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L94)).
  - `dependency_count`: defaults to `0` (`len(dependencies)`).
  - `title`: defaults to `""`.
- **Fallback Path:** **NONE.** If `model_est.pkl` is missing, `load_estimator_model()` raises `FileNotFoundError`. There is no try/except fallback block in `sprints.py`; the endpoint crashes with HTTP 500.

---

## D. Model 2 (Sprint Risk Classifier)

### 1. Saved Artifact on Disk
- **File Path:** [`models/artifacts/model_risk.pkl`](file:///d:/Gillo%20AI%20Service/models/artifacts/model_risk.pkl)
- **File Size:** 1,561,782 bytes (1.56 MB)
- **File Timestamp:** October 16, 2026
- **Contents:** `RiskPredictorModel` bundle containing `LGBMClassifier(n_estimators=200, learning_rate=0.04, num_leaves=31, max_depth=6)` and a `shap.TreeExplainer(model)`.
- **CRITICAL FINDING:** `model_risk.pkl` was trained on **SYNTHETIC DATA** expecting 6 features (`stalled_days`, `dependency_depth`, `active_blockers`, `assignee_load_ratio`, `logged_vs_estimated_ratio`, `story_points`). The TAWOS benchmark model was never saved to disk.

---

### 2. Unit of Prediction
- **Unit of Prediction:** **A single task within a sprint.**
- It does NOT predict sprint-level collapse; it predicts whether a specific task will slip past the sprint close boundary.

---

### 3. Binary Target Definition
Defined at [scripts/benchmark_tawos.py:86-87](file:///d:/Gillo%20AI%20Service/scripts/benchmark_tawos.py#L86-L87):
```python
    slipped = df["Resolution_Date"].isna() | (df["Resolution_Date"] > df["Sprint_End"])
    df["is_carried_over"] = slipped.astype(int)
```
- A task is labeled positive (`1`, carried over) if it has no resolution date (unresolved) OR if its resolution date occurred strictly after `Sprint_End`. Otherwise `0`.

---

### 4. Real Test-Set Metrics on TAWOS Split ($N = 10,057$)
- **Total test instances:** `10,057`
- **Class Balance:** Positive (`is_carried_over = 1`): `5,388` (**53.57%**); Negative (`0`): `4,669` (**46.43%**).
- **ROC-AUC:** `0.6281`
- **PR-AUC:** `0.6458`
- **Accuracy:** `0.5948` (59.48%)
- **Precision:** `0.6215` (62.15%)
- **Recall:** `0.6234` (62.34%)
- **F1-Score:** `0.6224`
- **Brier Score:** `0.2371` (Brier Skill Score: `+4.66%` over null model)
- **Full Confusion Matrix:**
  $$\begin{pmatrix} \text{TN: } 2623 & \text{FP: } 2046 \\ \text{FN: } 2029 & \text{TP: } 3359 \end{pmatrix}$$

---

### 5. Comparison Against Baselines (Same Test Set, $N = 10,057$)

| Model / Baseline | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1:** Dummy Majority Class (`1`) | `0.5000` | `0.5357` | `0.5357` | `1.0000` | `0.6977` |
| **Baseline 2:** Logistic Regression on Story Points alone | **`0.6294`** | `0.4668` | `0.8421` | `0.0059` | `0.0118` |
| **Baseline 3:** Logistic Regression on Full Scaled Features | `0.5296` | `0.5357` | `0.5357` | `1.0000` | `0.6977` |
| **Model 2: LightGBM (Full 14 Features)** | **`0.6281`** | **`0.5948`** | **`0.6215`** | **`0.6234`** | **`0.6224`** |

**Critical Observation:** Model 2's ROC-AUC (`0.6281`) does **NOT** beat a single-variable Logistic Regression on Story Points alone (`0.6294`). Its accuracy (59.48%) is only 5.9% higher than a dummy baseline that always predicts positive (53.57%).

---

### 6. Serving & Feature Pipeline Disconnect (`app/services/risk.py`)
- **Loaded Artifact:** Loaded at [app/services/risk.py:102-106](file:///d:/Gillo%20AI%20Service/app/services/risk.py#L102-L106):
  ```python
  model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "artifacts", "model_risk.pkl")
  with open(model_path, "rb") as f:
      _risk_model_instance = pickle.load(f)
  ```
- **Runtime Features in `sprints.py:338-345`:**
  1. `stalled_days`: extracted from task document.
  2. `dependency_depth`: count of dependencies.
  3. `active_blockers`: count of active blockers from `db.standup_blockers`.
  4. `assignee_load_ratio`: hardcoded heuristic: `1.25 if cap > 0 and logged_hours > 20 else 0.95`.
  5. `logged_vs_estimated_ratio`: `logged_hours / max(est_hours, 1.0)`.
  6. `story_points`: float from task.
- **COMPLETE PIPELINE DISCONNECT:**
  - The runtime pipeline expects **6 synthetic features**.
  - The TAWOS benchmark script trained on **14 completely different features** (`Sprint_Duration`, `backlog_age_days`, `dependency_count`, `Story_Point_Clean`, 5 type one-hots, 5 priority one-hots).
  - **The model evaluated in the benchmark cannot run in the production service, and the model in the production service has never been trained on real data.**

---

## E. Model 3 (Story Point / Complexity Predictor & LLM Summarizer)

### 1. Model Artifact on Disk
- **Path:** [`models/Modelfile`](file:///d:/Gillo%20AI%20Service/models/Modelfile)
- **Size:** 879 bytes
- **Timestamp:** October 16, 2026
- **Contents:** Text configuration for Ollama:
  ```dockerfile
  FROM qwen2.5:3b
  SYSTEM """You are Gillo, an expert Agile Project Intelligence AI Assistant..."""
  PARAMETER temperature 0.2
  ```
- **Artifact Weights:** **NONE.** No adapter weights, no LoRA checkpoints, and no model parameter files exist in the repository.

---

### 2. Training & Fine-Tuning Execution Verification
- **Notebook Inspected:** [`notebooks/03_qlora_llm_finetuning.ipynb`](file:///d:/Gillo%20AI%20Service/notebooks/03_qlora_llm_finetuning.ipynb)
- **Execution Count Status:**
  - Cell 1 (markdown): markdown
  - Cell 2 (pip install): `"execution_count": null`, `"outputs": []`
  - Cell 3 (imports): `"execution_count": null`, `"outputs": []`
  - Cell 4 (config setup): `"execution_count": null`, `"outputs": []`
  - Cell 5 (load dataset): `"execution_count": null`, `"outputs": []`
- **Result:** **NO MODEL WAS TRAINED, FINE-TUNED, OR ADAPTED.** The notebook was never executed on GPU or CPU.

---

### 3. What is Actually Running in Production (`app/services/llm.py`)
In `app/services/llm.py`, Model 3 is implemented entirely as **static procedural Python string templates**:
```python
    @staticmethod
    def generate_standup_draft(user_id, project_id, user_name, recent_tasks, logged_hours, blockers):
        start_time = time.time()
        yesterday_items = [f"Worked on '{t.get('title')}' ({t.get('status')})" for t in recent_tasks ...]
        today_items = [f"Continue development on '{t.get('title')}'" for t in recent_tasks ...]
        content = {
            "kind": "standup", "developer": user_name,
            "yesterday": yesterday_items, "today": today_items, "blockers": ...
        }
        latency_ms = int((time.time() - start_time) * 1000)
        return {"content": content, "latency_ms": latency_ms, "tokens_in": 120, "tokens_out": 85}
```
- **Nature of Service:** Zero neural networks. Pure Python dictionary creation.
- **Tokens In / Tokens Out:** Hardcoded static constants (`120` in, `85` out).
- **Chat Endpoint (`app/routers/chat.py`):** Rule-based regex keyword matching with an optional zero-shot call to an un-adapted local Ollama daemon or Groq cloud API.

---

### 4. Origin of Previous Claims ("98.4% Accuracy" & "<300ms Inference")
- **Claim: "98.4% Accuracy":**
  - **Verification:** Exhaustive search across the entire repository (Python, JSON, HTML, Markdown, and text files) found **ZERO occurrences** of "98.4%".
  - **Verdict:** The metric was **invented in presentation slides/documentation** and never had any code, benchmark, or empirical basis.
- **Claim: "<300ms Inference":**
  - **Verification:** Traced directly to [app/services/llm.py:53](file:///d:/Gillo%20AI%20Service/app/services/llm.py#L53):
    ```python
    latency_ms = int((time.time() - start_time) * 1000)
    ```
  - **Verdict:** `start_time` and `latency_ms` measured the execution time of **Python dictionary string concatenation in RAM** (which takes ~0.1 ms, hence always `< 300 ms`). It was mislabeled in project claims as deep learning LLM inference latency.

---

## F. Current State Summary Table

| Model | Purpose | Training Data | Model Family / Architecture | Deployment Status | Primary Metric on Real Data (TAWOS) | Strongest Baseline Metric | Primary Risk / Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | Task Effort / Cycle Time Estimation | Synthetic (Serving) / TAWOS (Benchmark only) | XGBoost Quantile Regressor ($P_{25}, P_{50}, P_{75}$) | Trained with caveats (Serving model is 100% synthetic; TAWOS benchmark unexported) | $R^2 = 0.3591$, MAE = 140.11h, MdAE = 72.57h, Pred(25) = 45.93% | Issue Type Median: MAE = 248.77h, MdAE = 238.67h ($R^2 = -0.2120$) | Positive $R^2$ is an artifact of `backlog_age_days` leaking into cycle time target; removing it collapses $R^2$ to $-0.1003$. |
| **Model 2** | Mid-Sprint Task Carry-Over Risk | Synthetic (Serving) / TAWOS (Benchmark only) | LightGBM Binary Classifier + SHAP TreeExplainer | Trained with caveats (Serving model uses synthetic features; disconnected from TAWOS) | ROC-AUC = 0.6281, PR-AUC = 0.6458, Accuracy = 59.48%, F1 = 0.6224 | Logistic Regression on Story Points alone: ROC-AUC = 0.6294 | Does not outperform a 1-feature baseline on Story Points, and production runtime features do not match training features. |
| **Model 3** | Agile Drafts & Project Assistant | None (Synthetic JSONL created, never trained) | Static Python Templates (serving) / Ollama Qwen-3B (chat zero-shot) | Untrained stub / Rule-based template placeholder | **N/A** (No model trained; evaluation count null) | N/A | No neural adapter exists; drafts are deterministic string templates and claimed "<300ms" was dict allocation. |
