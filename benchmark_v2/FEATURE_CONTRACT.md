# Model 2 Feature Contract: Taskove-Compatible Model

**Model Identifier:** `model2_sprint_carryover_risk_taskove`  
**Version:** `v2_taskove_features`  
**Target Variable:** Sprint Carry-Over (Binary: 1 = task open at sprint end; 0 = task completed before sprint end)  
**Taskove Integration Source:** [`app/models/read_models.py`](file:///d:/Gillo%20AI%20Service/app/models/read_models.py) (`TaskSnapshot`, `SprintSnapshot`)  
**Associated Artifacts:** [`benchmark_v2/artifacts/model2/v2_taskove_features/`](file:///d:/Gillo%20AI%20Service/benchmark_v2/artifacts/model2/v2_taskove_features)

---

## 1. Overview & Architectural Intent

The Taskove-compatible model is trained strictly on the subset of 23 features extractable from Taskove's live application read models without requiring full historical event streaming, external organization metrics, or team identity proxies.

- **Excluded from Full Model:** `project_prev_3_carry_rate` (historical inertia), `sprint_duration` (non-monotonic project identity proxy), `carry_count` (requires global issue history across all past boards), `priority` (missing from `TaskSnapshot`), and raw changelog counts (`status_changes_so_far`, `reassignments_so_far`).
- **Feature Vector Shape:** 23 numerical floats ordered identically to the model's training matrix.

---

## 2. Ordered Feature Contract Table (23 Features)

| Index | Feature Name | Level | Exact TAWOS Definition | Taskove Source Field | Match Quality | Definition Mismatch / Nuance |
| :---: | :--- | :---: | :--- | :--- | :---: | :--- |
| 1 | `f` | Sprint | Planned sprint elapsed fraction: $(t - \text{Start}) / (\text{End} - \text{Start})$. | `(now - SprintSnapshot.start_date) / (SprintSnapshot.end_date - SprintSnapshot.start_date)` | **exact** | Both represent continuous time elapsed between $0.0$ and $1.0$. |
| 2 | `sprint_open_tasks` | Sprint | Count of tasks in sprint membership at $t$ whose status $\ne$ `done`. | `len([t for t in sprint_tasks if t.status != 'done'])` | **exact** | Identical aggregation over active sprint tasks. |
| 3 | `sprint_open_points` | Sprint | Sum of story points for open sprint tasks at $t$. | `sum([t.story_points or 0.0 for t in sprint_tasks if t.status != 'done'])` | **exact** | Unestimated tasks contribute 0.0 points in both. |
| 4 | `fraction_points_done` | Sprint | Completed points at $t$ / total sprint points. | `done_points / (done_points + open_points)` | **exact** | Burn-down completion fraction. |
| 5 | `assignee_open_points` | Team | Sum of open story points for assignee in this sprint at $t$. | Aggregation over `TaskSnapshot`s with matching `assignee_id` and `sprint_id`. | **approximate** | In Taskove, `TaskSnapshot.story_points` defaults to 1.0 if not specified, whereas TAWOS unestimated defaults to 0.0. In Taskove, explicitly treat None as 0.0. |
| 6 | `assignee_open_tasks` | Team | Count of open tasks for assignee in this sprint at $t$. | `len([t for t in sprint_tasks if t.assignee_id == current.assignee_id and t.status != 'done'])` | **exact** | Assignee work-in-progress (WIP) task count. |
| 7 | `days_in_current_status` | Task | Elapsed days since the task's most recent status changelog transition prior to $t$. | `TaskSnapshot.stalled_days` or `(now - TaskSnapshot.updated_at).days` | **approximate** | `stalled_days` is an integer count. If `updated_at` is updated by comments or description edits without a status move, it prematurely resets elapsed status dwell time. |
| 8 | `days_since_last_event` | Task | Elapsed days since any changelog edit on this issue prior to $t$. | `(now - TaskSnapshot.updated_at).days` | **approximate** | Taskove has no discrete changelog table; `updated_at` represents the latest modification timestamp. |
| 9 | `story_points` | Task | Jira story points scalar at $t$ (0.0 if unestimated). | `TaskSnapshot.story_points or 0.0` | **exact** | Agile effort estimation points. |
| 10 | `has_sp` | Task | Binary indicator: 1.0 if `story_points > 0`, else 0.0. | `1.0 if TaskSnapshot.story_points and TaskSnapshot.story_points > 0 else 0.0` | **exact** | Distinguishes explicitly estimated tasks from unestimated tasks. |
| 11 | `added_after_sprint_start`| Task | Binary indicator: 1.0 if task was added to sprint after kickoff. | `1.0 if TaskSnapshot.created_at > SprintSnapshot.start_date else 0.0` | **approximate** | **Critical Mismatch:** `created_at` reflects task creation, not sprint assignment. An existing backlog task added mid-sprint would be misclassified as 0.0 unless sprint addition audit timestamps are captured. |
| 12 | `assignee_present` | Task | Binary indicator: 1.0 if task has an assignee at $t$, else 0.0. | `1.0 if TaskSnapshot.assignee_id is not None else 0.0` | **exact** | Captures unassigned task neglect risk. |
| 13 | `links_depends_on` | Task | Count of outbound blocking dependencies (tasks this issue depends on). | `len(TaskSnapshot.dependencies)` | **approximate** | In `TaskSnapshot`, `dependencies: List[str]` lists prerequisite task IDs. Callers must ensure this list tracks tasks that *block this task*, not reverse links. |
| 14 | `status_category_is_todo` | Task | Binary indicator: 1.0 if status category is `todo`, 0.0 if `in_progress`. | `1.0 if TaskSnapshot.status == 'todo' else 0.0` | **exact** | Tasks in `in_progress` have actively started. |
| 15 | `type_Bug` | Task | One-hot binary: 1.0 if Type == 'Bug'. | `1.0 if TaskSnapshot.task_type == 'bug' else 0.0` | **approximate** | Taskove types require lowercase normalization (`'bug'`). |
| 16 | `type_Story` | Task | One-hot binary: 1.0 if Type == 'Story'. | `1.0 if TaskSnapshot.task_type in ['story', 'feature'] else 0.0` | **approximate** | Taskove uses `'feature'`; maps to Jira `'Story'`. |
| 17 | `type_Improvement` | Task | One-hot binary: 1.0 if Type == 'Improvement'. | `1.0 if TaskSnapshot.task_type == 'improvement' else 0.0` | **approximate** | Typically absent in standard Taskove enum; defaults to 0.0. |
| 18 | `type_Task` | Task | One-hot binary: 1.0 if Type == 'Task'. | `1.0 if TaskSnapshot.task_type in ['task', 'chore', 'refactor'] else 0.0` | **approximate** | Taskove `'chore'`/`'refactor'` map to generic `'Task'`. |
| 19 | `type_New Feature` | Task | One-hot binary: 1.0 if Type == 'New Feature'. | `1.0 if TaskSnapshot.task_type == 'new_feature' else 0.0` | **approximate** | Often consolidated into `'feature'`. |
| 20 | `type_Suggestion` | Task | One-hot binary: 1.0 if Type == 'Suggestion'. | `0.0` | **unavailable** | User feedback suggestions; defaults to 0.0 in Taskove. |
| 21 | `type_Enhancement Request`| Task | One-hot binary: 1.0 if Type == 'Enhancement Request'. | `0.0` | **unavailable** | Jira-specific enhancement tag; defaults to 0.0 in Taskove. |
| 22 | `type_Epic` | Task | One-hot binary: 1.0 if Type == 'Epic'. | `1.0 if TaskSnapshot.task_type == 'epic' else 0.0` | **approximate** | Epic/Initiative identifier in Taskove. |
| 23 | `type_other` | Task | One-hot binary: 1.0 if Type not in top 8. | `1.0 if not matched above else 0.0` | **exact** | Fallback category for unrecognized types. |

---

## 3. Critical Production Warnings & Mismatches

1. **Sprint Addition Timestamp:** In `TaskSnapshot`, `created_at` represents when the task record was created, not when it was committed to the active sprint. To eliminate the `added_after_sprint_start` approximation, `TaskSnapshot` should track `sprint_assigned_at: Optional[datetime] = None`.
2. **Stalled Days vs Event Log:** `TaskSnapshot.stalled_days` is an integer counter updated daily. The model expects floating continuous days `(now - last_status_change).total_seconds() / 86400.0`.
3. **Dependency Directionality:** `TaskSnapshot.dependencies` must represent prerequisite blockers (tasks that must finish before this task can start). If callers populate reverse blockers (tasks waiting on this task), the risk feature will distort.

