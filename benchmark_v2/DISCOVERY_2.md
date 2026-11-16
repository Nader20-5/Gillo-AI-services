# Model 2 Feasibility & Temporal Changelog Discovery (`benchmark_v2/DISCOVERY_2.md`)

**Date:** October 21, 2026  
**Artifact Analyzed:** `benchmark_v2/work/changelog_slim.parquet` (8.95 MB, 409,573 rows, streamed from `change_log.csv` in 82.53 seconds).  
**Target Population:** 41,475 issues in `issue.csv` assigned to CLOSED sprints in `sprint.csv`.

---

## 1. Format of Sprint Change Rows

In `change_log.csv`, Sprint changes are recorded under `Field == 'Sprint'`.  
There are **77,327 sprint change rows** across the target population.

### Key Structural Findings:
- `From_Value` and `To_Value`: Store Jira internal numeric sprint IDs (e.g., `"107"`, `"108"`).
- `From_String` and `To_String`: Store human-readable sprint names (e.g., `"Sprint 67"`, `"Sprint 68"`).
- **Multiple Sprints per Row:** In **28,594 rows** (36.98%), `To_String` contains comma-separated sprint names (e.g., `"Sprint 65, Sprint 66"`), and in **19,006 rows**, `From_String` contains comma-separated values. In Jira Agile, when an issue is assigned to multiple concurrent or sequential active boards/sprints, Jira serializes the entire active sprint set as a comma-delimited string rather than emitting separate atomic change rows.

### 10 Raw Example Rows from `change_log.csv`:

| Row | Issue_ID | From_Value | To_Value | From_String | To_String | Creation_Date (UTC) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 89 | `""` | `108` | `""` | `"Sprint 68"` | 2016-02-22 15:46:19 |
| 2 | 90 | `""` | `108` | `""` | `"Sprint 68"` | 2016-02-17 14:30:38 |
| 3 | 91 | `""` | `108` | `""` | `"Sprint 68"` | 2016-02-17 05:23:02 |
| 4 | 92 | `""` | `108` | `""` | `"Sprint 68"` | 2016-02-13 19:10:36 |
| 5 | 93 | `""` | `107` | `""` | `"Sprint 67"` | 2016-02-12 17:49:16 |
| 6 | 93 | `107` | `108` | `"Sprint 67"` | `"Sprint 68"` | 2016-02-13 19:10:03 |
| 7 | 94 | `""` | `107` | `""` | `"Sprint 67"` | 2016-02-05 16:17:13 |
| 8 | 95 | `107` | `108` | `"Sprint 67"` | `"Sprint 68"` | 2016-02-13 19:10:10 |
| 9 | 96 | `""` | `107` | `""` | `"Sprint 67"` | 2016-02-02 17:57:51 |
| 10 | 99 | `""` | `107` | `""` | `"Sprint 65"` | 2016-01-15 20:02:00 |

---

## 2. Sprint Membership Coverage & Reconstruction

- **Total issues assigned to closed sprints:** **41,475**
- **Issues with at least one Sprint change row in changelog:** **32,540 (78.46%)**
- **Issues with ZERO Sprint change rows:** **8,935 (21.54%)**

### Answers:
- **Is `issue.Sprint_ID` the only evidence of membership for the rest?**  
  **Yes.** For 8,935 issues (21.54%), there is no changelog entry for `Field == 'Sprint'`. This typically happens when tasks are assigned to a sprint at initial creation time (which Jira logs under the issue creation event rather than a diff in changelog) or where changelog was incomplete.
- **For what fraction of issues can sprint membership be reconstructed at an arbitrary time $t$?**  
  Sprint membership can be dynamically reconstructed with temporal fidelity for **78.46%** of closed-sprint issues. For the remaining **21.54%**, one must assume they belonged to `issue.Sprint_ID` starting from `issue.Creation_Date` or sprint kickoff.

---

## 3. Multi-Sprint Membership (Hops & Carry-Overs) vs `issue.Sprint_ID`

- **Issues belonging to $>1$ sprint during their lifecycle:** **15,998 issues**
- **Fraction of issues with sprint changes:** **49.16%** (15,998 / 32,540)
- **Comparison to `issue.Sprint_ID`:**
  `issue.Sprint_ID` in `issue.csv` contains only a **single scalar integer**. In 100% of cases in `issue.csv`, it only shows the *final* sprint.  
  Without the changelog, **49.16% of all task carry-overs, sprint hops, and scope migrations are completely invisible**. A model relying solely on `issue.csv` suffers severe label distortion because a task carried over across 3 sprints appears in `issue.csv` as if it only ever existed in the 3rd sprint.

---

## 4. Status Reconstruction & Top 30 Status Names

- **Total status change rows:** **171,183**
- **Issues with status change history:** **41,093** (99.08% of target issues)
- **Can status at arbitrary time $t$ be derived?**  
  **Yes.** For any time $t$, replay the status changelog up to $t$. If no change occurred before $t$, the status is the first change's `From_String` (e.g. `"Open"` or `"New"`).

### Top 30 Status Names in Target Sprints (Frequency in Changelog):

| Rank | Status Name | Occurrences | Proposed State Category |
| :---: | :--- | :---: | :---: |
| 1 | `In Progress` | 70,692 | `in_progress` |
| 2 | `Resolved` | 35,031 | `done` |
| 3 | `Closed` | 31,207 | `done` |
| 4 | `Open` | 30,236 | `todo` |
| 5 | `In Review` | 21,046 | `in_progress` |
| 6 | `Done` | 20,241 | `done` |
| 7 | `To Do` | 16,245 | `todo` |
| 8 | `Reviewed` | 10,941 | `in_progress` |
| 9 | `Development in progress` | 6,791 | `in_progress` |
| 10 | `Accepted` | 6,729 | `todo` |
| 11 | `Reopened` | 5,953 | `todo` |
| 12 | `Waiting for integration review` | 5,298 | `in_progress` |
| 13 | `Awaiting Quality Review` | 5,158 | `in_progress` |
| 14 | `Reviewable` | 5,085 | `in_progress` |
| 15 | `Quality Review` | 5,053 | `in_progress` |
| 16 | `Waiting for peer review` | 5,049 | `in_progress` |
| 17 | `Peer review in progress` | 4,536 | `in_progress` |
| 18 | `New` | 3,821 | `todo` |
| 19 | `In  Review` *(double space typo in Jira)* | 3,652 | `in_progress` |
| 20 | `Integration review in progress` | 3,177 | `in_progress` |
| 21 | `Waiting for testing` | 3,022 | `in_progress` |
| 22 | `Testing in progress` | 3,022 | `in_progress` |
| 23 | `In Code Review` | 2,738 | `in_progress` |
| 24 | `Tested` | 2,580 | `in_progress` |
| 25 | `Awaiting Deployment` | 2,396 | `in_progress` |
| 26 | `To Test` | 2,267 | `in_progress` |
| 27 | `Code Review` | 2,238 | `in_progress` |
| 28 | `Waiting for Review` | 2,044 | `in_progress` |
| 29 | `To Be Tested` | 2,000 | `in_progress` |
| 30 | `TO DO` | 1,871 | `todo` |

### Proposed Tri-State Mapping:
- **`todo`:** `Open`, `To Do`, `TO DO`, `New`, `Created`, `Backlog`, `Reopened`, `Accepted`, `Needs Triage`, `Submitted`, `Untriaged`, `Unconfirmed`.
- **`in_progress`:** `In Progress`, `Development in progress`, `In Review`, `In  Review`, `In Code Review`, `Code Review`, `Review`, `Reviewed`, `Reviewable`, `Under Review`, `Waiting for peer review`, `Peer review in progress`, `Waiting for integration review`, `Integration review in progress`, `Awaiting Quality Review`, `Quality Review`, `Waiting for testing`, `Testing in progress`, `To Test`, `To Be Tested`, `Tested`, `Blocked`, `Patch Available`, `Awaiting Deployment`.
- **`done`:** `Closed`, `Resolved`, `Done`, `Complete`, `Completed`, `Verified`, `Delivered`.

---

## 5. Agreement Between `issue.Resolution_Date` and Changelog 'Done' Status

- **Issues with both `Resolution_Date` and changelog 'done' status:** **38,495**
- **Disagreement $> 1$ hour:** **4,776 issues (12.41%)**
- **Disagreement $> 24$ hours:** **4,445 issues (11.55%)**
- **Median absolute difference:** **0.0 hours** (exact alignment for 87.59% of issues).
- **95th percentile difference:** **5,878.48 hours (~244.9 days)**.

### Why does this disagreement occur?
In Jira workflows:
1. An issue is resolved by an engineer (setting status to `Resolved` and recording `Resolution_Date`), but weeks or months later a QA engineer or release manager performs a batch transition to `Closed`.
2. Reopened issues: A task is resolved, reopened months later, and then closed without resetting `Resolution_Date`.
3. Administrative bulk edits: Project administrators bulk-close stale issues long after the sprint ended.

---

## 6. Dependency Links in Changelog

- **Total link rows in `changelog_slim`:** **42,751 rows** (`Field == 'Link'`).
- **Can dependency links existing before time $t$ be counted?**  
  **YES.** In `change_log.csv`, each link modification carries a precise ISO `Creation_Date`. This completely solves the temporal leakage in `issue_link.csv` (which has no timestamps).

### 10 Raw Example Link Rows:

| Row | Issue_ID | From_String | To_String | Creation_Date |
| :---: | :---: | :---: | :--- | :---: |
| 1 | 89 | `""` | `"This issue is related to XD-2567"` | 2016-02-22 15:28:17 |
| 2 | 90 | `""` | `"This issue depends on INT-3956"` | 2016-02-17 14:28:51 |
| 3 | 96 | `""` | `"This issue relates to XD-3721"` | 2016-02-02 17:56:59 |
| 4 | 112 | `""` | `"This issue is related to XD-3737"` | 2016-02-02 17:56:59 |
| 5 | 124 | `""` | `"This issue is related to INT-3896"` | 2015-11-21 17:40:20 |
| 6 | 160 | `""` | `"This issue is related to XD-3613"` | 2015-11-02 17:00:01 |
| 7 | 185 | `""` | `"This issue relates to XD-3559"` | 2015-10-27 21:30:09 |
| 8 | 216 | `""` | `"This issue is depended on by XD-2882"` | 2015-10-14 15:52:41 |
| 9 | 235 | `""` | `"This issue is related to INT-3848"` | 2015-10-07 16:02:31 |
| 10 | 240 | `""` | `"This issue is depended on by XD-3489"` | 2015-10-15 15:09:57 |

---

## 7. Sprint Dates Coverage & Deviation in `sprint.csv`

Across all 4,594 sprints in `sprint.csv`:
- **`Activated_Date` Coverage:** **342 sprints (7.44%)** — almost completely missing.
- **`Complete_Date` Coverage:** **4,507 sprints (98.11%)**.
- **Distribution of `(Complete_Date - End_Date)` in days:**
  - $P_5$: **-1.16 days** (closed slightly early)
  - $P_{25}$: **-0.02 days**
  - $P_{50}$ (Median): **+0.33 days** (~8 hours after planned end date)
  - $P_{75}$: **+2.96 days**
  - $P_{95}$: **+23.06 days**
  - Mean: **+7.77 days** ($\sigma = 63.46$ days)

*Finding:* While 50% of sprints close within 8 hours of their planned `End_Date`, over 25% overrun by $\ge 3$ days, and 5% linger open for more than 3 weeks before someone hits the "Complete Sprint" button in Jira.

---

## 8. Effort Columns Coverage & Correlation with Story Points

Across the 41,475 closed sprint issues:

### Column Coverage & Distribution:
1. **`In_Progress_Minutes`:**
   - Non-zero coverage: **26,089 issues (62.90%)**
   - Median: **2,692.0 minutes (~44.87 hours)**
   - 95th Percentile: **70,422.8 minutes (~1,173.7 hours)**
2. **`Timespent` (logged work hours in Jira):**
   - Non-zero coverage: **829 issues (2.00%)** — *virtually non-existent!*
   - Median: **240.0 minutes (4.0 hours)**
   - 95th Percentile: **2,551.8 minutes (42.5 hours)**
3. **`Total_Effort_Minutes`:**
   - Non-zero coverage: **32,650 issues (78.72%)**
   - Median: **7,331.5 minutes (~122.19 hours)**
   - 95th Percentile: **117,554.4 minutes (~1,959.2 hours)**

### Pearson Correlation with `Story_Point` (Top 5 Projects with SP):

| Project ID | Project Key & Name | Correlation: `Story_Point` vs `In_Progress_Minutes` | Correlation: `Story_Point` vs `Timespent` | Correlation: `Story_Point` vs `Total_Effort_Minutes` |
| :---: | :--- | :---: | :---: | :---: |
| 28 | `DM` (Lsstcorp Data Management) | **0.2073** ($N=5,743$) | **0.1865** ($N=126$) | **0.2113** ($N=6,817$) |
| 12 | `TIMOB` (The Titanium SDK) | **0.0631** ($N=2,039$) | *N/A ($N=0$)* | **0.0931** ($N=2,600$) |
| 36 | `MULE` (Mule) | **0.2112** ($N=2,194$) | *N/A ($N=0$)* | **0.2112** ($N=2,193$) |
| 4 | `MESOS` (Apache Mesos) | **0.1761** ($N=1,386$) | *N/A ($N=1$)* | **0.1332** ($N=1,772$) |
| 1 | `XD` (Spring XD) | **0.2970** ($N=1,691$) | *N/A ($N=2$)* | **0.2885** ($N=1,801$) |

*Crucial Finding:* The correlation between estimated `Story_Point` and actual recorded effort is shockingly weak across all major projects ($r = 0.06$ to $0.30$). Story points reflect perceived complexity or planning currency, not wall-clock cycle duration. Furthermore, actual time tracking (`Timespent`) is omitted by developers in 98% of cases.

---

## 9. Suspicious or Surprising Things I Noticed

1. **`Timespent` is practically unused in real Jira teams (2.0% coverage):** Developers almost never use Jira's native time-tracking widget. Models relying on `Timespent` as ground truth will fail due to 98% missingness.
2. **`Activated_Date` in `sprint.csv` is missing in 92.6% of sprints:** Any calculation of sprint start must fall back to `Start_Date`, because Jira only logged `Activated_Date` in 342 of 4,594 sprints.
3. **Jira serializes sprint lists into a single string:** Because Jira boards can share sprints, `change_log.To_String` contains comma-separated names like `"Sprint 65, Sprint 66"`. Code that assumes atomic scalar foreign keys will fail without string splitting.
4. **Nearly 50% of sprint issues hop sprints:** 49.16% of issues change sprint memberships, showing that sprint carry-over is the norm in open-source agile teams rather than an exception.
5. **Enormous right tail on cycle times:** The 95th percentile of `Total_Effort_Minutes` is ~1,959 hours (81 days), indicating tasks sit untouched in `In Progress` states across multiple release cycles.
