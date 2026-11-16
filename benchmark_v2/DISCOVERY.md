# TAWOS Dataset Discovery & Schema Inspection (`tawos_csv`)

**Date:** October 21, 2026  
**Target Directory:** `d:\Gillo AI Service\tawos_csv`  
**Purpose:** Empirical inventory and ground-truth audit of raw Jira agile data for `benchmark_v2` pipeline planning.

---

## 1. Catalog of All CSV Files (Columns and Row Counts)

| File Name | File Size (MB) | Row Count (Excl. Header) | Column Count | Columns |
| :--- | :--- | :--- | :--- | :--- |
| `affected_version.csv` | 3.21 MB | 264,004 | 2 | `Issue_ID`, `Affected_Version_ID` |
| `change_log.csv` | 1,649.53 MB | 21,116,864 | 10 | `ID`, `Field`, `From_Value`, `To_Value`, `From_String`, `To_String`, `Change_Type`, `Creation_Date`, `Author_ID`, `Issue_ID` |
| `comment.csv` | 1,253.18 MB | 8,273,653 | 7 | `ID`, `Comment`, `Comment_Text`, `Comment_Code`, `Creation_Date`, `Author_ID`, `Issue_ID` |
| `component.csv` | 0.08 MB | 2,001 | 5 | `ID`, `Jira_ID`, `Name`, `Description`, `Project_ID` |
| `fix_version.csv` | 2.71 MB | 224,254 | 2 | `Issue_ID`, `Fix_Version_ID` |
| `issue.csv` | 967.82 MB | 458,232 | 30 | `ID`, `Jira_ID`, `Issue_Key`, `URL`, `Title`, `Description`, `Description_Text`, `Description_Code`, `Type`, `Priority`, `Status`, `Resolution`, `Creation_Date`, `Estimation_Date`, `Resolution_Date`, `Last_Updated`, `Story_Point`, `Timespent`, `In_Progress_Minutes`, `Total_Effort_Minutes`, `Resolution_Time_Minutes`, `Title_Changed_After_Estimation`, `Description_Changed_After_Estimation`, `Story_Point_Changed_After_Estimation`, `Pull_Request_URL`, `Creator_ID`, `Reporter_ID`, `Assignee_ID`, `Project_ID`, `Sprint_ID` |
| `issue_component.csv` | 4.36 MB | 366,922 | 2 | `Issue_ID`, `Component_ID` |
| `issue_link.csv` | 12.77 MB | 246,587 | 6 | `ID`, `Issue_ID`, `Name`, `Description`, `Direction`, `Target_Issue_ID` |
| `project.csv` | 0.01 MB | 40 | 9 | `ID`, `Project_Key`, `Name`, `URL`, `Description`, `Start_Date`, `Last_Update_Date`, `SP_Field_ID`, `Repository_ID` |
| `repository.csv` | 0.00 MB | 12 | 4 | `ID`, `Name`, `Description`, `URL` |
| `sprint.csv` | 0.43 MB | 4,594 | 9 | `ID`, `JiraID`, `Name`, `State`, `Start_Date`, `End_Date`, `Activated_Date`, `Complete_Date`, `Project_ID` |
| `user.csv` | 2.05 MB | 206,162 | 2 | `ID`, `Project_ID` |
| `version.csv` | 0.36 MB | 6,911 | 8 | `ID`, `Jira_ID`, `Name`, `Description`, `Archived`, `Released`, `Release_Date`, `Project_ID` |

---

## 2. Description Column in `issue.csv`

**Yes**, `issue.csv` contains a `Description` column.  
In fact, the dataset provides **three distinct description-related columns**:
1. `Description`: Full raw HTML/markup description as entered by the user in Jira.
2. `Description_Text`: Clean, stripped textual narrative extracted from the description (suitable for NLP/vectorization).
3. `Description_Code`: Code blocks, stack traces, and snippets extracted separately from the description.

---

## 3. Issues in CLOSED Sprints with Non-Null `Story_Point`

### Sprint States Distribution (`sprint.csv`):
- `CLOSED`: **4,507** sprints (98.1%)
- `FUTURE`: **51** sprints (1.1%)
- `ACTIVE`: **36** sprints (0.8%)
- **Total Sprints:** 4,594

### Issue Counts (`issue.csv`):
- **Total issues across all 40 projects:** 458,232
- **Issues with `Story_Point > 0` (non-null/positive):** 63,011
- **Issues assigned to a `CLOSED` sprint:** 41,475
- **Issues in a `CLOSED` sprint WITH non-null `Story_Point`:** **28,914**

### Top 15 Projects by Issues in Closed Sprints with Story Points:

| Rank | Project ID | Project Key | Project Name | Issues in Closed Sprints with SP |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 28 | `DM` | Lsstcorp Data management | **7,774** |
| 2 | 12 | `TIMOB` | The Titanium SDK | **3,279** |
| 3 | 36 | `MULE` | Mule | **2,850** |
| 4 | 4 | `MESOS` | Apache Mesos | **2,436** |
| 5 | 1 | `XD` | Spring XD | **2,114** |
| 6 | 13 | `TISTUD` | Appcelerator Studio | **1,274** |
| 7 | 34 | `MDL` | Moodle | **1,179** |
| 8 | 11 | `APPC` | Titanium Mobile Platform | **973** |
| 9 | 27 | `STL` | Hyperledger Sawtooth | **802** |
| 10 | 3 | `NEXUS` | Sonatype Nexus | **781** |
| 11 | 26 | `IS` | Hyperledger Indy SDK | **588** |
| 12 | 25 | `INDY` | Hyperledger Indy Node | **509** |
| 13 | 35 | `APIKIT` | Mule APIkit | **358** |
| 14 | 30 | `COMPASS` | MongoDB Compass | **357** |
| 15 | 23 | `BE` | Hyperledger Blockchain Explorer | **326** |

*(These top 15 projects account for 25,600 of the 28,914 issues, or 88.5% of all estimated agile sprint issues).*

---

## 4. History / Change Log Table (`change_log.csv`)

**Yes**, a change log table exists: **`change_log.csv`**.

- **Row Count:** **21,116,864 rows** (1.65 GB).
- **Columns (10):**
  - `ID`: Unique changelog entry ID.
  - `Field`: Name of the modified field.
  - `From_Value`: Raw previous value (IDs or codes).
  - `To_Value`: Raw new value.
  - `From_String`: Human-readable previous string.
  - `To_String`: Human-readable new string.
  - `Change_Type`: Type of change (`jira`, etc.).
  - `Creation_Date`: **The change timestamp column** (ISO-8601 UTC string).
  - `Author_ID`: User ID who made the change.
  - `Issue_ID`: Foreign key linking to `issue.ID`.

### Specific Field Tracking:
- **`status`:** **Yes**, tracked as `status` (167,240 occurrences in 1M sample). Also records `Epic Status`, `Test Status`, and `QA Kickoff Status`.
- **`Story Points`:** **Yes**, tracked as `Story Points`, `Story points`, and `Actual Story Points` (8,955+ occurrences in sample).
- **`Sprint`:** **Yes**, tracked as `Sprint` (25,511 occurrences in sample). Captures whenever an issue is added to, removed from, or moved between sprints.
- **`assignee`:** **Yes**, tracked as `assignee` (47,264 occurrences in sample).
- **Other Tracked Fields:** `resolution`, `Fix Version`, `description`, `timespent`, `timeestimate`, `timeoriginalestimate`, `summary`, `Component`, `labels`, `issuetype`, `Rank`, and `Link`.

---

## 5. Creation Timestamp in `issue_link.csv`

**NO, `issue_link.csv` does NOT have a creation timestamp.**

- **Columns in `issue_link.csv`:**
  `['ID', 'Issue_ID', 'Name', 'Description', 'Direction', 'Target_Issue_ID']`
- **Implication for Point-in-Time Temporal Reconstruction:**
  You **cannot** know when an issue dependency or duplicate link was created by inspecting `issue_link.csv` alone. Taking links directly from `issue_link.csv` introduces look-ahead / temporal leakage because dependencies linked *late* in a sprint will appear as if they existed at sprint kickoff.
- **Remedy:** To determine whether a link existed at time $T$ (e.g. sprint start), one must look at `change_log.csv` where `Field == 'Link'` or `Field == 'RemoteIssueLink'`, using its `Creation_Date` timestamp.

---

## 6. Assignee / Developer Fields & Plain Warnings on Assumed Non-Existent Fields

### Assignee Fields Present:
1. `issue.csv`:
   - `Assignee_ID`: Foreign key pointing to an assignee.
   - `Creator_ID`: Foreign key pointing to the issue creator.
   - `Reporter_ID`: Foreign key pointing to the issue reporter.
2. `change_log.csv`:
   - `Author_ID`: Foreign key pointing to the user who triggered the change.
   - `Field == 'assignee'`: Tracks reassignments from `From_String` to `To_String`.

### Plain Warnings: What You Might Assume Exists, But DOES NOT:

1. **`user.csv` contains NO user metadata whatsoever:**
   - The columns of `user.csv` are strictly: `['ID', 'Project_ID']`.
   - **There are NO names, NO emails, NO roles (e.g. Developer vs QA vs TL vs PM), NO seniority levels, and NO skills/tags.**
   - All users in TAWOS are opaque anonymized integer IDs. Developer experience or velocity cannot be fetched from user metadata tables; it can only be inferred retrospectively by aggregating past task completion logs per `Assignee_ID`.

2. **`issue.csv` does NOT have a sprint assignment history column:**
   - `Sprint_ID` in `issue.csv` stores only the **final/current sprint** the issue was in when the dump was exported.
   - If a task was started in Sprint 1 and carried over to Sprint 2, `issue.Sprint_ID` shows only Sprint 2.
   - Sprint movement history must be reconstructed exclusively through `change_log.csv` (`Field == 'Sprint'`).

3. **`sprint.csv` does NOT contain task lists:**
   - `sprint.csv` contains only sprint metadata (`ID`, `Name`, `State`, `Start_Date`, `End_Date`, `Activated_Date`, `Complete_Date`, `Project_ID`).
   - Sprints do not store an array of issue IDs; relationships are formed foreign-key style from `issue` or `change_log`.

4. **`repository.csv` has NO commit-level links to tasks:**
   - `repository.csv` contains only 12 rows describing high-level git URLs for 12 out of the 40 projects.
   - `issue.Pull_Request_URL` exists on `issue.csv`, but individual commits, changed LOC, or diff sizes are not stored in TAWOS tabular CSVs.
