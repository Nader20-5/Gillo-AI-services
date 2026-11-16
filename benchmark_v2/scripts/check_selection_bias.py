"""
benchmark_v2/scripts/check_selection_bias.py

Checks selection bias in Model 2:
1. Identifies 3,331 eligible closed sprints and builds (Project_ID, JiraID) -> Sprint ID map.
2. Streams tawos_csv/change_log.csv in 500k-row chunks.
3. Finds all issues appearing in any Sprint change row (From_Value or To_Value) mapped to an eligible sprint.
4. Identifies issues NOT in the current dataset (issue.Sprint_ID not in closed sprints).
5. Reports counts, final issue.Status, final Resolution distribution, and sprint outcomes.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
work_dir = os.path.join(BASE_DIR, "work")
tawos_dir = r"d:\Gillo AI Service\tawos_csv"
sys.path.insert(0, BASE_DIR)

print("=== STEP 1: Identifying Eligible Sprints and Sprint Map ===")
t0 = time.time()
sprint_df = pd.read_csv(os.path.join(tawos_dir, "sprint.csv"))
sprint_df['Start_dt'] = pd.to_datetime(sprint_df['Start_Date'], errors='coerce')
sprint_df['End_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')
sprint_df['Complete_dt'] = pd.to_datetime(sprint_df['Complete_Date'], errors='coerce')

duration_days = (sprint_df['End_dt'] - sprint_df['Start_dt']).dt.total_seconds() / (24*3600)
diff_complete_end = (sprint_df['Complete_dt'] - sprint_df['End_dt']).dt.total_seconds() / (24*3600)

closed_mask = sprint_df['State'] == 'CLOSED'
duration_mask = (duration_days >= 5.0) & (duration_days <= 35.0)
complete_mask = (diff_complete_end <= 3.0) & diff_complete_end.notna()

eligible_sprints = sprint_df[closed_mask & duration_mask & complete_mask].copy()
eligible_sprint_ids = set(eligible_sprints['ID'].astype(int).tolist())
print(f"Total eligible closed sprints: {len(eligible_sprint_ids)}")

sprint_map = {(int(r.Project_ID), int(r.JiraID)): int(r.ID) for r in sprint_df.itertuples() if pd.notna(r.Project_ID) and pd.notna(r.JiraID)}

print("=== STEP 2: Loading issue.csv Metadata ===")
issues_all = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=[
    'ID', 'Sprint_ID', 'Project_ID', 'Status', 'Resolution', 'Creation_Date'
], low_memory=False)
issues_all.columns = [c.lstrip('\ufeff') for c in issues_all.columns]
print(f"Total issues in issue.csv: {len(issues_all):,}")

issue_project_map = dict(zip(issues_all['ID'].astype(int), issues_all['Project_ID'].dropna().astype(int)))
issue_sprint_map = dict(zip(issues_all['ID'].astype(int), issues_all['Sprint_ID']))

# Current dataset issue IDs
df_curr = pd.read_parquet(os.path.join(work_dir, "model2_dataset.parquet"))
current_dataset_issue_ids = set(df_curr['issue_id'].unique())
print(f"Issues currently in model2_dataset: {len(current_dataset_issue_ids):,}")

print("=== STEP 3: Streaming change_log.csv (500k chunks) for Sprint events ===")
cl_path = os.path.join(tawos_dir, "change_log.csv")
cl_iter = pd.read_csv(cl_path, chunksize=500000, low_memory=False)

sprint_field_names = {'sprint', 'Sprint'}

# Map of issue_id -> set of eligible sprint_ids it belonged to
issue_eligible_sprints = {}
total_cl_rows = 0
sprint_event_rows = 0

for idx, chunk in enumerate(cl_iter):
    total_cl_rows += len(chunk)
    chunk.columns = [c.lstrip('\ufeff') for c in chunk.columns]
    sp_chunk = chunk[chunk['Field'].isin(sprint_field_names)]
    sprint_event_rows += len(sp_chunk)

    for r in sp_chunk.itertuples():
        iid = r.Issue_ID
        if pd.isna(iid):
            continue
        iid = int(iid)
        pid = issue_project_map.get(iid)
        if pid is None:
            continue

        for val in [r.From_Value, r.To_Value]:
            if pd.notna(val) and str(val).strip():
                for part in str(val).split(','):
                    part = part.strip()
                    if part.isdigit():
                        jid = int(part)
                        sid = sprint_map.get((pid, jid))
                        if sid is not None and sid in eligible_sprint_ids:
                            issue_eligible_sprints.setdefault(iid, set()).add(sid)

    if (idx + 1) % 10 == 0:
        print(f"  Scanned {total_cl_rows:,} rows ({time.time()-t0:.1f}s) -> Found {len(issue_eligible_sprints):,} issues in eligible sprints...")

print(f"\nFinished streaming changelog: {total_cl_rows:,} total rows scanned.")
print(f"Total Sprint changelog rows: {sprint_event_rows:,}")
print(f"Total issues with changelog events in eligible sprints: {len(issue_eligible_sprints):,}")

# Also add issues whose initial/final Sprint_ID in issue.csv is in eligible sprints
for iid, sid_val in issue_sprint_map.items():
    if pd.notna(sid_val):
        try:
            sid_int = int(float(sid_val))
            if sid_int in eligible_sprint_ids:
                issue_eligible_sprints.setdefault(iid, set()).add(sid_int)
        except (ValueError, TypeError):
            pass

full_universe_issue_ids = set(issue_eligible_sprints.keys())
print(f"Total issues in FULL UNIVERSE (changelog + issue.Sprint_ID in eligible sprints): {len(full_universe_issue_ids):,}")

missing_issue_ids = full_universe_issue_ids - current_dataset_issue_ids
print(f"Issues in full universe but NOT in current dataset: {len(missing_issue_ids):,}")

# Analyze missing issues
issues_missing_df = issues_all[issues_all['ID'].isin(missing_issue_ids)].copy()
print("\n--- Missing Issues Analysis ---")
print("Final Status distribution of missing issues:")
print(issues_missing_df['Status'].value_counts().head(10))

print("\nFinal Resolution distribution of missing issues:")
print(issues_missing_df['Resolution'].fillna('Unresolved').value_counts().head(10))

print("\nWhere did their final issue.Sprint_ID go?")
print(f"Null / Backlog (no final sprint): {issues_missing_df['Sprint_ID'].isna().sum():,} ({issues_missing_df['Sprint_ID'].isna().mean()*100:.2f}%)")
print(f"Has another Sprint_ID (not in closed sprints): {issues_missing_df['Sprint_ID'].notna().sum():,}")

# Save selection bias stats to json
selection_bias_stats = {
    'total_changelog_rows_scanned': total_cl_rows,
    'total_sprint_changelog_rows': sprint_event_rows,
    'current_dataset_issues': len(current_dataset_issue_ids),
    'full_universe_issues': len(full_universe_issue_ids),
    'missing_issues_count': len(missing_issue_ids),
    'missing_issues_pct_of_full': round(len(missing_issue_ids) / len(full_universe_issue_ids) * 100, 2),
    'missing_final_status_top5': issues_missing_df['Status'].value_counts().head(5).to_dict(),
    'missing_final_resolution_top5': issues_missing_df['Resolution'].fillna('Unresolved').value_counts().head(5).to_dict(),
    'missing_backlog_null_sprint_count': int(issues_missing_df['Sprint_ID'].isna().sum()),
    'missing_backlog_null_sprint_pct': round(float(issues_missing_df['Sprint_ID'].isna().mean() * 100), 2)
}

stats_path = os.path.join(work_dir, "selection_bias_check_metrics.json")
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(selection_bias_stats, f, indent=2)

print(f"\nSaved selection bias stats to {stats_path}")
print(f"Elapsed time: {time.time()-t0:.1f}s")

