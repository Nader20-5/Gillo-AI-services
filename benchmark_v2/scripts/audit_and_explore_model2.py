import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
work_dir = os.path.join(BASE_DIR, "work")
tawos_dir = r"d:\Gillo AI Service\tawos_csv"
sys.path.insert(0, BASE_DIR)

from replay import categorize_status, TODO_SET, DONE_SET, IN_PROGRESS_SET

print("=== AUDIT STEP 1: Status names explicit vs default fallback ===")
changelog = pd.read_parquet(os.path.join(work_dir, "changelog_slim.parquet"))
status_rows = changelog[changelog['Field'] == 'status']
print(f"Total status changelog rows: {len(status_rows):,}")

# Gather all unique status strings seen in From_String and To_String
all_statuses = set(status_rows['From_String'].dropna().unique()) | set(status_rows['To_String'].dropna().unique())
print(f"Total unique status strings in changelog: {len(all_statuses)}")

explicit_map = {}
fallback_map = {}

for s in all_statuses:
    clean = ' '.join(s.strip().lower().split())
    if clean in DONE_SET:
        explicit_map[s] = ('done', 'exact_set')
    elif clean in TODO_SET:
        explicit_map[s] = ('todo', 'exact_set')
    elif clean in IN_PROGRESS_SET:
        explicit_map[s] = ('in_progress', 'exact_set')
    elif any(w in clean for w in ['resolved', 'closed', 'done', 'complete', 'verified', 'shipped']):
        explicit_map[s] = ('done', 'substring')
    elif any(w in clean for w in ['to do', 'backlog', 'reopen', 'new', 'triage', 'deferred']):
        explicit_map[s] = ('todo', 'substring')
    elif any(w in clean for w in ['progress', 'review', 'testing', 'test', 'qa', 'accepted', 'active', 'pr', 'coding', 'deploy', 'investigat', 'working', 'merge']):
        explicit_map[s] = ('in_progress', 'substring')
    else:
        fallback_map[s] = 'in_progress (default fallback)'

print(f"Explicitly mapped: {len(explicit_map)}")
print(f"Fallback mapped (unmatched default): {len(fallback_map)}")
print("Fallback status names:", fallback_map)

# Top 10 resolution To_String values
print("\n=== AUDIT STEP 2: Top 10 Resolution To_String values ===")
res_rows = changelog[changelog['Field'].str.lower() == 'resolution']
print(f"Total resolution changelog rows: {len(res_rows):,}")
top_res = res_rows['To_String'].value_counts().head(15)
print(top_res)

# Issues with NO status rows in changelog
print("\n=== AUDIT STEP 3: Issues with NO status rows ===")
sprint_df = pd.read_csv(os.path.join(tawos_dir, "sprint.csv"))
closed_sprint_ids = set(sprint_df[sprint_df['State'] == 'CLOSED']['ID'])

issues_df = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=[
    'ID', 'Sprint_ID', 'Project_ID', 'Status', 'Creation_Date', 'Story_Point', 'Assignee_ID', 'Type', 'Priority', 'Resolution'
], low_memory=False)
issues_closed = issues_df[issues_df['Sprint_ID'].isin(closed_sprint_ids)].copy()

issues_with_status_events = set(status_rows['Issue_ID'].unique())
all_closed_issue_ids = set(issues_closed['ID'].unique())

no_status_event_issue_ids = all_closed_issue_ids - issues_with_status_events
print(f"Total closed-sprint issues: {len(all_closed_issue_ids):,}")
print(f"Issues with status changelog events: {len(all_closed_issue_ids & issues_with_status_events):,}")
print(f"Issues with NO status changelog events: {len(no_status_event_issue_ids):,} ({len(no_status_event_issue_ids)/len(all_closed_issue_ids)*100:.2f}%)")

# How many rows in model2_dataset.parquet belong to issues with no status events?
df_model2 = pd.read_parquet(os.path.join(work_dir, "model2_dataset.parquet"))
rows_no_status_events = df_model2[df_model2['issue_id'].isin(no_status_event_issue_ids)]
print(f"Total rows in model2_dataset: {len(df_model2):,}")
print(f"Rows in model2_dataset from issues with no status changelog events: {len(rows_no_status_events):,} ({len(rows_no_status_events)/len(df_model2)*100:.2f}%)")
print(f"In test split: {len(rows_no_status_events[rows_no_status_events['split']=='test']):,} / {len(df_model2[df_model2['split']=='test']):,}")


