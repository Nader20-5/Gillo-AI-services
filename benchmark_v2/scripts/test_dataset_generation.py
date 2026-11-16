import os
import sys
import time
import pandas as pd
import numpy as np

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
sys.path.insert(0, BASE_DIR)

from replay import ReplayEngine, categorize_status

tawos_dir = r"d:\Gillo AI Service\tawos_csv"
work_dir = os.path.join(BASE_DIR, "work")

t0 = time.time()
print("Loading changelog...")
changelog = pd.read_parquet(os.path.join(work_dir, "changelog_slim.parquet"))
print(f"Changelog loaded in {time.time()-t0:.2f}s")

t0 = time.time()
print("Loading issues...")
issues_df = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=[
    'ID', 'Sprint_ID', 'Project_ID', 'Status', 'Creation_Date', 'Story_Point', 'Assignee_ID', 'Type', 'Priority', 'Resolution'
], low_memory=False)
issues_df['Creation_Date'] = pd.to_datetime(issues_df['Creation_Date'], errors='coerce')
changelog['Creation_Date'] = pd.to_datetime(changelog['Creation_Date'], errors='coerce')
print(f"Issues loaded in {time.time()-t0:.2f}s")

sprint_df = pd.read_csv(os.path.join(tawos_dir, "sprint.csv"))
sprint_df['Start_dt'] = pd.to_datetime(sprint_df['Start_Date'], errors='coerce')
sprint_df['End_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')
sprint_df['Complete_dt'] = pd.to_datetime(sprint_df['Complete_Date'], errors='coerce')

sprint_map = {(r.Project_ID, r.JiraID): r.ID for r in sprint_df.itertuples()}

print("Grouping events by issue...")
t0 = time.time()
events_by_issue = {}
for iid, group in changelog.groupby('Issue_ID'):
    events_by_issue[int(iid)] = group.sort_values('Creation_Date').to_dict('records')
print(f"Events indexed in {time.time()-t0:.2f}s")

issues_dict = {}
for r in issues_df.itertuples():
    issues_dict[int(r.ID)] = {
        'id': int(r.ID),
        'project_id': int(r.Project_ID) if pd.notna(r.Project_ID) else None,
        'sprint_id': int(r.Sprint_ID) if pd.notna(r.Sprint_ID) else None,
        'status': r.Status,
        'creation_date': r.Creation_Date,
        'story_point': r.Story_Point,
        'assignee_id': r.Assignee_ID,
        'type': r.Type,
        'priority': r.Priority,
        'resolution': r.Resolution
    }

print("Building ReplayEngine...")
t0 = time.time()
engine = ReplayEngine(issues_dict, events_by_issue, sprint_map)
print(f"ReplayEngine initialized in {time.time()-t0:.2f}s")

# Test candidate issues per sprint mapping
candidate_issues_by_sprint = {}
for iid, iss in issues_dict.items():
    # Candidates from initial sprint or issue.Sprint_ID
    for sid in engine.initial_sprint_sets.get(iid, set()):
        candidate_issues_by_sprint.setdefault(sid, set()).add(iid)
    # Candidates from all Sprint changelog events
    evs = events_by_issue.get(iid, [])
    pid = iss.get('project_id')
    for e in evs:
        if e['Field'] == 'Sprint':
            for v in [e.get('From_Value'), e.get('To_Value')]:
                if pd.notna(v) and str(v).strip():
                    for part in str(v).split(','):
                        part = part.strip()
                        if part.isdigit():
                            jid = int(part)
                            sid = sprint_map.get((pid, jid), jid)
                            candidate_issues_by_sprint.setdefault(sid, set()).add(iid)

print(f"Total sprints with candidate issues: {len(candidate_issues_by_sprint)}")
sample_sids = list(candidate_issues_by_sprint.keys())[:5]
for sid in sample_sids:
    print(f"Sprint {sid} candidate issues count: {len(candidate_issues_by_sprint[sid])}")

