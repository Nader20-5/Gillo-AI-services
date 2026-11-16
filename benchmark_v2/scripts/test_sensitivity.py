import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
work_dir = os.path.join(BASE_DIR, "work")
tawos_dir = r"d:\Gillo AI Service\tawos_csv"
sys.path.insert(0, BASE_DIR)

from replay import ReplayEngine, categorize_status

print("Testing label sensitivity on test set at f=0.5...")
df_model2 = pd.read_parquet(os.path.join(work_dir, "model2_dataset.parquet"))
df_test_f50 = df_model2[(df_model2['split'] == 'test') & (df_model2['f'] == 0.5)].copy()
print(f"Loaded test f=0.5 instances: {len(df_test_f50)}")

# Load sprint info
sprint_df = pd.read_csv(os.path.join(tawos_dir, "sprint.csv"))
sprint_df['End_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')
sprint_df['Complete_dt'] = pd.to_datetime(sprint_df['Complete_Date'], errors='coerce')
sprint_end_map = sprint_df.set_index('ID')['End_dt'].to_dict()
sprint_complete_map = sprint_df.set_index('ID')['Complete_dt'].to_dict()

# Load changelog and issues
changelog = pd.read_parquet(os.path.join(work_dir, "changelog_slim.parquet"))
changelog['Creation_Date'] = pd.to_datetime(changelog['Creation_Date'], errors='coerce')

test_issue_ids = set(df_test_f50['issue_id'].unique())
print(f"Unique test issues: {len(test_issue_ids)}")

cl_test = changelog[changelog['Issue_ID'].isin(test_issue_ids)]
events_by_issue = {}
for iid, grp in cl_test.groupby('Issue_ID'):
    events_by_issue[int(iid)] = grp.sort_values('Creation_Date').to_dict('records')

issues_df = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=[
    'ID', 'Sprint_ID', 'Project_ID', 'Status', 'Creation_Date', 'Story_Point', 'Assignee_ID', 'Type', 'Priority', 'Resolution'
], low_memory=False)
issues_test = issues_df[issues_df['ID'].isin(test_issue_ids)]

issues_dict = {}
for r in issues_test.itertuples():
    issues_dict[int(r.ID)] = {
        'id': int(r.ID),
        'project_id': int(r.Project_ID) if pd.notna(r.Project_ID) else None,
        'sprint_id': int(r.Sprint_ID) if pd.notna(r.Sprint_ID) else None,
        'status': r.Status,
        'creation_date': pd.to_datetime(r.Creation_Date, errors='coerce'),
        'story_point': r.Story_Point,
        'assignee_id': r.Assignee_ID,
        'type': r.Type,
        'priority': r.Priority,
        'resolution': r.Resolution
    }

engine = ReplayEngine(issues_dict, events_by_issue)

labels_end = []
labels_end_plus_24h = []
labels_complete = []

for r in df_test_f50.itertuples():
    iid = r.issue_id
    sid = r.sprint_id
    end_dt = sprint_end_map[sid]
    comp_dt = sprint_complete_map.get(sid, end_dt)
    if pd.isna(comp_dt):
        comp_dt = end_dt

    # At End_Date
    stat_end = engine.status_category_at(iid, end_dt)
    l_end = 1 if stat_end != 'done' else 0

    # At End_Date + 24h
    stat_end_24 = engine.status_category_at(iid, end_dt + pd.Timedelta(hours=24))
    l_end_24 = 1 if stat_end_24 != 'done' else 0

    # At Complete_Date
    stat_comp = engine.status_category_at(iid, comp_dt)
    l_comp = 1 if stat_comp != 'done' else 0

    labels_end.append(l_end)
    labels_end_plus_24h.append(l_end_24)
    labels_complete.append(l_comp)

df_test_f50['l_end'] = labels_end
df_test_f50['l_end_24h'] = labels_end_plus_24h
df_test_f50['l_comp'] = labels_complete

print(f"Base rate at End_Date:         {np.mean(labels_end)*100:.2f}% (original: {df_test_f50['label'].mean()*100:.2f}%)")
print(f"Base rate at End_Date + 24h:   {np.mean(labels_end_plus_24h)*100:.2f}%")
print(f"Base rate at Complete_Date:    {np.mean(labels_complete)*100:.2f}%")

# Check matches with original label
matches = (df_test_f50['label'] == df_test_f50['l_end']).mean()
print(f"Matches original dataset label: {matches*100:.2f}%")

