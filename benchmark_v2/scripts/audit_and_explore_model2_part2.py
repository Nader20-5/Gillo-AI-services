import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
work_dir = os.path.join(BASE_DIR, "work")
tawos_dir = r"d:\Gillo AI Service\tawos_csv"
sys.path.insert(0, BASE_DIR)

from replay import ReplayEngine, categorize_status

print("=== AUDIT STEP 4: Investigating Sprints with No Open Tasks ===")
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
eligible_sprints['duration_days'] = duration_days[eligible_sprints.index]
eligible_sprints = eligible_sprints.sort_values('End_dt').reset_index(drop=True)
total_eligible_sprints = len(eligible_sprints)
print(f"Total eligible sprints: {total_eligible_sprints}")

# Check model2_dataset.parquet sprints
df_model2 = pd.read_parquet(os.path.join(work_dir, "model2_dataset.parquet"))
sprints_in_dataset = set(df_model2['sprint_id'].unique())
print(f"Sprints with >=1 open task in model2_dataset: {len(sprints_in_dataset)}")
sprints_without_open_tasks = set(eligible_sprints['ID']) - sprints_in_dataset
print(f"Eligible sprints without open tasks in dataset: {len(sprints_without_open_tasks)}")

# Let's see how many eligible sprints are in each split
D1 = eligible_sprints['End_dt'].quantile(0.70)
D2 = eligible_sprints['End_dt'].quantile(0.85)
train_mask = eligible_sprints['End_dt'] <= D1
val_mask = (eligible_sprints['Start_dt'] >= D1) & (eligible_sprints['End_dt'] <= D2)
test_mask = eligible_sprints['Start_dt'] >= D2
straddle_mask = ~(train_mask | val_mask | test_mask)

print("\n--- SPRINT COUNTS TABLE (Item 9) ---")
for name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask), ('Straddle', straddle_mask)]:
    s_ids = set(eligible_sprints[mask]['ID'])
    s_with_open = s_ids & sprints_in_dataset
    sub_rows = df_model2[df_model2['sprint_id'].isin(s_ids)]
    sub_rows_f50 = df_model2[(df_model2['sprint_id'].isin(s_ids)) & (df_model2['f'] == 0.5)]
    s_with_open_f50 = set(sub_rows_f50['sprint_id'].unique())
    print(f"{name:10s} | Total Sprints: {len(s_ids):4d} | Sprints w/ Any Open: {len(s_with_open):4d} | Sprints w/ f=0.5 Open: {len(s_with_open_f50):4d} | Total Rows: {len(sub_rows):6d} | f=0.5 Rows: {len(sub_rows_f50):5d}")

# Why do 665 (or 492) eligible sprints have no open tasks?
# Let's inspect some of these sprints
print("\n--- Why sprints have no open tasks (Item 7d) ---")
# Check how many issues in issue.csv map to these sprints
issues_df = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=['ID', 'Sprint_ID', 'Project_ID', 'Status', 'Creation_Date', 'Resolution'], low_memory=False)
sprint_issue_counts = issues_df.groupby('Sprint_ID').size()

no_issue_sprints = [sid for sid in sprints_without_open_tasks if sid not in sprint_issue_counts]
zero_or_done_sprints = [sid for sid in sprints_without_open_tasks if sid in sprint_issue_counts]
print(f"Eligible sprints without open tasks: {len(sprints_without_open_tasks)}")
print(f"  Of which 0 issues map to Sprint_ID in issue.csv: {len(no_issue_sprints)}")
print(f"  Of which issues exist but all were already done before snapshot (or empty at snapshots): {len(zero_or_done_sprints)}")

print("\n=== AUDIT STEP 5: Project 4 late-completion test (Item 8) ===")
# Compute the share of Project 4 tasks that change to done within 24h before End_Date vs other projects.
# Let's check status changes in changelog for issues in eligible sprints
changelog = pd.read_parquet(os.path.join(work_dir, "changelog_slim.parquet"))
changelog['Creation_Date'] = pd.to_datetime(changelog['Creation_Date'], errors='coerce')
status_evs = changelog[changelog['Field'] == 'status'].copy()

# Map status events to whether To_String is 'done'
status_evs['to_cat'] = status_evs['To_String'].apply(categorize_status)
done_transitions = status_evs[status_evs['to_cat'] == 'done'].copy()
print(f"Total transitions to 'done': {len(done_transitions):,}")

# Join with issues to get project_id and sprint_id
issues_map = issues_df.set_index('ID')[['Project_ID', 'Sprint_ID']].to_dict('index')
sprint_end_map = eligible_sprints.set_index('ID')['End_dt'].to_dict()

# Look at done transitions occurring within 24h before sprint End_Date
# For each done transition, check if it occurred in the issue's sprint and within 24h before sprint end
records = []
for r in done_transitions.itertuples():
    iss_info = issues_map.get(r.Issue_ID)
    if not iss_info:
        continue
    sid = iss_info['Sprint_ID']
    pid = iss_info['Project_ID']
    if sid in sprint_end_map:
        end_dt = sprint_end_map[sid]
        diff_hours = (end_dt - r.Creation_Date).total_seconds() / 3600.0
        # If changed to done during sprint and within 24h before end: 0 <= diff_hours <= 24
        is_last_24h = (0 <= diff_hours <= 24)
        is_during_sprint = (diff_hours >= 0)
        records.append({
            'issue_id': r.Issue_ID,
            'project_id': pid,
            'sprint_id': sid,
            'is_last_24h': is_last_24h,
            'is_during_sprint': is_during_sprint
        })

df_done_timing = pd.DataFrame(records)
print(f"Analyzed {len(df_done_timing)} done transitions in eligible sprints.")

# Group by project
p4_during = df_done_timing[(df_done_timing['project_id'] == 4) & (df_done_timing['is_during_sprint'])]
other_during = df_done_timing[(df_done_timing['project_id'] != 4) & (df_done_timing['is_during_sprint'])]

p4_share_24h = p4_during['is_last_24h'].mean() * 100
other_share_24h = other_during['is_last_24h'].mean() * 100
print(f"Project 4: {len(p4_during)} done events during sprint. Share in last 24h: {p4_share_24h:.2f}%")
print(f"Other Projects: {len(other_during)} done events during sprint. Share in last 24h: {other_share_24h:.2f}%")

# Also check top projects
for pid, grp in df_done_timing[df_done_timing['is_during_sprint']].groupby('project_id'):
    if len(grp) >= 200:
        print(f"  Project {pid:2d} ({len(grp):4d} done events): {grp['is_last_24h'].mean()*100:.2f}% in last 24h")

print("\n=== AUDIT STEP 6: Sprint Duration vs Carry Rate (Item 4) ===")
# Show partial dependence or mean carry rate by sprint_duration bucket to test "longer sprints slip more"
df_test = df_model2[df_model2['split'] == 'test']
df_test_f50 = df_test[df_test['f'] == 0.5]

print("Carry rate by sprint_duration in Test (f=0.5):")
duration_bins = [0, 7, 14, 21, 28, 35]
df_test_f50_dur = df_test_f50.copy()
df_test_f50_dur['dur_bucket'] = pd.cut(df_test_f50_dur['sprint_duration'], bins=duration_bins)
dur_summary = df_test_f50_dur.groupby('dur_bucket', observed=False).agg(
    tasks=('label', 'count'),
    carried=('label', 'sum'),
    carry_rate=('label', 'mean')
)
print(dur_summary)

print("\nCarry rate by sprint_duration across ENTIRE dataset (f=0.5):")
df_all_f50 = df_model2[df_model2['f'] == 0.5].copy()
df_all_f50['dur_bucket'] = pd.cut(df_all_f50['sprint_duration'], bins=duration_bins)
dur_all_summary = df_all_f50.groupby('dur_bucket', observed=False).agg(
    tasks=('label', 'count'),
    carried=('label', 'sum'),
    carry_rate=('label', 'mean')
)
print(dur_all_summary)

# Check correlation of sprint_duration with project_id or project identity
print("\nIs sprint_duration an identity proxy? Check duration std per project:")
proj_dur_stats = df_model2.groupby('project_id')['sprint_duration'].agg(['count', 'nunique', 'min', 'max', 'std'])
print(proj_dur_stats.head(15))

