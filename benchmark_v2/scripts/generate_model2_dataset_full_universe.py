"""
benchmark_v2/scripts/generate_model2_dataset_full_universe.py

Rebuilds Model 2 dataset using the FULL UNIVERSE of issues:
- Includes issues whose initial sprint or changelog sprint was in an eligible sprint
  (capturing carry-overs moved to backlog or subsequent sprints).
- Excludes the 382 status-less issues across all splits (Item 2).
- Uses corrected status mapping in replay.py (Item 4).
- Retains identical D1/D2 temporal cutoffs and split definitions.
- Saves to benchmark_v2/work/model2_dataset_full_universe.parquet.
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

from replay import ReplayEngine, categorize_status

def main():
    print("=== Generating Model 2 Full Universe Dataset ===")
    t_start = time.time()

    # 1. Load Sprints and compute eligibility
    print("Loading sprint.csv...")
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
    eligible_sprint_ids = set(eligible_sprints['ID'].astype(int).tolist())
    print(f"Total eligible closed sprints: {total_eligible_sprints}")

    sprint_map = {(int(r.Project_ID), int(r.JiraID)): int(r.ID) for r in sprint_df.itertuples() if pd.notna(r.Project_ID) and pd.notna(r.JiraID)}

    # Global Date Cutoffs and Splits
    D1 = eligible_sprints['End_dt'].quantile(0.70)
    D2 = eligible_sprints['End_dt'].quantile(0.85)
    print(f"Cutoffs: D1 = {D1}, D2 = {D2}")

    train_mask = eligible_sprints['End_dt'] <= D1
    val_mask = (eligible_sprints['Start_dt'] >= D1) & (eligible_sprints['End_dt'] <= D2)
    test_mask = eligible_sprints['Start_dt'] >= D2
    straddle_mask = ~(train_mask | val_mask | test_mask)

    split_by_sprint_id = {}
    for r in eligible_sprints[train_mask].itertuples():
        split_by_sprint_id[r.ID] = 'train'
    for r in eligible_sprints[val_mask].itertuples():
        split_by_sprint_id[r.ID] = 'val'
    for r in eligible_sprints[test_mask].itertuples():
        split_by_sprint_id[r.ID] = 'test'
    for r in eligible_sprints[straddle_mask].itertuples():
        split_by_sprint_id[r.ID] = 'straddle'

    # Held out 5 projects variant (fixed seed 42)
    proj_sprint_counts = eligible_sprints.groupby('Project_ID').size()
    eligible_pids = sorted(proj_sprint_counts[proj_sprint_counts >= 20].index.tolist())
    np.random.seed(42)
    held_out_5_pids = sorted(np.random.choice(eligible_pids, size=5, replace=False).tolist())
    print(f"Held out 5 projects: {held_out_5_pids}")

    # 2. Load Full Universe changelog
    print("Loading changelog_full_universe.parquet...")
    cl_path = os.path.join(work_dir, "changelog_full_universe.parquet")
    changelog = pd.read_parquet(cl_path)
    changelog['Creation_Date'] = pd.to_datetime(changelog['Creation_Date'], errors='coerce')
    target_issue_ids = set(changelog['Issue_ID'].unique())
    print(f"Loaded changelog rows: {len(changelog):,} across {len(target_issue_ids):,} issues.")

    # 3. Load issues metadata for full universe
    print("Loading issue.csv for target issues...")
    issues_all = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=[
        'ID', 'Sprint_ID', 'Project_ID', 'Status', 'Creation_Date', 'Story_Point', 'Assignee_ID', 'Type', 'Priority', 'Resolution'
    ], low_memory=False)
    issues_all.columns = [c.lstrip('\ufeff') for c in issues_all.columns]
    issues_df = issues_all[issues_all['ID'].isin(target_issue_ids)].copy()
    issues_df['Creation_Date'] = pd.to_datetime(issues_df['Creation_Date'], errors='coerce')
    print(f"Loaded issue records: {len(issues_df):,}")

    events_by_issue = {}
    for iid, group in changelog.groupby('Issue_ID'):
        events_by_issue[int(iid)] = group.sort_values('Creation_Date').to_dict('records')

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

    print("Initializing ReplayEngine...")
    engine = ReplayEngine(issues_dict, events_by_issue, sprint_map)

    # Candidate issues per sprint (using full universe)
    print("Building candidate issues per sprint...")
    candidates_by_sprint = {}
    for iid, iss in issues_dict.items():
        # From initial sprint set
        for sid in engine.initial_sprint_sets.get(iid, set()):
            if sid in eligible_sprint_ids:
                candidates_by_sprint.setdefault(sid, set()).add(iid)
        # From Sprint events
        evs = events_by_issue.get(iid, [])
        pid = iss.get('project_id')
        for e in evs:
            if str(e['Field']).lower() == 'sprint':
                for v in [e.get('From_Value'), e.get('To_Value')]:
                    if pd.notna(v) and str(v).strip():
                        for part in str(v).split(','):
                            part = part.strip()
                            if part.isdigit():
                                jid = int(part)
                                sid = sprint_map.get((pid, jid), jid)
                                if sid in eligible_sprint_ids:
                                    candidates_by_sprint.setdefault(sid, set()).add(iid)

    print(f"Sprints with candidate issues: {len(candidates_by_sprint):,}")

    # 4. Precompute sprint-level outcomes at End_Date
    print("Precomputing sprint-level outcomes at End_Date...")
    sprint_end_outcomes = {}
    for r in eligible_sprints.itertuples():
        sid = r.ID
        cands = candidates_by_sprint.get(sid, set())
        tot = 0
        not_done = 0
        for iid in cands:
            if sid in engine.membership_at(iid, r.End_dt):
                tot += 1
                if engine.status_category_at(iid, r.End_dt) != 'done':
                    not_done += 1
        rate = (not_done / tot) if tot > 0 else np.nan
        sprint_end_outcomes[sid] = {'total': tot, 'not_done': not_done, 'rate': rate, 'end_dt': r.End_dt, 'start_dt': r.Start_dt, 'project_id': r.Project_ID}

    sprints_by_project = {}
    for r in eligible_sprints.itertuples():
        sprints_by_project.setdefault(r.Project_ID, []).append(r)

    def get_prev_3_rate(proj_id, sprint_start_dt):
        p_sprints = sprints_by_project.get(proj_id, [])
        ended_before = [s for s in p_sprints if s.End_dt < sprint_start_dt]
        if not ended_before:
            return np.nan
        ended_before = sorted(ended_before, key=lambda s: s.End_dt, reverse=True)[:3]
        rates = [sprint_end_outcomes[s.ID]['rate'] for s in ended_before if not np.isnan(sprint_end_outcomes[s.ID]['rate'])]
        return float(np.mean(rates)) if rates else np.nan

    # 5. Generate Snapshot Instances
    print("Generating snapshot rows for full universe at f in {0.25, 0.5, 0.75}...")
    fractions = [0.25, 0.5, 0.75]
    all_rows = []
    total_already_done_excluded = 0
    total_open_tasks_included = 0

    sprint_loop_t0 = time.time()
    for idx, s in enumerate(eligible_sprints.itertuples()):
        if (idx + 1) % 500 == 0 or (idx + 1) == total_eligible_sprints:
            print(f"Processed {idx+1}/{total_eligible_sprints} sprints ({len(all_rows):,} task instances, {time.time()-sprint_loop_t0:.1f}s)...")

        sid = s.ID
        pid = s.Project_ID
        s_dur = float(s.duration_days)
        s_start = s.Start_dt
        s_end = s.End_dt
        s_split = split_by_sprint_id.get(sid, 'unknown')
        is_heldout = 1 if pid in held_out_5_pids else 0

        prev_3_rate = get_prev_3_rate(pid, s_start)
        cands = candidates_by_sprint.get(sid, set())
        if not cands:
            continue

        for f in fractions:
            t = s_start + pd.Timedelta(seconds=f * s_dur * 86400.0)

            open_task_ids = []
            done_points_at_t = 0.0

            for iid in cands:
                mem = engine.membership_at(iid, t)
                if sid in mem:
                    stat_cat = engine.status_category_at(iid, t)
                    if stat_cat == 'done':
                        total_already_done_excluded += 1
                        sp = engine.story_points_at(iid, t)
                        if sp is not None and sp > 0:
                            done_points_at_t += float(sp)
                    else:
                        open_task_ids.append(iid)

            if not open_task_ids:
                continue

            task_data_list = []
            assignee_open_pts_map = {}
            assignee_open_cnt_map = {}
            sprint_open_points = 0.0

            for iid in open_task_ids:
                feats = engine.compute_all_task_features(iid, sid, t, s_start)
                sp_val = feats['story_points']
                sprint_open_points += sp_val
                as_id = feats['assignee_id']
                if as_id is not None:
                    assignee_open_pts_map[as_id] = assignee_open_pts_map.get(as_id, 0.0) + sp_val
                    assignee_open_cnt_map[as_id] = assignee_open_cnt_map.get(as_id, 0) + 1

                final_stat_cat = engine.status_category_at(iid, s_end)
                label = 1 if (final_stat_cat != 'done') else 0
                res_val = issues_dict[iid].get('resolution')
                task_data_list.append((iid, feats, label, res_val))

            sprint_open_tasks = len(open_task_ids)
            tot_pts = done_points_at_t + sprint_open_points
            frac_done = (done_points_at_t / tot_pts) if tot_pts > 0 else 0.0

            for iid, feats, label, res_val in task_data_list:
                total_open_tasks_included += 1
                as_id = feats['assignee_id']
                as_open_pts = assignee_open_pts_map.get(as_id, 0.0) if as_id is not None else 0.0
                as_open_cnt = assignee_open_cnt_map.get(as_id, 0) if as_id is not None else 0

                row = {
                    'issue_id': iid,
                    'sprint_id': sid,
                    'project_id': pid,
                    'split': s_split,
                    'is_heldout_project': is_heldout,
                    'final_resolution': str(res_val) if pd.notna(res_val) else '',
                    'label': label,
                    'f': f,
                    'sprint_duration': s_dur,
                    'sprint_open_tasks': sprint_open_tasks,
                    'sprint_open_points': round(sprint_open_points, 2),
                    'fraction_points_done': round(frac_done, 4),
                    'assignee_open_points': round(as_open_pts, 2),
                    'assignee_open_tasks': as_open_cnt,
                    'project_prev_3_carry_rate': round(prev_3_rate, 4) if not np.isnan(prev_3_rate) else np.nan,
                    'status_category': feats['status_category'],
                    'status_name': feats['status_name'],
                    'days_in_current_status': feats['days_in_current_status'],
                    'days_since_last_event': feats['days_since_last_event'],
                    'status_changes_so_far': feats['status_changes_so_far'],
                    'story_points': feats['story_points'],
                    'has_sp': feats['has_sp'],
                    'added_after_sprint_start': feats['added_after_sprint_start'],
                    'carry_count': feats['carry_count'],
                    'assignee_present': feats['assignee_present'],
                    'reassignments_so_far': feats['reassignments_so_far'],
                    'links_depends_on': feats['links_depends_on'],
                    'links_is_depended_on_by': feats['links_is_depended_on_by'],
                    'links_other': feats['links_other'],
                    'type': feats['type'],
                    'priority': feats['priority']
                }
                all_rows.append(row)

    df_full = pd.DataFrame(all_rows)
    print(f"\nTotal rows generated for full universe: {len(df_full):,}")
    print(f"Total already-done exclusions: {total_already_done_excluded:,}")

    out_dataset_path = os.path.join(work_dir, "model2_dataset_full_universe.parquet")
    df_full.to_parquet(out_dataset_path, index=False)
    print(f"Saved full universe dataset to {out_dataset_path} ({os.path.getsize(out_dataset_path):,} bytes).")
    print(f"Total time elapsed: {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()

