import os
import sys
import time
import json
import pandas as pd
import numpy as np

# Add benchmark_v2 to path
BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
sys.path.insert(0, BASE_DIR)

from replay import ReplayEngine, categorize_status

def main():
    print("=== Model 2 Dataset Generator ===")
    t_start = time.time()
    work_dir = os.path.join(BASE_DIR, "work")
    tawos_dir = r"d:\Gillo AI Service\tawos_csv"

    # 1. Load Sprints
    print("Loading sprint.csv...")
    sprint_df = pd.read_csv(os.path.join(tawos_dir, "sprint.csv"))
    sprint_df['Start_dt'] = pd.to_datetime(sprint_df['Start_Date'], errors='coerce')
    sprint_df['End_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')
    sprint_df['Complete_dt'] = pd.to_datetime(sprint_df['Complete_Date'], errors='coerce')

    total_sprints_raw = len(sprint_df)
    duration_days = (sprint_df['End_dt'] - sprint_df['Start_dt']).dt.total_seconds() / (24*3600)
    diff_complete_end = (sprint_df['Complete_dt'] - sprint_df['End_dt']).dt.total_seconds() / (24*3600)

    closed_mask = sprint_df['State'] == 'CLOSED'
    duration_mask = (duration_days >= 5.0) & (duration_days <= 35.0)
    complete_mask = (diff_complete_end <= 3.0) & diff_complete_end.notna()

    dropped_not_closed = int((~closed_mask).sum())
    dropped_duration = int((closed_mask & ~duration_mask).sum())
    dropped_late_complete = int((closed_mask & duration_mask & ~complete_mask).sum())

    eligible_sprints = sprint_df[closed_mask & duration_mask & complete_mask].copy()
    eligible_sprints['duration_days'] = duration_days[eligible_sprints.index]
    eligible_sprints = eligible_sprints.sort_values('End_dt').reset_index(drop=True)
    total_eligible_sprints = len(eligible_sprints)

    print(f"Total raw sprints: {total_sprints_raw}")
    print(f"  Dropped not CLOSED: {dropped_not_closed}")
    print(f"  Dropped duration not in [5, 35] days: {dropped_duration}")
    print(f"  Dropped Complete_Date - End_Date > 3 days: {dropped_late_complete}")
    print(f"Eligible sprints: {total_eligible_sprints}")

    sprint_map = {(r.Project_ID, r.JiraID): r.ID for r in sprint_df.itertuples()}
    sprint_info = {r.ID: r for r in sprint_df.itertuples()}

    # 2. Global Date Cutoffs and Splits
    D1 = eligible_sprints['End_dt'].quantile(0.70)
    D2 = eligible_sprints['End_dt'].quantile(0.85)

    print(f"Cutoffs: D1 (70th pct) = {D1}, D2 (85th pct) = {D2}")

    train_mask = eligible_sprints['End_dt'] <= D1
    val_mask = (eligible_sprints['Start_dt'] >= D1) & (eligible_sprints['End_dt'] <= D2)
    test_mask = eligible_sprints['Start_dt'] >= D2
    straddle_mask = ~(train_mask | val_mask | test_mask)

    dropped_straddling = int(straddle_mask.sum())
    print(f"Straddling sprints dropped: {dropped_straddling}")

    split_by_sprint_id = {}
    for r in eligible_sprints[train_mask].itertuples():
        split_by_sprint_id[r.ID] = 'train'
    for r in eligible_sprints[val_mask].itertuples():
        split_by_sprint_id[r.ID] = 'val'
    for r in eligible_sprints[test_mask].itertuples():
        split_by_sprint_id[r.ID] = 'test'
    for r in eligible_sprints[straddle_mask].itertuples():
        split_by_sprint_id[r.ID] = 'straddle'

    # Held out 5 projects variant (fixed seed 42, >= 20 eligible sprints)
    proj_sprint_counts = eligible_sprints.groupby('Project_ID').size()
    eligible_pids = sorted(proj_sprint_counts[proj_sprint_counts >= 20].index.tolist())
    np.random.seed(42)
    held_out_5_pids = sorted(np.random.choice(eligible_pids, size=5, replace=False).tolist())
    print(f"Held out 5 projects (seed 42): {held_out_5_pids}")

    # 3. Load issues and changelog
    print("Loading issues...")
    closed_sprint_ids = set(sprint_df[sprint_df['State'] == 'CLOSED']['ID'])
    issues_df = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=[
        'ID', 'Sprint_ID', 'Project_ID', 'Status', 'Creation_Date', 'Story_Point', 'Assignee_ID', 'Type', 'Priority', 'Resolution'
    ], low_memory=False)
    issues_df = issues_df[issues_df['Sprint_ID'].isin(closed_sprint_ids)].copy()
    issues_df['Creation_Date'] = pd.to_datetime(issues_df['Creation_Date'], errors='coerce')
    print(f"Closed-sprint issues loaded: {len(issues_df)}")

    print("Loading changelog_slim.parquet...")
    changelog = pd.read_parquet(os.path.join(work_dir, "changelog_slim.parquet"))
    changelog['Creation_Date'] = pd.to_datetime(changelog['Creation_Date'], errors='coerce')

    print("Indexing changelog events by issue...")
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

    # Candidate issues map: sprint_id -> set of issue_ids
    print("Building candidate issues per sprint...")
    candidates_by_sprint = {}
    for iid, iss in issues_dict.items():
        # From initial sprint set
        for sid in engine.initial_sprint_sets.get(iid, set()):
            candidates_by_sprint.setdefault(sid, set()).add(iid)
        # From Sprint events
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
                                candidates_by_sprint.setdefault(sid, set()).add(iid)

    # 4. Precompute Sprint-level carry-over rates for past sprints
    # For every eligible sprint, determine which issues were in it at End_dt and whether they were done
    print("Precomputing sprint-level outcomes at End_Date...")
    sprint_end_outcomes = {} # sid -> {'total': int, 'not_done': int, 'rate': float}
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

    # Helper for project_prev_3_carry_rate:
    # previous sprints in same project that ENDED before this sprint's Start_Date
    sprints_by_project = {}
    for r in eligible_sprints.itertuples():
        sprints_by_project.setdefault(r.Project_ID, []).append(r)

    def get_prev_3_rate(proj_id, sprint_start_dt):
        p_sprints = sprints_by_project.get(proj_id, [])
        # Ended before sprint_start_dt
        ended_before = [s for s in p_sprints if s.End_dt < sprint_start_dt]
        if not ended_before:
            return np.nan
        # Take up to 3 most recent
        ended_before = sorted(ended_before, key=lambda s: s.End_dt, reverse=True)[:3]
        rates = [sprint_end_outcomes[s.ID]['rate'] for s in ended_before if not np.isnan(sprint_end_outcomes[s.ID]['rate'])]
        return float(np.mean(rates)) if rates else np.nan

    # 5. Generate Snapshot Instances
    print("Generating snapshot rows for eligible sprints at f in {0.25, 0.5, 0.75}...")
    fractions = [0.25, 0.5, 0.75]
    all_rows = []
    total_already_done_excluded = 0
    total_open_tasks_included = 0

    sprint_loop_t0 = time.time()
    for idx, s in enumerate(eligible_sprints.itertuples()):
        if (idx + 1) % 500 == 0 or (idx + 1) == total_eligible_sprints:
            print(f"Processed {idx+1}/{total_eligible_sprints} sprints ({len(all_rows)} task instances, {time.time()-sprint_loop_t0:.1f}s)...")

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

            # Step A: Identify all members at t and classify as open or done
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

            # Step B: Gather task state at t for open tasks
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

                # Label at End_Date: status != 'done'
                final_stat_cat = engine.status_category_at(iid, s_end)
                label = 1 if (final_stat_cat != 'done') else 0

                res_val = issues_dict[iid].get('resolution')
                task_data_list.append((iid, feats, label, res_val))

            sprint_open_tasks = len(open_task_ids)
            tot_pts = done_points_at_t + sprint_open_points
            frac_done = (done_points_at_t / tot_pts) if tot_pts > 0 else 0.0

            # Step C: Assemble instances
            for iid, feats, label, res_val in task_data_list:
                total_open_tasks_included += 1
                as_id = feats['assignee_id']
                as_open_pts = assignee_open_pts_map.get(as_id, 0.0) if as_id is not None else 0.0
                as_open_cnt = assignee_open_cnt_map.get(as_id, 0) if as_id is not None else 0

                row = {
                    # Identifiers / Split metadata
                    'issue_id': iid,
                    'sprint_id': sid,
                    'project_id': pid,
                    'split': s_split,
                    'is_heldout_project': is_heldout,
                    'final_resolution': str(res_val) if pd.notna(res_val) else '',
                    'label': label,

                    # Sprint / Team Features
                    'f': f,
                    'sprint_duration': s_dur,
                    'sprint_open_tasks': sprint_open_tasks,
                    'sprint_open_points': round(sprint_open_points, 2),
                    'fraction_points_done': round(frac_done, 4),
                    'assignee_open_points': round(as_open_pts, 2),
                    'assignee_open_tasks': as_open_cnt,
                    'project_prev_3_carry_rate': round(prev_3_rate, 4) if not np.isnan(prev_3_rate) else np.nan,

                    # Task Features
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

    df_dataset = pd.DataFrame(all_rows)
    print(f"\nGenerated total dataset rows: {len(df_dataset)}")
    print(f"Total already-done task snapshot instances excluded: {total_already_done_excluded}")

    # Save to parquet
    out_parquet = os.path.join(work_dir, "model2_dataset.parquet")
    df_dataset.to_parquet(out_parquet, index=False)
    print(f"Saved dataset to {out_parquet}")

    # 6. Compute Funnel, Base Rates, and Stats
    # Exclusion reasons for resolutions: Won't Fix, Invalid, Duplicate, Cannot Reproduce
    invalid_res_set = {"won't fix", "invalid", "duplicate", "cannot reproduce"}
    wont_fix_share = float(df_dataset['final_resolution'].str.lower().isin(invalid_res_set).mean())

    # Funnel
    funnel = {
        'total_sprints_in_sprint_csv': total_sprints_raw,
        'dropped_not_closed': dropped_not_closed,
        'dropped_duration_outside_5_35_days': dropped_duration,
        'dropped_late_complete_gt_3_days': dropped_late_complete,
        'eligible_sprints': total_eligible_sprints,
        'train_sprints': int(train_mask.sum()),
        'val_sprints': int(val_mask.sum()),
        'test_sprints': int(test_mask.sum()),
        'straddling_sprints_dropped': dropped_straddling,
        'total_task_snapshot_rows': len(df_dataset),
        'already_done_instances_excluded': total_already_done_excluded,
        'share_rows_invalid_duplicate_resolution': wont_fix_share
    }

    # Split stats
    split_stats = {}
    for sp_name in ['train', 'val', 'test', 'straddle']:
        sub = df_dataset[df_dataset['split'] == sp_name]
        n_rows = len(sub)
        n_sprints = sub['sprint_id'].nunique()
        n_projs = sub['project_id'].nunique()
        base_rate = float(sub['label'].mean()) if n_rows > 0 else 0.0
        # Largest project share
        if n_rows > 0:
            top_proj_share = float(sub['project_id'].value_counts().iloc[0] / n_rows)
            top_proj_id = int(sub['project_id'].value_counts().index[0])
        else:
            top_proj_share = 0.0
            top_proj_id = None

        split_stats[sp_name] = {
            'rows': n_rows,
            'sprints': n_sprints,
            'projects': n_projs,
            'base_rate': base_rate,
            'largest_project_id': top_proj_id,
            'largest_project_share': top_proj_share
        }

    # How many test issues also appear in train sprints
    train_issues = set(df_dataset[df_dataset['split'] == 'train']['issue_id'])
    test_issues = set(df_dataset[df_dataset['split'] == 'test']['issue_id'])
    overlap_issues = train_issues.intersection(test_issues)
    test_issue_overlap = {
        'total_test_issues': len(test_issues),
        'test_issues_also_in_train': len(overlap_issues),
        'test_issues_overlap_pct': float(len(overlap_issues) / len(test_issues)) if test_issues else 0.0
    }

    # Base rate per fraction
    frac_base_rates = {}
    for f in fractions:
        sub_f = df_dataset[df_dataset['f'] == f]
        frac_base_rates[str(f)] = {
            'total_rows': len(sub_f),
            'base_rate': float(sub_f['label'].mean()),
            'train_base_rate': float(sub_f[sub_f['split'] == 'train']['label'].mean()),
            'val_base_rate': float(sub_f[sub_f['split'] == 'val']['label'].mean()),
            'test_base_rate': float(sub_f[sub_f['split'] == 'test']['label'].mean())
        }

    # Base rate per project (top 15 projects)
    proj_base_rates = {}
    top_pids = df_dataset['project_id'].value_counts().head(15).index.tolist()
    for pid in top_pids:
        sub_p = df_dataset[df_dataset['project_id'] == pid]
        proj_base_rates[int(pid)] = {
            'total_rows': len(sub_p),
            'sprints': sub_p['sprint_id'].nunique(),
            'base_rate': float(sub_p['label'].mean())
        }

    # Held out projects stats
    heldout_sub = df_dataset[df_dataset['is_heldout_project'] == 1]
    heldout_stats = {
        'heldout_pids': held_out_5_pids,
        'rows': len(heldout_sub),
        'sprints': heldout_sub['sprint_id'].nunique(),
        'base_rate': float(heldout_sub['label'].mean()) if len(heldout_sub) > 0 else 0.0
    }

    metrics_out = {
        'funnel': funnel,
        'split_stats': split_stats,
        'test_issue_overlap': test_issue_overlap,
        'fraction_base_rates': frac_base_rates,
        'project_base_rates': proj_base_rates,
        'heldout_stats': heldout_stats,
        'cutoffs': {'D1': str(D1), 'D2': str(D2)},
        'duration_seconds': round(time.time() - t_start, 2)
    }

    metrics_file = os.path.join(work_dir, "model2_funnel_and_split_stats.json")
    with open(metrics_file, 'w') as fp:
        json.dump(metrics_out, fp, indent=2)
    print(f"Saved metrics to {metrics_file}")
    print(f"=== Dataset Generation Completed in {time.time()-t_start:.1f}s ===")

if __name__ == "__main__":
    main()

