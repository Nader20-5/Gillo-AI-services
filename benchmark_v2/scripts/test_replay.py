import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add benchmark_v2 to path
BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
sys.path.insert(0, BASE_DIR)

from replay import ReplayEngine, categorize_status

class TestHandBuiltMiniHistories(unittest.TestCase):
    def setUp(self):
        # Build mini history
        self.t0 = pd.Timestamp("2026-01-01 09:00:00")
        self.t1 = pd.Timestamp("2026-01-05 10:00:00")
        self.t2 = pd.Timestamp("2026-01-10 12:00:00")
        self.t3 = pd.Timestamp("2026-01-15 15:00:00")

        self.issues = {
            1: {
                'id': 1, 'project_id': 1, 'sprint_id': 100, 'status': 'Closed',
                'creation_date': self.t0, 'story_point': 5.0, 'assignee_id': 'alice',
                'type': 'Bug', 'priority': 'High'
            },
            2: {
                # Issue with NO changelog entries
                'id': 2, 'project_id': 1, 'sprint_id': 101, 'status': 'Open',
                'creation_date': self.t0, 'story_point': 3.0, 'assignee_id': 'bob',
                'type': 'Story', 'priority': 'Medium'
            }
        }

        self.events = {
            1: [
                # Sprint event at t1: moved from 100 to 101 (carry-over)
                {
                    'Field': 'Sprint', 'From_Value': '100', 'To_Value': '100, 101',
                    'From_String': 'Sprint 1', 'To_String': 'Sprint 1, Sprint 2',
                    'Creation_Date': self.t1
                },
                # Status event at t1: moved to in progress
                {
                    'Field': 'status', 'From_String': 'Open', 'To_String': 'In Progress',
                    'Creation_Date': self.t1
                },
                # Assignee event at t1: reassigned to charlie
                {
                    'Field': 'assignee', 'From_Value': 'alice', 'To_Value': 'charlie',
                    'Creation_Date': self.t1
                },
                # Link event at t2: depends on another task
                {
                    'Field': 'Link', 'To_String': 'This issue depends on PROJ-99',
                    'Creation_Date': self.t2
                },
                # Link event at t2: is depended on by another task
                {
                    'Field': 'Link', 'To_String': 'This issue is depended on by PROJ-100',
                    'Creation_Date': self.t2
                },
                # Story points event at t2: re-estimated from 5 to 8
                {
                    'Field': 'Story Points', 'From_String': '5', 'To_String': '8',
                    'Creation_Date': self.t2
                },
                # Status event at t3: Resolved (done)
                {
                    'Field': 'status', 'From_String': 'In Progress', 'To_String': 'Resolved',
                    'Creation_Date': self.t3
                }
            ],
            2: []
        }

        self.engine = ReplayEngine(self.issues, self.events)

    def test_membership_at(self):
        # Before t1: initial set from first row's From_Value
        self.assertEqual(self.engine.membership_at(1, pd.Timestamp("2026-01-03 00:00:00")), {100})
        # After t1, before t3: {100, 101}
        self.assertEqual(self.engine.membership_at(1, pd.Timestamp("2026-01-06 00:00:00")), {100, 101})
        # Issue 2 has no events -> {101}
        self.assertEqual(self.engine.membership_at(2, pd.Timestamp("2026-01-06 00:00:00")), {101})

    def test_status_category_at(self):
        # Before t1: initial status from first status row's From_String ('Open' -> 'todo')
        self.assertEqual(self.engine.status_category_at(1, pd.Timestamp("2026-01-03 00:00:00")), 'todo')
        # Between t1 and t3: 'In Progress' -> 'in_progress'
        self.assertEqual(self.engine.status_category_at(1, pd.Timestamp("2026-01-06 00:00:00")), 'in_progress')
        # After t3: 'Resolved' -> 'done'
        self.assertEqual(self.engine.status_category_at(1, pd.Timestamp("2026-01-16 00:00:00")), 'done')
        # Issue 2 has no events -> 'Open' -> 'todo'
        self.assertEqual(self.engine.status_category_at(2, pd.Timestamp("2026-01-06 00:00:00")), 'todo')

    def test_story_points_at(self):
        # Before t2: initial SP = 5.0
        self.assertEqual(self.engine.story_points_at(1, pd.Timestamp("2026-01-03 00:00:00")), 5.0)
        # After t2: updated SP = 8.0
        self.assertEqual(self.engine.story_points_at(1, pd.Timestamp("2026-01-12 00:00:00")), 8.0)
        # Issue 2: initial SP = 3.0
        self.assertEqual(self.engine.story_points_at(2, pd.Timestamp("2026-01-03 00:00:00")), 3.0)

    def test_assignee_at(self):
        # Before t1: initial assignee alice
        self.assertEqual(self.engine.assignee_at(1, pd.Timestamp("2026-01-03 00:00:00")), 'alice')
        # After t1: charlie
        self.assertEqual(self.engine.assignee_at(1, pd.Timestamp("2026-01-06 00:00:00")), 'charlie')
        # Issue 2: bob
        self.assertEqual(self.engine.assignee_at(2, pd.Timestamp("2026-01-03 00:00:00")), 'bob')

    def test_links_before(self):
        # Before t2: 0 links
        l0 = self.engine.links_before(1, pd.Timestamp("2026-01-03 00:00:00"))
        self.assertEqual(l0, {'depends_on': 0, 'is_depended_on_by': 0, 'other': 0})
        # After t2: 1 depends_on, 1 is_depended_on_by
        l1 = self.engine.links_before(1, pd.Timestamp("2026-01-12 00:00:00"))
        self.assertEqual(l1, {'depends_on': 1, 'is_depended_on_by': 1, 'other': 0})


def run_leakage_test_200():
    print("\n--- Running Leakage Test on 200 Random (issue, sprint, t) from Real Data ---")
    work_dir = os.path.join(BASE_DIR, "work")
    tawos_dir = r"d:\Gillo AI Service\tawos_csv"

    changelog = pd.read_parquet(os.path.join(work_dir, "changelog_slim.parquet"))
    sprint_df = pd.read_csv(os.path.join(tawos_dir, "sprint.csv"))
    issues_df = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=[
        'ID', 'Sprint_ID', 'Project_ID', 'Status', 'Creation_Date', 'Story_Point', 'Assignee_ID', 'Type', 'Priority'
    ], low_memory=False)

    issues_df['Creation_Date'] = pd.to_datetime(issues_df['Creation_Date'], errors='coerce')
    changelog['Creation_Date'] = pd.to_datetime(changelog['Creation_Date'], errors='coerce')
    sprint_df['Start_dt'] = pd.to_datetime(sprint_df['Start_Date'], errors='coerce')
    sprint_df['End_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')
    sprint_df['Complete_dt'] = pd.to_datetime(sprint_df['Complete_Date'], errors='coerce')

    # Filter eligible sprints: CLOSED, 5-35 days, Complete - End <= 3 days
    duration_days = (sprint_df['End_dt'] - sprint_df['Start_dt']).dt.total_seconds() / (24*3600)
    diff_complete_end = (sprint_df['Complete_dt'] - sprint_df['End_dt']).dt.total_seconds() / (24*3600)
    eligible_sprints = sprint_df[
        (sprint_df['State'] == 'CLOSED') &
        (duration_days >= 5.0) & (duration_days <= 35.0) &
        (diff_complete_end <= 3.0) & (diff_complete_end.notna())
    ].copy()

    sprint_map = {(r.Project_ID, r.JiraID): r.ID for r in sprint_df.itertuples()}

    # Group events by issue_id
    print("Indexing events by issue...")
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
            'priority': r.Priority
        }

    full_engine = ReplayEngine(issues_dict, events_by_issue, sprint_map)

    # Sample 200 random eligible (issue, sprint, t)
    # Take issues that have events
    issues_with_events = [iid for iid, evs in events_by_issue.items() if len(evs) > 2]
    np.random.seed(42)
    sample_issue_ids = np.random.choice(issues_with_events, size=200, replace=False)
    
    passed_count = 0
    for idx, iid in enumerate(sample_issue_ids):
        evs = events_by_issue[iid]
        # Choose a random timestamp between first event and last event
        t_min = evs[0]['Creation_Date']
        t_max = evs[-1]['Creation_Date']
        if t_min >= t_max:
            t = t_min + pd.Timedelta(hours=1)
        else:
            delta_sec = (t_max - t_min).total_seconds()
            t = t_min + pd.Timedelta(seconds=np.random.uniform(0.1, 0.9) * delta_sec)

        sprint_id = issues_dict[iid].get('sprint_id', 1)
        sprint_start = t - pd.Timedelta(days=5)

        # 1. Compute state with FULL history
        mem_orig = full_engine.membership_at(iid, t)
        stat_orig = full_engine.status_category_at(iid, t)
        sp_orig = full_engine.story_points_at(iid, t)
        as_orig = full_engine.assignee_at(iid, t)
        links_orig = full_engine.links_before(iid, t)
        feats_orig = full_engine.compute_all_task_features(iid, sprint_id, t, sprint_start)

        # 2. Truncate history: delete every event after t
        truncated_events = {iid: [e for e in evs if e['Creation_Date'] < t]}
        trunc_engine = ReplayEngine(issues_dict, truncated_events, sprint_map, initial_states=full_engine.get_initial_states())

        mem_trunc = trunc_engine.membership_at(iid, t)
        stat_trunc = trunc_engine.status_category_at(iid, t)
        sp_trunc = trunc_engine.story_points_at(iid, t)
        as_trunc = trunc_engine.assignee_at(iid, t)
        links_trunc = trunc_engine.links_before(iid, t)
        feats_trunc = trunc_engine.compute_all_task_features(iid, sprint_id, t, sprint_start)

        # Assert identical
        assert mem_orig == mem_trunc, f"Membership mismatch at t for issue {iid}"
        assert stat_orig == stat_trunc, f"Status mismatch at t for issue {iid}"
        assert (sp_orig == sp_trunc) or (np.isnan(sp_orig) and np.isnan(sp_trunc)), f"SP mismatch at t for issue {iid}"
        assert as_orig == as_trunc, f"Assignee mismatch at t for issue {iid}"
        assert links_orig == links_trunc, f"Links mismatch at t for issue {iid}"
        assert feats_orig == feats_trunc, f"Features mismatch at t for issue {iid}"
        passed_count += 1

    print(f"Leakage Test PASSED: {passed_count}/200 random (issue, sprint, t) evaluations bit-for-bit identical!")
    return passed_count


if __name__ == "__main__":
    # 1. Run Mini history unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHandBuiltMiniHistories)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if not res.wasSuccessful():
        sys.exit(1)

    # 2. Run 200 random leakage tests
    passed = run_leakage_test_200()
    print("ALL UNIT TESTS AND LEAKAGE TESTS COMPLETED SUCCESSFULLY!")
