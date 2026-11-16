"""
benchmark_v2/replay.py - Temporal Replay Engine for Jira Agile State Reconstruction

Computes the exact state of any issue as of time t using strictly events with timestamp < t:
- membership_at(issue_id, t)
- status_category_at(issue_id, t)
- story_points_at(issue_id, t)
- assignee_at(issue_id, t)
- links_before(issue_id, t)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Set, Optional, Any, List

TODO_SET = {
    'open', 'to do', 'to_do', 'todo', 'new', 'reopened', 'backlog', 'created',
    'needs triage', 'triage', 'submitted', 'untriaged', 'unconfirmed',
    'planning', 'selected for development', 'ready for work', 'ready',
    'deferred', 'defered', 'parked', 'on hold', 'blocked', 'waiting for information',
    'to develop', 'awaiting development', 'ready for development', 'ready to groom',
    'under consideration', 'future consideration', 'icebox', 'refine', 'scheduled',
    'raw', 'gathering impact', 'gathering interest', 'pending 3rd-party',
    'waiting for response', 'waiting for feedback', 'waiting (blocked)', 'monitor / on hold',
    'waiting for support', 'waiting for customer', 'escalated', 'assessment', 'assessed',
    'scoped', 'unplanned', 'open.'
}

DONE_SET = {
    'resolved', 'closed', 'done', 'complete', 'completed', 'verified',
    'delivered', 'released', 'finished', 'shipped', 'cancelled', 'canceled',
    "won't fix", 'invalid', 'duplicate', 'implemented', 'released to server',
    'released to cloud', 'awaiting release', 'waiting for release', 'not being considered'
}

IN_PROGRESS_SET = {
    'awaiting merge', 'to be merged', 'ready to merge', 'validate',
    'customer validation', 'in validation', 'in development', 'awaiting soak',
    'soaking', 'patch available', 'debugging with submitter', 'design',
    'needs verification', 'received', 'development complete', 'review complete'
}

def categorize_status(s: Optional[str]) -> str:
    """Categorizes a raw status string into 'todo', 'in_progress', or 'done'."""
    if not isinstance(s, str) or not s.strip():
        return 'todo'
    clean = ' '.join(s.strip().lower().split())
    if clean in DONE_SET:
        return 'done'
    if clean in TODO_SET:
        return 'todo'
    if clean in IN_PROGRESS_SET:
        return 'in_progress'
    if 'unverified' in clean:
        return 'in_progress'
    if any(w in clean for w in ['resolved', 'closed', 'done', 'complete', 'verified', 'shipped']):
        return 'done'
    if any(w in clean for w in ['to do', 'backlog', 'reopen', 'new', 'triage', 'deferred']):
        return 'todo'
    if any(w in clean for w in ['progress', 'review', 'testing', 'test', 'qa', 'accepted', 'active', 'pr', 'coding', 'deploy', 'investigat', 'working', 'merge']):
        return 'in_progress'
    return 'in_progress'


class ReplayEngine:
    def __init__(self, issues_dict: Dict[int, Dict[str, Any]], events_by_issue: Dict[int, List[Dict[str, Any]]], sprint_map: Optional[Dict[tuple, int]] = None, initial_states: Optional[Dict[str, Dict[int, Any]]] = None):
        """
        issues_dict: dict of issue_id -> {
            'id': int, 'project_id': int, 'sprint_id': Optional[int], 'status': str,
            'creation_date': pd.Timestamp, 'story_point': Optional[float], 'assignee_id': Optional[Any], ...
        }
        events_by_issue: dict of issue_id -> list of event dicts sorted by creation_date
        sprint_map: dict of (project_id, jira_sprint_id) -> sprint.csv ID
        initial_states: pre-computed initial states dict with keys 'sprints', 'statuses', 'sps', 'assignees'
        """
        self.issues = issues_dict
        self.events_by_issue = events_by_issue
        self.sprint_map = sprint_map or {}

        if initial_states is not None:
            self.initial_sprint_sets = dict(initial_states.get('sprints', {}))
            self.initial_status_names = dict(initial_states.get('statuses', {}))
            self.initial_sps = dict(initial_states.get('sps', {}))
            self.initial_assignees = dict(initial_states.get('assignees', {}))
        else:
            # Pre-index initial states per issue
            self.initial_sprint_sets = {}
            self.initial_status_names = {}
            self.initial_sps = {}
            self.initial_assignees = {}

            for iid, iss in self.issues.items():
                evs = self.events_by_issue.get(iid, [])
                pid = iss.get('project_id')

                # 1. Initial Sprint set
                sp_evs = [e for e in evs if e['Field'] == 'Sprint']
                if sp_evs:
                    f_val = sp_evs[0].get('From_Value', '')
                    self.initial_sprint_sets[iid] = self._parse_sprint_ids(f_val, pid)
                else:
                    s_id = iss.get('sprint_id')
                    self.initial_sprint_sets[iid] = {int(s_id)} if pd.notna(s_id) else set()

                # 2. Initial Status
                st_evs = [e for e in evs if e['Field'] == 'status']
                if st_evs:
                    f_str = st_evs[0].get('From_String', '')
                    self.initial_status_names[iid] = f_str if f_str else iss.get('status', 'Open')
                else:
                    self.initial_status_names[iid] = iss.get('status', 'Open')

                # 3. Initial Story Points
                sp_pt_evs = [e for e in evs if 'story' in str(e['Field']).lower()]
                if sp_pt_evs:
                    f_sp = sp_pt_evs[0].get('From_String') or sp_pt_evs[0].get('From_Value')
                    try:
                        self.initial_sps[iid] = float(f_sp) if pd.notna(f_sp) and str(f_sp).strip() else None
                    except (ValueError, TypeError):
                        self.initial_sps[iid] = None
                    if self.initial_sps[iid] is None and iss.get('story_point') is not None:
                        # If issue was never changed before first edit or had baseline
                        self.initial_sps[iid] = float(iss['story_point']) if pd.notna(iss['story_point']) else None
                else:
                    sp_val = iss.get('story_point')
                    self.initial_sps[iid] = float(sp_val) if pd.notna(sp_val) else None

                # 4. Initial Assignee
                as_evs = [e for e in evs if e['Field'] == 'assignee']
                if as_evs:
                    f_as = as_evs[0].get('From_Value') or as_evs[0].get('From_String')
                    self.initial_assignees[iid] = str(f_as).strip() if pd.notna(f_as) and str(f_as).strip() else None
                else:
                    as_val = iss.get('assignee_id')
                    self.initial_assignees[iid] = str(as_val).strip() if pd.notna(as_val) and str(as_val).strip() else None

    def get_initial_states(self) -> Dict[str, Dict[int, Any]]:
        return {
            'sprints': self.initial_sprint_sets,
            'statuses': self.initial_status_names,
            'sps': self.initial_sps,
            'assignees': self.initial_assignees,
        }

    def _parse_sprint_ids(self, val: Any, project_id: Optional[int]) -> Set[int]:
        if pd.isna(val) or not str(val).strip():
            return set()
        s_ids = set()
        for part in str(val).split(','):
            part = part.strip()
            if part.isdigit():
                jid = int(part)
                # Map (project_id, jid) to sprint.ID if available
                sid = self.sprint_map.get((project_id, jid), jid)
                s_ids.add(sid)
        return s_ids

    def membership_at(self, issue_id: int, t: pd.Timestamp) -> Set[int]:
        """Returns the set of active sprint IDs for issue_id strictly before time t."""
        iss = self.issues.get(issue_id)
        if not iss:
            return set()
        c_date = iss.get('creation_date')
        if c_date is not None and t < c_date:
            return set()

        evs = self.events_by_issue.get(issue_id, [])
        sp_evs = [e for e in evs if e['Field'] == 'Sprint']

        if not sp_evs:
            # Issues with no Sprint rows -> {issue.Sprint_ID} from Creation_Date
            return set(self.initial_sprint_sets.get(issue_id, set()))

        # Events strictly before t
        past_evs = [e for e in sp_evs if e['Creation_Date'] < t]
        if past_evs:
            latest = past_evs[-1]
            return self._parse_sprint_ids(latest.get('To_Value', ''), iss.get('project_id'))
        else:
            return set(self.initial_sprint_sets.get(issue_id, set()))

    def status_category_at(self, issue_id: int, t: pd.Timestamp) -> str:
        """Returns the status category ('todo', 'in_progress', 'done') strictly before time t."""
        raw_name = self.status_name_at(issue_id, t)
        return categorize_status(raw_name)

    def status_name_at(self, issue_id: int, t: pd.Timestamp) -> str:
        """Returns the raw status name strictly before time t."""
        iss = self.issues.get(issue_id)
        if not iss:
            return 'Open'

        evs = self.events_by_issue.get(issue_id, [])
        st_evs = [e for e in evs if e['Field'] == 'status']

        if not st_evs:
            return self.initial_status_names.get(issue_id, iss.get('status', 'Open'))

        past_evs = [e for e in st_evs if e['Creation_Date'] < t]
        if past_evs:
            latest = past_evs[-1]
            return str(latest.get('To_String', ''))
        else:
            return self.initial_status_names.get(issue_id, iss.get('status', 'Open'))

    def story_points_at(self, issue_id: int, t: pd.Timestamp) -> Optional[float]:
        """Returns the story point value strictly before time t, or None."""
        evs = self.events_by_issue.get(issue_id, [])
        sp_evs = [e for e in evs if 'story' in str(e['Field']).lower()]

        if not sp_evs:
            return self.initial_sps.get(issue_id)

        past_evs = [e for e in sp_evs if e['Creation_Date'] < t]
        if past_evs:
            latest = past_evs[-1]
            val = latest.get('To_String') or latest.get('To_Value')
            try:
                return float(val) if pd.notna(val) and str(val).strip() else None
            except (ValueError, TypeError):
                return None
        else:
            return self.initial_sps.get(issue_id)

    def assignee_at(self, issue_id: int, t: pd.Timestamp) -> Optional[str]:
        """Returns the assignee identifier strictly before time t, or None."""
        evs = self.events_by_issue.get(issue_id, [])
        as_evs = [e for e in evs if e['Field'] == 'assignee']

        if not as_evs:
            return self.initial_assignees.get(issue_id)

        past_evs = [e for e in as_evs if e['Creation_Date'] < t]
        if past_evs:
            latest = past_evs[-1]
            val = latest.get('To_Value') or latest.get('To_String')
            s_val = str(val).strip() if pd.notna(val) else ''
            return s_val if s_val else None
        else:
            return self.initial_assignees.get(issue_id)

    def links_before(self, issue_id: int, t: pd.Timestamp) -> Dict[str, int]:
        """Returns counts of dependency links created strictly before time t."""
        evs = self.events_by_issue.get(issue_id, [])
        link_evs = [e for e in evs if e['Field'] == 'Link' and e['Creation_Date'] < t]

        counts = {'depends_on': 0, 'is_depended_on_by': 0, 'other': 0}
        for e in link_evs:
            to_s = str(e.get('To_String', '')).lower()
            if 'is depended on by' in to_s or 'blocks' in to_s or 'done before' in to_s:
                counts['is_depended_on_by'] += 1
            elif 'depends on' in to_s or 'blocked by' in to_s or 'done after' in to_s:
                counts['depends_on'] += 1
            else:
                counts['other'] += 1
        return counts

    def compute_all_task_features(self, issue_id: int, sprint_id: int, t: pd.Timestamp, sprint_start: pd.Timestamp) -> Dict[str, Any]:
        """Computes all task-level temporal features strictly as of time t."""
        iss = self.issues.get(issue_id, {})
        evs = self.events_by_issue.get(issue_id, [])
        past_evs = [e for e in evs if e['Creation_Date'] < t]

        status_cat = self.status_category_at(issue_id, t)
        status_name = self.status_name_at(issue_id, t)

        # Status change events before t
        past_st_evs = [e for e in past_evs if e['Field'] == 'status']
        status_changes_so_far = len(past_st_evs)

        # Days in current status
        c_date = iss.get('creation_date') or t
        if past_st_evs:
            last_status_transition = past_st_evs[-1]['Creation_Date']
            days_in_status = max(0.0, (t - last_status_transition).total_seconds() / (24 * 3600))
        else:
            # If never changed: days since max(Creation_Date, time added to sprint)
            # Find when added to this sprint
            past_sp_evs = [e for e in past_evs if e['Field'] == 'Sprint']
            sprint_added_time = c_date
            for e in past_sp_evs:
                if sprint_id in self._parse_sprint_ids(e.get('To_Value', ''), iss.get('project_id')):
                    sprint_added_time = e['Creation_Date']
                    break
            base_time = max(c_date, sprint_added_time)
            days_in_status = max(0.0, (t - base_time).total_seconds() / (24 * 3600))

        # Days since last event
        if past_evs:
            last_event_time = max(e['Creation_Date'] for e in past_evs)
            days_since_last_event = max(0.0, (t - last_event_time).total_seconds() / (24 * 3600))
        else:
            days_since_last_event = max(0.0, (t - c_date).total_seconds() / (24 * 3600))

        # Story Points at t
        sp_val = self.story_points_at(issue_id, t)
        has_sp = 1 if (sp_val is not None and not np.isnan(sp_val) and sp_val > 0) else 0
        sp_clean = float(sp_val) if has_sp else 0.0

        # Added after sprint start flag
        # Check earliest time issue was added to this sprint
        past_sp_evs = [e for e in past_evs if e['Field'] == 'Sprint']
        first_time_in_sprint = c_date
        found_in_event = False
        for e in past_sp_evs:
            if sprint_id in self._parse_sprint_ids(e.get('To_Value', ''), iss.get('project_id')):
                first_time_in_sprint = e['Creation_Date']
                found_in_event = True
                break
        if not found_in_event:
            # Initial sprint set had it from creation
            first_time_in_sprint = c_date

        added_after_start = 1 if (first_time_in_sprint > sprint_start) else 0

        # Number of earlier sprints this issue belonged to before this sprint (carry_count)
        all_sprints_seen_before_this = set()
        for e in past_sp_evs:
            if e['Creation_Date'] < t:
                s_set = self._parse_sprint_ids(e.get('To_Value', ''), iss.get('project_id'))
                for sid in s_set:
                    if sid != sprint_id:
                        all_sprints_seen_before_this.add(sid)
        # Also check initial set
        for sid in self.initial_sprint_sets.get(issue_id, set()):
            if sid != sprint_id:
                all_sprints_seen_before_this.add(sid)
        carry_count = len(all_sprints_seen_before_this)

        # Assignee
        as_val = self.assignee_at(issue_id, t)
        assignee_present = 1 if as_val is not None else 0

        # Reassignments so far
        past_as_evs = [e for e in past_evs if e['Field'] == 'assignee']
        reassignments_so_far = len(past_as_evs)

        # Dependency links
        links = self.links_before(issue_id, t)

        return {
            'status_category': status_cat,
            'status_name': status_name,
            'days_in_current_status': round(days_in_status, 4),
            'days_since_last_event': round(days_since_last_event, 4),
            'status_changes_so_far': status_changes_so_far,
            'story_points': sp_clean,
            'has_sp': has_sp,
            'added_after_sprint_start': added_after_start,
            'carry_count': carry_count,
            'assignee_present': assignee_present,
            'assignee_id': as_val,
            'reassignments_so_far': reassignments_so_far,
            'links_depends_on': links['depends_on'],
            'links_is_depended_on_by': links['is_depended_on_by'],
            'links_other': links['other'],
            'type': iss.get('type', 'Unknown'),
            'priority': iss.get('priority', 'Unknown')
        }
