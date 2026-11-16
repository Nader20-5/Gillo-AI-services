"""
benchmark_v2/features_v2.py - Feature Extraction & Encoding Pipeline for Model 2 (Sprint Carry-Over Risk)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

NUMERICAL_FEATURES = [
    'f',
    'sprint_duration',
    'sprint_open_tasks',
    'sprint_open_points',
    'fraction_points_done',
    'assignee_open_points',
    'assignee_open_tasks',
    'project_prev_3_carry_rate',
    'days_in_current_status',
    'days_since_last_event',
    'status_changes_so_far',
    'story_points',
    'has_sp',
    'added_after_sprint_start',
    'carry_count',
    'assignee_present',
    'reassignments_so_far',
    'links_depends_on',
    'links_is_depended_on_by',
    'links_other'
]

TASK_LEVEL_NUMERICAL = [
    'days_in_current_status',
    'days_since_last_event',
    'status_changes_so_far',
    'story_points',
    'has_sp',
    'added_after_sprint_start',
    'carry_count',
    'assignee_present',
    'reassignments_so_far',
    'links_depends_on',
    'links_is_depended_on_by',
    'links_other'
]

TEAM_LEVEL_NUMERICAL = [
    'f',
    'sprint_duration',
    'sprint_open_tasks',
    'sprint_open_points',
    'fraction_points_done',
    'assignee_open_points',
    'assignee_open_tasks',
    'project_prev_3_carry_rate'
]

CATEGORICAL_FEATURES = [
    'status_category',
    'status_name',
    'type',
    'priority'
]

class Model2FeatureTransformer:
    def __init__(self, top_n_status: int = 15, top_n_types: int = 8, top_n_priorities: int = 6):
        self.top_n_status = top_n_status
        self.top_n_types = top_n_types
        self.top_n_priorities = top_n_priorities

        self.top_statuses: List[str] = []
        self.top_types: List[str] = []
        self.top_priorities: List[str] = []
        self.carry_rate_median: float = 0.5
        self.feature_names_: List[str] = []
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame):
        # Learn top categories on train data only
        self.top_statuses = df['status_name'].value_counts().head(self.top_n_status).index.tolist()
        self.top_types = df['type'].value_counts().head(self.top_n_types).index.tolist()
        self.top_priorities = df['priority'].value_counts().head(self.top_n_priorities).index.tolist()

        # Learn median carry rate
        valid_rates = df['project_prev_3_carry_rate'].dropna()
        self.carry_rate_median = float(valid_rates.median()) if len(valid_rates) > 0 else 0.5

        # Build feature names
        names = list(NUMERICAL_FEATURES)
        # indicator for carry rate missing
        names.append('project_prev_3_carry_rate_isna')

        # One-hot features
        names.append('status_category_is_todo')
        for s in self.top_statuses:
            names.append(f"status_name_{s}")
        names.append("status_name_other")

        for t in self.top_types:
            names.append(f"type_{t}")
        names.append("type_other")

        for p in self.top_priorities:
            names.append(f"priority_{p}")
        names.append("priority_other")

        self.feature_names_ = names
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ValueError("Model2FeatureTransformer is not fitted yet.")

        out = pd.DataFrame(index=df.index)

        # 1. Numerical features
        for col in NUMERICAL_FEATURES:
            if col == 'project_prev_3_carry_rate':
                isna = df[col].isna().astype(float)
                filled = df[col].fillna(self.carry_rate_median).astype(float)
                out[col] = filled
                out['project_prev_3_carry_rate_isna'] = isna
            else:
                out[col] = df[col].fillna(0.0).astype(float)

        # 2. status_category
        out['status_category_is_todo'] = (df['status_category'].str.lower() == 'todo').astype(float)

        # 3. status_name one-hot
        s_series = df['status_name'].fillna('')
        for s in self.top_statuses:
            out[f"status_name_{s}"] = (s_series == s).astype(float)
        out["status_name_other"] = (~s_series.isin(self.top_statuses)).astype(float)

        # 4. type one-hot
        t_series = df['type'].fillna('')
        for t in self.top_types:
            out[f"type_{t}"] = (t_series == t).astype(float)
        out["type_other"] = (~t_series.isin(self.top_types)).astype(float)

        # 5. priority one-hot
        p_series = df['priority'].fillna('')
        for p in self.top_priorities:
            out[f"priority_{p}"] = (p_series == p).astype(float)
        out["priority_other"] = (~p_series.isin(self.top_priorities)).astype(float)

        return out[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names_)

    def get_task_feature_names(self) -> List[str]:
        """Returns only the task-level feature column names."""
        task_names = []
        for name in self.feature_names_:
            if any(name.startswith(p) for p in [
                'days_in_current_status', 'days_since_last_event', 'status_changes_so_far',
                'story_points', 'has_sp', 'added_after_sprint_start', 'carry_count',
                'assignee_present', 'reassignments_so_far', 'links_', 'status_category',
                'status_name', 'type', 'priority'
            ]):
                task_names.append(name)
        return task_names

    def get_team_feature_names(self) -> List[str]:
        """Returns only the sprint/team-level feature column names."""
        team_names = []
        for name in self.feature_names_:
            if any(name.startswith(p) for p in [
                'f', 'sprint_duration', 'sprint_open_tasks', 'sprint_open_points',
                'fraction_points_done', 'assignee_open_points', 'assignee_open_tasks',
                'project_prev_3_carry_rate'
            ]):
                team_names.append(name)
        return team_names


class TaskoveFeatureTransformer:
    def __init__(self, top_n_types: int = 8):
        self.top_n_types = top_n_types
        self.top_types: List[str] = []
        self.feature_names_: List[str] = []
        self.fitted_: bool = False

    def fit(self, df_sub: pd.DataFrame):
        self.top_types = df_sub['type'].value_counts().head(self.top_n_types).index.tolist()
        names = [
            'f',
            'sprint_open_tasks',
            'sprint_open_points',
            'fraction_points_done',
            'assignee_open_points',
            'assignee_open_tasks',
            'days_in_current_status',
            'days_since_last_event',
            'story_points',
            'has_sp',
            'added_after_sprint_start',
            'assignee_present',
            'links_depends_on',
            'status_category_is_todo'
        ]
        for t in self.top_types:
            names.append(f"type_{t}")
        names.append("type_other")
        self.feature_names_ = names
        self.fitted_ = True
        return self

    def transform(self, df_sub: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise ValueError("TaskoveFeatureTransformer is not fitted yet.")
        out = pd.DataFrame(index=df_sub.index)
        num_cols = [
            'f', 'sprint_open_tasks', 'sprint_open_points', 'fraction_points_done',
            'assignee_open_points', 'assignee_open_tasks', 'days_in_current_status',
            'days_since_last_event', 'story_points', 'has_sp', 'added_after_sprint_start',
            'assignee_present', 'links_depends_on'
        ]
        for c in num_cols:
            out[c] = df_sub[c].fillna(0.0).astype(float)
        out['status_category_is_todo'] = (df_sub['status_category'].str.lower() == 'todo').astype(float)

        t_series = df_sub['type'].fillna('')
        for t in self.top_types:
            out[f"type_{t}"] = (t_series == t).astype(float)
        out["type_other"] = (~t_series.isin(self.top_types)).astype(float)
        return out[self.feature_names_]

    def fit_transform(self, df_sub: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_sub).transform(df_sub)


