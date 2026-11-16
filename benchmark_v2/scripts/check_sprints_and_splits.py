import os
import pandas as pd
import numpy as np

tawos_dir = r"d:\Gillo AI Service\tawos_csv"
sprint_df = pd.read_csv(os.path.join(tawos_dir, "sprint.csv"))

sprint_df['Start_dt'] = pd.to_datetime(sprint_df['Start_Date'], errors='coerce')
sprint_df['End_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')
sprint_df['Complete_dt'] = pd.to_datetime(sprint_df['Complete_Date'], errors='coerce')

total_sprints = len(sprint_df)
not_closed = (sprint_df['State'] != 'CLOSED').sum()

duration_days = (sprint_df['End_dt'] - sprint_df['Start_dt']).dt.total_seconds() / (24*3600)
diff_complete_end = (sprint_df['Complete_dt'] - sprint_df['End_dt']).dt.total_seconds() / (24*3600)

closed_mask = sprint_df['State'] == 'CLOSED'
duration_mask = (duration_days >= 5.0) & (duration_days <= 35.0)
complete_mask = (diff_complete_end <= 3.0) & diff_complete_end.notna()

dropped_not_closed = (~closed_mask).sum()
dropped_duration = (closed_mask & ~duration_mask).sum()
dropped_late = (closed_mask & duration_mask & ~complete_mask).sum()

eligible_sprints = sprint_df[closed_mask & duration_mask & complete_mask].copy()
eligible_sprints['duration_days'] = duration_days[eligible_sprints.index]

print(f"Total sprints: {total_sprints}")
print(f"Dropped not CLOSED: {dropped_not_closed}")
print(f"Dropped duration not in [5, 35]: {dropped_duration}")
print(f"Dropped Complete_Date - End_Date > 3 days: {dropped_late}")
print(f"Eligible sprints: {len(eligible_sprints)}")

# Sort eligible sprints by End_dt
eligible_sprints = eligible_sprints.sort_values('End_dt').reset_index(drop=True)

# Global date cutoffs: D1 = 70th percentile, D2 = 85th percentile of End_dt
D1 = eligible_sprints['End_dt'].quantile(0.70)
D2 = eligible_sprints['End_dt'].quantile(0.85)

print(f"\nD1 (70th percentile End_Date): {D1}")
print(f"D2 (85th percentile End_Date): {D2}")

# Train: End <= D1
# Validation: Start >= D1 and End <= D2
# Test: Start >= D2
# Straddling:
# 1) Start < D1 and End > D1
# 2) Start < D2 and End > D2
train_mask = eligible_sprints['End_dt'] <= D1
val_mask = (eligible_sprints['Start_dt'] >= D1) & (eligible_sprints['End_dt'] <= D2)
test_mask = eligible_sprints['Start_dt'] >= D2

straddle_1 = (eligible_sprints['Start_dt'] < D1) & (eligible_sprints['End_dt'] > D1)
straddle_2 = (eligible_sprints['Start_dt'] < D2) & (eligible_sprints['End_dt'] > D2)
straddle_mask = straddle_1 | straddle_2

print(f"\nSplit counts (sprints):")
print(f"Train sprints: {train_mask.sum()}")
print(f"Validation sprints: {val_mask.sum()}")
print(f"Test sprints: {test_mask.sum()}")
print(f"Straddling sprints dropped: {straddle_mask.sum()} (D1: {straddle_1.sum()}, D2: {straddle_2.sum()})")
print(f"Sum accounted: {train_mask.sum() + val_mask.sum() + test_mask.sum() + straddle_mask.sum()} / {len(eligible_sprints)}")

# Check project distribution
proj_counts = eligible_sprints.groupby('Project_ID').size()
eligible_for_holdout = proj_counts[proj_counts >= 20].index.tolist()
print(f"\nProjects with >= 20 eligible sprints: {len(eligible_for_holdout)}")
print(f"Eligible project IDs: {eligible_for_holdout}")

np.random.seed(42)
held_out_5 = sorted(np.random.choice(eligible_for_holdout, size=5, replace=False).tolist())
print(f"Held out 5 projects (seed 42): {held_out_5}")
for pid in held_out_5:
    print(f"  Project {pid}: {proj_counts[pid]} sprints")

