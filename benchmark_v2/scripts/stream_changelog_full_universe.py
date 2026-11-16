"""
benchmark_v2/scripts/stream_changelog_full_universe.py

Streams tawos_csv/change_log.csv and builds changelog_full_universe.parquet
for all 37,909 issues that ever belonged to eligible closed sprints,
excluding the 382 status-less issues.
"""

import os
import sys
import time
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
work_dir = os.path.join(BASE_DIR, "work")
tawos_dir = r"d:\Gillo AI Service\tawos_csv"
OUTPUT_PARQUET = os.path.join(work_dir, "changelog_full_universe.parquet")

TARGET_FIELDS = {
    "status",
    "Sprint",
    "sprint",
    "Story Points",
    "Story points",
    "Actual Story Points",
    "assignee",
    "resolution",
    "Link"
}

SCHEMA = pa.schema([
    ("ID", pa.int64()),
    ("Field", pa.string()),
    ("From_Value", pa.string()),
    ("To_Value", pa.string()),
    ("From_String", pa.string()),
    ("To_String", pa.string()),
    ("Change_Type", pa.string()),
    ("Creation_Date", pa.string()),
    ("Author_ID", pa.string()),
    ("Issue_ID", pa.int64()),
])

def main():
    t0 = time.time()
    print("=== Streaming Changelog for Full Universe ===")

    # 1. Load eligible sprints
    print("Loading sprint.csv to identify eligible sprints...")
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
    eligible_sprint_ids = set(eligible_sprints['ID'].astype(int).tolist())
    sprint_map = {(int(r.Project_ID), int(r.JiraID)): int(r.ID) for r in sprint_df.itertuples() if pd.notna(r.Project_ID) and pd.notna(r.JiraID)}
    print(f"Eligible sprints: {len(eligible_sprint_ids)}")

    # 2. Load issues metadata to map (Project_ID) and find full universe
    print("Loading issue.csv...")
    issues_all = pd.read_csv(os.path.join(tawos_dir, "issue.csv"), usecols=['ID', 'Sprint_ID', 'Project_ID', 'Status'], low_memory=False)
    issues_all.columns = [c.lstrip('\ufeff') for c in issues_all.columns]
    issue_project_map = dict(zip(issues_all['ID'].astype(int), issues_all['Project_ID'].dropna().astype(int)))
    issue_sprint_map = dict(zip(issues_all['ID'].astype(int), issues_all['Sprint_ID']))

    # Load 382 status-less issues to exclude them completely (Item 2)
    cl_slim = pd.read_parquet(os.path.join(work_dir, "changelog_slim.parquet"))
    status_issues = set(cl_slim[cl_slim['Field'] == 'status']['Issue_ID'].unique())
    closed_sprint_all_ids = set(sprint_df[sprint_df['State'] == 'CLOSED']['ID'])
    closed_issues = issues_all[issues_all['Sprint_ID'].isin(closed_sprint_all_ids)]
    closed_issue_ids = set(closed_issues['ID'].unique())
    status_less_issue_ids = closed_issue_ids - status_issues
    print(f"Status-less issues to exclude completely: {len(status_less_issue_ids)}")

    # Load full universe issue IDs from selection bias check
    with open(os.path.join(work_dir, "selection_bias_check_metrics.json"), 'r') as f:
        sb_stats = json.load(f)

    # Re-verify full universe set from sprint_map and changelog
    # Stream changelog in 500k chunks
    cl_path = os.path.join(tawos_dir, "change_log.csv")
    cl_iter = pd.read_csv(cl_path, chunksize=500000, low_memory=False)

    # First pass was done in check_selection_bias.py; let's get the exact set
    # Load issues that appeared in Sprint changelog
    print("Collecting full universe issue IDs...")
    issue_eligible_sprints = set()
    for iid, sid_val in issue_sprint_map.items():
        if pd.notna(sid_val):
            try:
                sid_int = int(float(sid_val))
                if sid_int in eligible_sprint_ids:
                    issue_eligible_sprints.add(iid)
            except (ValueError, TypeError):
                pass

    # We also have the full universe issues from check_selection_bias:
    # Let's stream and collect/write directly
    print("Streaming change_log.csv to write changelog_full_universe.parquet...")
    # First, quick scan of Sprint rows to get full set of issue IDs
    full_target_issue_ids = set(issue_eligible_sprints)
    
    # We can do this in one streaming pass!
    # If we read chunks, any row for an issue in full_target_issue_ids gets written;
    # AND any Sprint row that maps to eligible_sprints adds the issue to full_target_issue_ids!
    # But an issue's status change might appear before its Sprint change in the file.
    # So pass 1: collect all Issue_IDs that belong to eligible sprints.
    # Pass 2: filter and write.
    print("Pass 1: Identifying all Issue_IDs belonging to eligible sprints...")
    cl_iter1 = pd.read_csv(cl_path, chunksize=500000, low_memory=False)
    for chunk in cl_iter1:
        chunk.columns = [c.lstrip('\ufeff') for c in chunk.columns]
        sp_chunk = chunk[chunk['Field'].isin({'sprint', 'Sprint'})]
        for r in sp_chunk.itertuples():
            if pd.isna(r.Issue_ID):
                continue
            iid = int(r.Issue_ID)
            pid = issue_project_map.get(iid)
            if pid is None:
                continue
            for val in [r.From_Value, r.To_Value]:
                if pd.notna(val) and str(val).strip():
                    for part in str(val).split(','):
                        part = part.strip()
                        if part.isdigit():
                            jid = int(part)
                            sid = sprint_map.get((pid, jid))
                            if sid is not None and sid in eligible_sprint_ids:
                                full_target_issue_ids.add(iid)

    # Exclude status-less issues (Item 2)
    full_target_issue_ids -= status_less_issue_ids
    print(f"Target issues for full universe changelog (excluding status-less): {len(full_target_issue_ids):,}")

    # Pass 2: Stream and write changelog_full_universe.parquet
    print("Pass 2: Filtering and writing changelog_full_universe.parquet...")
    cl_iter2 = pd.read_csv(cl_path, chunksize=500000, low_memory=False)
    writer = None
    total_retained = 0
    total_scanned = 0

    for idx, chunk in enumerate(cl_iter2):
        total_scanned += len(chunk)
        chunk.columns = [c.lstrip('\ufeff') for c in chunk.columns]
        chunk = chunk[chunk['Issue_ID'].notna()]
        chunk['Issue_ID'] = chunk['Issue_ID'].astype('int64')

        mask = chunk['Issue_ID'].isin(full_target_issue_ids) & chunk['Field'].isin(TARGET_FIELDS)
        filtered = chunk[mask].copy()

        if len(filtered) > 0:
            filtered['ID'] = filtered['ID'].astype('int64')
            for col in ['Field', 'From_Value', 'To_Value', 'From_String', 'To_String', 'Change_Type', 'Creation_Date', 'Author_ID']:
                filtered[col] = filtered[col].fillna('').astype(str)

            total_retained += len(filtered)
            table = pa.Table.from_pandas(filtered, schema=SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PARQUET, SCHEMA, compression='snappy')
            writer.write_table(table)

        if (idx + 1) % 5 == 0:
            print(f"  Processed {total_scanned:,} rows ({time.time()-t0:.1f}s) -> Retained {total_retained:,} rows...")

    if writer is not None:
        writer.close()

    print(f"\nSuccessfully wrote {total_retained:,} rows to {OUTPUT_PARQUET}")
    print(f"File size: {os.path.getsize(OUTPUT_PARQUET):,} bytes. Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

