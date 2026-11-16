import os
import time
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TAWOS_DIR = r"d:\Gillo AI Service\tawos_csv"
OUTPUT_DIR = r"d:\Gillo AI Service\benchmark_v2\work"
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "changelog_slim.parquet")
STATS_JSON = os.path.join(OUTPUT_DIR, "changelog_stream_stats.json")

TARGET_FIELDS = {
    "status",
    "Sprint",
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

def stream_and_filter():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    
    print("[1/3] Loading sprint.csv to identify closed sprints...")
    sprint_df = pd.read_csv(os.path.join(TAWOS_DIR, "sprint.csv"))
    sprint_df.columns = [c.lstrip('\ufeff') for c in sprint_df.columns]
    closed_sprints = sprint_df[sprint_df['State'].str.upper() == 'CLOSED']
    closed_sprint_ids = set(closed_sprints['ID'].dropna().astype(int).tolist())
    print(f"      Found {len(closed_sprint_ids)} closed sprints.")

    print("[2/3] Scanning issue.csv for issues with non-null Sprint_ID in closed sprints...")
    # Read issue.csv to get relevant Issue_IDs
    issue_iter = pd.read_csv(
        os.path.join(TAWOS_DIR, "issue.csv"),
        usecols=["ID", "Sprint_ID"],
        chunksize=100000,
        low_memory=False
    )
    valid_issue_ids = set()
    for chunk in issue_iter:
        chunk.columns = [c.lstrip('\ufeff') for c in chunk.columns]
        mask = chunk['Sprint_ID'].notna() & chunk['Sprint_ID'].isin(closed_sprint_ids)
        valid_ids = chunk.loc[mask, 'ID'].astype(int).tolist()
        valid_issue_ids.update(valid_ids)
    print(f"      Identified {len(valid_issue_ids)} target issues in closed sprints.")

    print(f"[3/3] Streaming change_log.csv in chunks of 500,000 rows...")
    cl_path = os.path.join(TAWOS_DIR, "change_log.csv")
    cl_chunks = pd.read_csv(cl_path, chunksize=500000, low_memory=False)

    writer = None
    total_scanned = 0
    total_retained = 0
    chunk_idx = 0

    for chunk in cl_chunks:
        chunk_idx += 1
        total_scanned += len(chunk)
        chunk.columns = [c.lstrip('\ufeff') for c in chunk.columns]
        
        # Ensure Issue_ID is clean
        chunk = chunk[chunk['Issue_ID'].notna()]
        chunk['Issue_ID'] = chunk['Issue_ID'].astype('int64')

        # Filter Issue_ID and Field
        mask = chunk['Issue_ID'].isin(valid_issue_ids) & chunk['Field'].isin(TARGET_FIELDS)
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

        if chunk_idx % 5 == 0 or total_scanned >= 21000000:
            elapsed = time.time() - t0
            print(f"      Processed {total_scanned:,} rows ({elapsed:.1f}s) -> Retained: {total_retained:,}")

    if writer is not None:
        writer.close()
    
    total_time = time.time() - t0
    file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)

    stats = {
        "total_rows_scanned": total_scanned,
        "rows_retained": total_retained,
        "valid_issues_count": len(valid_issue_ids),
        "target_fields": list(TARGET_FIELDS),
        "file_path": OUTPUT_PARQUET,
        "file_size_mb": round(file_size_mb, 2),
        "elapsed_seconds": round(total_time, 2)
    }

    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[DONE] Finished in {total_time:.2f}s!")
    print(f"       Scanned: {total_scanned:,} rows")
    print(f"       Retained: {total_retained:,} rows")
    print(f"       Saved to: {OUTPUT_PARQUET} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    stream_and_filter()
