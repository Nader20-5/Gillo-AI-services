import os
import json
import pandas as pd
import numpy as np

TAWOS_DIR = r"d:\Gillo AI Service\tawos_csv"
WORK_DIR = r"d:\Gillo AI Service\benchmark_v2\work"
PARQUET_PATH = os.path.join(WORK_DIR, "changelog_slim.parquet")
OUTPUT_JSON = os.path.join(WORK_DIR, "discovery_2_metrics.json")

def analyze():
    print("Loading changelog_slim.parquet...")
    cl_df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(cl_df):,} rows from changelog_slim.parquet.")

    print("Loading sprint.csv...")
    sprint_df = pd.read_csv(os.path.join(TAWOS_DIR, "sprint.csv"))
    sprint_df.columns = [c.lstrip('\ufeff') for c in sprint_df.columns]
    closed_sprints = sprint_df[sprint_df['State'].str.upper() == 'CLOSED']
    closed_sprint_ids = set(closed_sprints['ID'].dropna().astype(int).tolist())

    print("Loading issue.csv...")
    issue_df = pd.read_csv(
        os.path.join(TAWOS_DIR, "issue.csv"),
        low_memory=False
    )
    issue_df.columns = [c.lstrip('\ufeff') for c in issue_df.columns]
    
    # Filter issues in closed sprints with non-null Sprint_ID
    closed_issues = issue_df[issue_df['Sprint_ID'].notna() & issue_df['Sprint_ID'].isin(closed_sprint_ids)].copy()
    n_closed_issues = len(closed_issues)
    print(f"Total target issues in closed sprints: {n_closed_issues}")

    results = {}

    # -------------------------------------------------------------
    # 1. Format of Sprint change rows
    # -------------------------------------------------------------
    print("\n--- Question 1: Format of Sprint change rows ---")
    sprint_changes = cl_df[cl_df['Field'] == 'Sprint'].copy()
    q1_samples = sprint_changes[['Issue_ID', 'From_Value', 'To_Value', 'From_String', 'To_String', 'Creation_Date']].head(10).to_dict(orient="records")
    
    # Inspect multiple sprint IDs or names in single cells
    has_multiple_from = sprint_changes['From_String'].dropna().str.contains(',').sum()
    has_multiple_to = sprint_changes['To_String'].dropna().str.contains(',').sum()
    
    results["q1_sprint_format"] = {
        "total_sprint_change_rows": len(sprint_changes),
        "from_string_multiple_count": int(has_multiple_from),
        "to_string_multiple_count": int(has_multiple_to),
        "sample_rows": q1_samples
    }

    # -------------------------------------------------------------
    # 2. Issues in closed sprints with Sprint change rows
    # -------------------------------------------------------------
    print("\n--- Question 2: Sprint membership coverage ---")
    issues_with_sprint_change = set(sprint_changes['Issue_ID'].unique())
    target_issue_ids = set(closed_issues['ID'].unique())
    overlap = target_issue_ids.intersection(issues_with_sprint_change)
    reconstructable_fraction = len(overlap) / n_closed_issues

    results["q2_sprint_coverage"] = {
        "total_closed_sprint_issues": n_closed_issues,
        "issues_with_sprint_change_rows": len(overlap),
        "issues_without_sprint_change_rows": n_closed_issues - len(overlap),
        "fraction_with_change_history": round(reconstructable_fraction, 4),
        "only_issue_sprint_id_evidence": n_closed_issues - len(overlap)
    }

    # -------------------------------------------------------------
    # 3. Multi-sprint membership during life vs issue.Sprint_ID
    # -------------------------------------------------------------
    print("\n--- Question 3: Multi-sprint movement ---")
    # For each issue, check how many distinct sprint names/IDs appear in change_log + issue.Sprint_ID
    issue_sprint_counts = {}
    for issue_id, group in sprint_changes.groupby('Issue_ID'):
        sprints_seen = set()
        for val in group['From_String'].dropna():
            for s in str(val).split(','):
                s_clean = s.strip()
                if s_clean and s_clean != 'None':
                    sprints_seen.add(s_clean)
        for val in group['To_String'].dropna():
            for s in str(val).split(','):
                s_clean = s.strip()
                if s_clean and s_clean != 'None':
                    sprints_seen.add(s_clean)
        issue_sprint_counts[issue_id] = len(sprints_seen)

    multi_sprint_issues = sum(1 for cnt in issue_sprint_counts.values() if cnt > 1)
    results["q3_multi_sprint"] = {
        "issues_with_multi_sprints_in_history": multi_sprint_issues,
        "total_issues_analyzed": len(issue_sprint_counts),
        "percentage_of_issues_with_sprint_changes": round(multi_sprint_issues / max(len(issue_sprint_counts), 1) * 100, 2),
        "issue_sprint_id_column_limitation": "issue.csv only stores a single scalar Sprint_ID representing the final sprint at export time."
    }

    # -------------------------------------------------------------
    # 4. Status reconstruction & Top 30 status names
    # -------------------------------------------------------------
    print("\n--- Question 4: Status reconstruction ---")
    status_changes = cl_df[cl_df['Field'] == 'status'].copy()
    
    # Collect all status names from From_String and To_String
    status_counter = pd.concat([status_changes['From_String'], status_changes['To_String']]).value_counts()
    top_30_statuses = status_counter.head(30).to_dict()

    # Can initial status at time t be derived?
    # For issues with status change, the first change's From_String is the initial status.
    issues_with_status_changes = set(status_changes['Issue_ID'].unique())

    # Proposed mapping
    todo_set = {"Open", "To Do", "Reopened", "Backlog", "Created", "New", "Submitted", "Unconfirmed", "Needs Triage", "Untriaged"}
    in_prog_set = {"In Progress", "In Review", "Review", "Under Review", "In Development", "Blocked", "In Testing", "Testing", "Code Review", "Patch Available"}
    done_set = {"Closed", "Resolved", "Done", "Complete", "Completed", "Verified", "Delivered"}

    results["q4_status_reconstruction"] = {
        "total_status_change_rows": len(status_changes),
        "issues_with_status_history": len(issues_with_status_changes),
        "top_30_statuses": {str(k): int(v) for k, v in top_30_statuses.items()},
        "proposed_mapping": {
            "todo": list(todo_set),
            "in_progress": list(in_prog_set),
            "done": list(done_set)
        }
    }

    # -------------------------------------------------------------
    # 5. Resolution_Date agreement with change to 'done' status
    # -------------------------------------------------------------
    print("\n--- Question 5: Resolution_Date agreement ---")
    # Join issue.Resolution_Date with status changes where To_String is in done_set or Field == 'resolution'
    # Find the earliest transition to Done/Resolved
    done_changes = status_changes[status_changes['To_String'].isin(["Closed", "Resolved", "Done", "Completed"])].copy()
    done_changes['Creation_Date_dt'] = pd.to_datetime(done_changes['Creation_Date'], errors='coerce')
    
    # Also check Field == 'resolution'
    res_changes = cl_df[cl_df['Field'] == 'resolution'].copy()
    res_changes['Creation_Date_dt'] = pd.to_datetime(res_changes['Creation_Date'], errors='coerce')

    # Earliest done status change per issue
    earliest_done = done_changes.groupby('Issue_ID')['Creation_Date_dt'].min().reset_index()
    earliest_done.rename(columns={'Creation_Date_dt': 'Status_Done_Date'}, inplace=True)

    # Merge with closed_issues
    merged_res = closed_issues[['ID', 'Resolution_Date']].dropna(subset=['Resolution_Date']).copy()
    merged_res['Resolution_Date_dt'] = pd.to_datetime(merged_res['Resolution_Date'], errors='coerce')
    comp_df = pd.merge(merged_res, earliest_done, left_on='ID', right_on='Issue_ID', how='inner')
    
    diff_hours = (comp_df['Resolution_Date_dt'] - comp_df['Status_Done_Date']).dt.total_seconds().abs() / 3600.0
    disagree_over_1h = (diff_hours > 1.0).sum()
    disagree_over_24h = (diff_hours > 24.0).sum()
    total_comp = len(comp_df)

    results["q5_resolution_date_agreement"] = {
        "issues_compared": total_comp,
        "disagreement_gt_1h": int(disagree_over_1h),
        "disagreement_rate_gt_1h": round(disagree_over_1h / max(total_comp, 1), 4),
        "disagreement_gt_24h": int(disagree_over_24h),
        "disagreement_rate_gt_24h": round(disagree_over_24h / max(total_comp, 1), 4),
        "median_diff_hours": round(float(diff_hours.median()), 2),
        "p95_diff_hours": round(float(diff_hours.quantile(0.95)), 2)
    }

    # -------------------------------------------------------------
    # 6. Link rows in change log
    # -------------------------------------------------------------
    print("\n--- Question 6: Link rows ---")
    link_changes = cl_df[cl_df['Field'] == 'Link'].copy()
    link_samples = link_changes[['Issue_ID', 'From_String', 'To_String', 'Creation_Date']].head(10).to_dict(orient="records")
    top_link_types = link_changes['To_String'].value_counts().head(10).to_dict()

    results["q6_link_rows"] = {
        "total_link_changes": len(link_changes),
        "sample_rows": link_samples,
        "top_to_strings": {str(k): int(v) for k, v in top_link_types.items()},
        "can_count_links_before_t": True,
        "explanation": "Yes, unlike issue_link.csv which has NO timestamps, change_log.csv records each link addition with an exact Creation_Date timestamp."
    }

    # -------------------------------------------------------------
    # 7. sprint.csv Activated_Date and Complete_Date coverage & diff
    # -------------------------------------------------------------
    print("\n--- Question 7: Sprint date coverage & diff ---")
    total_sprints = len(sprint_df)
    act_cov = sprint_df['Activated_Date'].notna().sum()
    comp_cov = sprint_df['Complete_Date'].notna().sum()
    
    # Difference (Complete_Date - End_Date) in days
    sprint_df['Complete_Date_dt'] = pd.to_datetime(sprint_df['Complete_Date'], errors='coerce')
    sprint_df['End_Date_dt'] = pd.to_datetime(sprint_df['End_Date'], errors='coerce')
    
    valid_diff = sprint_df.dropna(subset=['Complete_Date_dt', 'End_Date_dt']).copy()
    diff_days = (valid_diff['Complete_Date_dt'] - valid_diff['End_Date_dt']).dt.total_seconds() / 86400.0

    diff_quantiles = {
        "P5": round(float(diff_days.quantile(0.05)), 2),
        "P25": round(float(diff_days.quantile(0.25)), 2),
        "P50": round(float(diff_days.median()), 2),
        "P75": round(float(diff_days.quantile(0.75)), 2),
        "P95": round(float(diff_days.quantile(0.95)), 2),
        "mean": round(float(diff_days.mean()), 2),
        "std": round(float(diff_days.std()), 2)
    }

    results["q7_sprint_dates"] = {
        "total_sprints": total_sprints,
        "activated_date_coverage": int(act_cov),
        "activated_date_coverage_pct": round(act_cov / total_sprints * 100, 2),
        "complete_date_coverage": int(comp_cov),
        "complete_date_coverage_pct": round(comp_cov / total_sprints * 100, 2),
        "complete_minus_end_days_quantiles": diff_quantiles
    }

    # -------------------------------------------------------------
    # 8. Effort columns coverage & correlation with Story_Point
    # -------------------------------------------------------------
    print("\n--- Question 8: Effort coverage & correlation ---")
    effort_cols = ['In_Progress_Minutes', 'Timespent', 'Total_Effort_Minutes']
    effort_stats = {}
    
    for ec in effort_cols:
        col_series = closed_issues[ec].dropna()
        gt0_series = col_series[col_series > 0]
        effort_stats[ec] = {
            "total_notna": len(col_series),
            "total_gt0": len(gt0_series),
            "coverage_pct": round(len(gt0_series) / n_closed_issues * 100, 2),
            "median": round(float(gt0_series.median()), 2) if len(gt0_series) > 0 else None,
            "p95": round(float(gt0_series.quantile(0.95)), 2) if len(gt0_series) > 0 else None
        }

    # Correlations for top 5 projects by closed issues with SP
    top5_pids = [28, 12, 36, 4, 1]
    top5_corr = {}
    for pid in top5_pids:
        p_df = closed_issues[closed_issues['Project_ID'] == pid].copy()
        p_valid = p_df[(p_df['Story_Point'] > 0) & (p_df['Story_Point'] <= 100)].copy()
        
        corr_dict = {}
        for ec in effort_cols:
            subset = p_valid[p_valid[ec] > 0]
            if len(subset) >= 10:
                corr = subset['Story_Point'].corr(subset[ec])
                corr_dict[ec] = {
                    "spearman_or_pearson_corr": round(float(corr), 4),
                    "n_samples": len(subset)
                }
            else:
                corr_dict[ec] = {"spearman_or_pearson_corr": None, "n_samples": len(subset)}
        top5_corr[str(pid)] = corr_dict

    results["q8_effort_metrics"] = {
        "effort_coverage_and_quantiles": effort_stats,
        "top_5_projects_correlation_with_story_points": top5_corr
    }

    # Save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Analysis saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    analyze()
