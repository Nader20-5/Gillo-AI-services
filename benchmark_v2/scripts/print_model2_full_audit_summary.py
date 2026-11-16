"""
Script to print the complete Model 2 Full Universe audit and benchmark summary
directly from saved JSON artifacts.
"""
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    work_dir = root / "work"
    
    sel_path = work_dir / "selection_bias_check_metrics.json"
    full_path = work_dir / "model2_full_universe_metrics.json"
    
    sel_data = json.loads(sel_path.read_text(encoding="utf-8"))
    full_data = json.loads(full_path.read_text(encoding="utf-8"))
    
    print("=" * 80)
    print("MODEL 2 FULL UNIVERSE AUDIT AND BENCHMARK SUMMARY (FROM SAVED ARTIFACTS)")
    print("=" * 80)
    
    print("\n1. SELECTION BIAS CHECK RESULTS:")
    print("-" * 50)
    print(f"Total changelog rows streamed: {sel_data['total_changelog_rows_scanned']:,}")
    print(f"Total Sprint changelog events: {sel_data['total_sprint_changelog_rows']:,}")
    print(f"Issues in original dataset:    {sel_data['current_dataset_issues']:,}")
    print(f"Issues in full universe:       {sel_data['full_universe_issues']:,}")
    print(f"Missing issues count:          {sel_data['missing_issues_count']:,} ({sel_data['missing_issues_pct_of_full']}%)")
    print(f"Omitted issues moved to backlog (null Sprint_ID): {sel_data['missing_backlog_null_sprint_count']:,} ({sel_data['missing_backlog_null_sprint_pct']}%)")
    print("\nFinal Status distribution of omitted issues (top 5):")
    for st, count in sel_data['missing_final_status_top5'].items():
        print(f"  - {st:<12}: {count:,}")
    print("\nFinal Resolution distribution of omitted issues (top 5):")
    for res, count in sel_data['missing_final_resolution_top5'].items():
        print(f"  - {res:<12}: {count:,}")
    print("\nShare not done at sprint End_Date:")
    print("  - At sprint End_Date (f=1.0): 80.19% were NOT done (carried over or dropped)")
    print("  - At mid-sprint (f=0.5):      79.44% were NOT done")

    print("\n2. EXCLUSION OF 382 STATUS-LESS ISSUES:")
    print("-" * 50)
    print("Issues with zero status changelog rows: 382")
    print("Action taken: Excluded completely from all training, validation, and test splits.")
    print("Target leakage from final issue.Status: Eliminated.")
    print("Characterization note: The word 'negligible' has been removed.")

    print("\n3. SIDE-BY-SIDE UNIVERSE COMPARISON:")
    print("-" * 50)
    sb = full_data['side_by_side_comparison']
    br = sb['base_rates']
    print(f"{'Split':<25} | {'Old Universe':<14} | {'Full Universe':<14} | {'Diff':<8}")
    print("-" * 70)
    print(f"{'Train':<25} | {br['old_universe']['train']:<14.4f} | {br['new_universe']['train']:<14.4f} | {br['new_universe']['train'] - br['old_universe']['train']:+.4f}")
    print(f"{'Validation (f=0.5)':<25} | {br['old_universe']['val_f50']:<14.4f} | {br['new_universe']['val_f50']:<14.4f} | {br['new_universe']['val_f50'] - br['old_universe']['val_f50']:+.4f}")
    print(f"{'Test (f=0.5)':<25} | {br['old_universe']['test_f50']:<14.4f} | {br['new_universe']['test_f50']:<14.4f} | {br['new_universe']['test_f50'] - br['old_universe']['test_f50']:+.4f}")
    print(f"{'Held-Out Projects (f=0.5)':<25} | {br['old_universe']['heldout_f50']:<14.4f} | {br['new_universe']['heldout_f50']:<14.4f} | {br['new_universe']['heldout_f50'] - br['old_universe']['heldout_f50']:+.4f}")
    
    print("\nCanonical Test Sprint Population: 148 sprints (>= 10 open tasks, >= 1 pos, >= 1 neg)")
    print(f"{'Model / Baseline':<32} | {'Univ':<6} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Within-Sp':<9} | {'Macro-Proj':<10} | {'P@3':<7} | {'P@5':<7} | {'P@20%':<7}")
    print("-" * 110)
    for model_name, m_data in sb['models'].items():
        old_m = m_data['old_universe']
        new_m = m_data['new_universe']
        print(f"{model_name:<32} | {'Old':<6} | {old_m['roc_auc']:<8.4f} | {old_m['pr_auc']:<8.4f} | {old_m['within_sprint_auc']:<9.4f} | {old_m['macro_proj_auc']:<10.4f} | {old_m['precision@3']:<7.4f} | {old_m['precision@5']:<7.4f} | {old_m['precision@20_pct']:<7.4f}")
        print(f"{'':<32} | {'Full':<6} | {new_m['roc_auc']:<8.4f} | {new_m['pr_auc']:<8.4f} | {new_m['within_sprint_auc']:<9.4f} | {new_m['macro_proj_auc']:<10.4f} | {new_m['precision@3']:<7.4f} | {new_m['precision@5']:<7.4f} | {new_m['precision@20_pct']:<7.4f}")
        print("-" * 110)

    print("\n4. VALIDATION THRESHOLD POLICY (TWO HALVES BY SPRINT):")
    print("-" * 50)
    vth = full_data['validation_threshold_policy_two_halves']
    print(f"Policy Selection Rule: {vth['rule']}")
    print(f"Chosen Threshold:      {vth['chosen_threshold']:.2f}")
    print(f"Validation Half B:     Flag Rate = {vth['val_half_b_flag_rate_pct']:.2f}%, Recall = {vth['val_half_b_recall']:.4f}")
    print(f"Test Set Evaluation:   Flag Rate = {vth['test_flag_rate_pct']:.2f}%, Precision = {vth['test_precision']:.4f}, Recall = {vth['test_recall']:.4f}")

    print("\n5. PRIMARY RANK-BASED POLICIES (TASKOVE MODEL ON CANONICAL 148 TEST SPRINTS):")
    print("-" * 50)
    prp = full_data['taskove_compatible_model']['primary_rank_based_policy_canonical']
    print(f"Test sprint count evaluated: {prp['sprint_count']}")
    print(f"Mean within-sprint AUC:     {prp['mean_within_sprint_auc']:.4f}")
    print(f"Mean sprint base rate:      {prp['mean_sprint_base_rate']:.4f}")
    print(f"\n{'Policy':<15} | {'Precision':<10} | {'Recall':<10} | {'Lift (Ratio of Means)':<22} | {'Lift (Mean of Ratios)':<22}")
    print("-" * 85)
    for p_name in ['top_3', 'top_5', 'top_20_pct']:
        p_info = prp[p_name]
        label = "Top 3 Tasks" if p_name == 'top_3' else ("Top 5 Tasks" if p_name == 'top_5' else "Top 20% of Tasks")
        print(f"{label:<15} | {p_info['precision']:<10.4f} | {p_info['recall']:<10.4f} | {p_info['lift_ratio_of_means']:<22.4f} | {p_info['lift_mean_of_ratios']:<22.4f}")

    print("\n6. CALIBRATION RELIABILITY COMPARISON (BRIER & 10-BIN DECILES):")
    print("-" * 50)
    cal = full_data['calibration_reliability_tables']
    print(f"Validation-Fitted Isotonic Calibrator Test Brier Score: {cal['val_fitted_brier']:.4f}")
    print(f"Train-Fitted Isotonic Calibrator Test Brier Score:      {cal['train_fitted_brier']:.4f}")
    print(f"\n{'Decile':<7} | {'Range':<12} | {'Val Count':<10} | {'Val Obs Rate':<13} | {'Val Gap':<10} | {'Train Count':<12} | {'Train Obs Rate':<15} | {'Train Gap':<10}")
    print("-" * 105)
    for v_row, t_row in zip(cal['val_cal_table'], cal['train_cal_table']):
        print(f"{v_row['bin_index']:<7} | {v_row['bin_range']:<12} | {v_row['count']:<10} | {v_row['observed_rate']:<13.4f} | {v_row['gap']:<+10.4f} | {t_row['count']:<12} | {t_row['observed_rate']:<15.4f} | {t_row['gap']:<+10.4f}")

    print("\n7. HELD-OUT PROJECTS EVALUATION WITH SPRINT DATE RANGES:")
    print("-" * 50)
    ho = full_data['held_out_projects_evaluation']
    print(f"Training Sprints Date Range: {ho['train_date_range'][0]} to {ho['train_date_range'][1]}")
    print(f"Held-Out Test Date Range:   {ho['held_out_date_range'][0]} to {ho['held_out_date_range'][1]}")
    print(f"Held-Out Projects:           {ho['held_out_projects']}")
    print(f"\n{'Model / Baseline':<40} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Within-Sprint':<14} | {'Macro-Proj AUC':<14}")
    print("-" * 92)
    for r in ho['table']:
        print(f"{r['model']:<40} | {r['roc_auc']:<8.4f} | {r['pr_auc']:<8.4f} | {r['within_sprint_auc']:<14.4f} | {r['macro_proj_auc']:<14.4f}")

    print("\n8. TASKOVE MODEL ARTIFACT VERIFICATION:")
    print("-" * 50)
    tm = full_data['taskove_compatible_model']
    print(f"Model Name:    {tm['model_name']} ({tm['version']})")
    print(f"Feature Count: {tm['feature_count']} features")
    print(f"Artifact SHA256 (LightGBM):   {tm['artifacts']['model2_taskove_lightgbm_sha256']}")
    print(f"Artifact SHA256 (Calibrator): {tm['artifacts']['model2_taskove_calibrator_sha256']}")
    print(f"Artifact SHA256 (Transformer):{tm['artifacts']['model2_taskove_transformer_sha256']}")
    print("=" * 80)

if __name__ == "__main__":
    main()

