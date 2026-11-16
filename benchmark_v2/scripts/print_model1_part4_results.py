import os
import json

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
WORK_DIR = os.path.join(BASE_DIR, "work")

def main():
    path = os.path.join(WORK_DIR, "model1_part4_metrics.json")
    with open(path, "r") as f:
        data = json.load(f)

    print("=" * 105)
    print("AUDIT PART 4: SUMMARY OF ALL NEW BENCHMARK METRICS (FROM model1_part4_metrics.json)")
    print("=" * 105)

    # 1. Taskove Types Audit
    print("\n" + "-" * 105)
    print("1. TASKOVE TYPES CODEBASE AUDIT & MAPPING TABLE")
    print("-" * 105)
    audit = data['item1_taskove_types_audit']
    print(f"Verified Distinct Real Taskove Types in Codebase ({len(audit['distinct_real_taskove_types'])}): {audit['distinct_real_taskove_types']}")
    print(f"Invented Types Removed: {audit['invented_types_in_previous_table']}")
    print(f"Omitted Real Types Added: {audit['omitted_real_types_in_previous_table']}")
    print("\nUpdated Taskove to TAWOS Type Mapping Table:")
    print(f"{'Taskove Type':<15} | {'TAWOS Target Type':<20} | {'Fidelity':<12}")
    print("-" * 52)
    for k, v in audit['updated_taskove_type_mapping'].items():
        print(f"{k:<15} | {v['tawos_type']:<20} | {v['fidelity']:<12}")

    # 2. Type-Mapping Sensitivity
    print("\n" + "-" * 105)
    print("2. TYPE-MAPPING SENSITIVITY FOR FINAL MODEL (TF-IDF + TYPE, NO PRIORITY, EPIC-EXCLUDED)")
    print("-" * 105)
    sens = data['item2_type_mapping_sensitivity']
    print(f"{'Variant / Perturbation':<45} | {'Inf Test MAE':<12} | {'Retr Test MAE':<13} | {'Cold-Start Mean MAE (± Std)':<28}")
    print("-" * 105)
    names = {
        'a_current': '(a) Current Mapping (TAWOS unchanged)',
        'b_tech_to_task': '(b) Tech Task mapped to Task',
        'c_bug_vs_nonbug': '(c) Collapsed to {Bug, non-Bug}',
        'd_all_task': '(d) All types set to Task'
    }
    for k, name in names.items():
        v = sens[k]
        cs_str = f"{v['cold_start_mean_MAE']:.4f} ± {v['cold_start_std_MAE']:.4f}"
        print(f"{name:<45} | {v['test_MAE_inference_pretrained']:<12.4f} | {v['test_MAE_retrained']:<13.4f} | {cs_str:<28}")

    print("\nCold-Start Fold Breakdown across Variants:")
    print(f"{'Variant':<45} | {'Fold 1':<8} | {'Fold 2':<8} | {'Fold 3':<8} | {'Fold 4':<8} | {'Fold 5':<8}")
    print("-" * 92)
    for k, name in names.items():
        folds = sens[k]['cold_start_fold_maes']
        print(f"{name:<45} | {folds[0]:<8.4f} | {folds[1]:<8.4f} | {folds[2]:<8.4f} | {folds[3]:<8.4f} | {folds[4]:<8.4f}")

    # 3. Practical Metrics
    print("\n" + "-" * 105)
    print("3. PRACTICAL METRICS: SNAPPED ERROR, ACCURACY WITHIN 1 POINT, AND COLD-START WINS")
    print("-" * 105)
    prac = data['item3_practical_metrics']['chronological_test_comparison']
    f_mod = prac['final_model']
    base = prac['per_type_median']
    print(f"{'Metric':<35} | {'Final Model (v2_taskove)':<25} | {'Baseline (Per-Type Median)':<25}")
    print("-" * 90)
    print(f"{'Continuous MAE':<35} | {f_mod['raw_MAE']:<25.4f} | {base['raw_MAE']:<25.4f}")
    print(f"{'Continuous MdAE':<35} | {f_mod['raw_MdAE']:<25.4f} | {base['raw_MdAE']:<25.4f}")
    print(f"{'Snapped MAE (Fibonacci)':<35} | {f_mod['snapped_MAE']:<25.4f} | {base['snapped_MAE']:<25.4f}")
    print(f"{'Snapped MdAE (Fibonacci)':<35} | {f_mod['snapped_MdAE']:<25.4f} | {base['snapped_MdAE']:<25.4f}")
    print(f"{'Fraction with |error| <= 1.0 (Raw)':<35} | {f_mod['raw_fraction_abs_error_le_1']*100:<24.2f}% | {base['raw_fraction_abs_error_le_1']*100:<24.2f}%")
    print(f"{'Fraction with |error| <= 1.0 (Snapped)':<35} | {f_mod['snapped_fraction_abs_error_le_1']*100:<24.2f}% | {base['snapped_fraction_abs_error_le_1']*100:<24.2f}%")

    wins = data['item3_practical_metrics']['cold_start_win_counts']
    print(f"\nCold-Start Win Counts:")
    print(f"  Per-Fold Wins:    {wins['fold_wins']}")
    print(f"  Per-Project Wins: {wins['project_wins']} (Losses: {wins['project_losses']}, Ties: {wins['project_ties']})")

    # 4. Interval Coverage & Width
    print("\n" + "-" * 105)
    print("4. CONFORMAL INTERVAL COVERAGE & MEAN WIDTH")
    print("-" * 105)
    print("Top 10 Test Projects (Chronological Test Set, N=6,167):")
    print(f"{'Project ID':<12} | {'Test Tickets':<14} | {'Conformal Coverage (%)':<24} | {'Mean Width (points)':<20}")
    print("-" * 75)
    for p in data['item4_interval_metrics']['top_10_test_projects']:
        print(f"{p['project_id']:<12} | {p['test_tickets']:<14} | {p['conformal_coverage_pct']:<24.2f} | {p['mean_interval_width']:<20.4f}")

    print("\nCold-Start 50% Interval Coverage across Folds:")
    print(f"{'Fold':<8} | {'Test Tickets':<14} | {'Retrained Cov (%)':<18} | {'Retrained Width':<16} | {'Zero-Shot Cov (%)':<18} | {'Zero-Shot Width':<16}")
    print("-" * 96)
    for f_info in data['item4_interval_metrics']['cold_start_folds_50pct_interval']:
        print(f"Fold {f_info['fold']:<3} | {f_info['test_tickets']:<14} | {f_info['retrained_coverage_pct']:<18.2f} | {f_info['retrained_mean_width']:<16.4f} | {f_info['v2_zero_shot_coverage_pct']:<18.2f} | {f_info['v2_zero_shot_mean_width']:<16.4f}")

if __name__ == "__main__":
    main()

