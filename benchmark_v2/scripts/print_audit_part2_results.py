import os
import json
import pandas as pd

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
JSON_PATH = os.path.join(BASE_DIR, "work", "model1_part2_metrics.json")

def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    print("=" * 90)
    print("AUDIT PART 2: SUMMARY OF ALL BENCHMARK METRICS (FROM model1_part2_metrics.json)")
    print("=" * 90)

    # 1. Seed Noise
    print("\n" + "-" * 90)
    print("1. SEED NOISE ANALYSIS (5 RANDOM SEEDS: 42, 100, 200, 300, 400)")
    print("-" * 90)
    print(f"{'Model Architecture':45s} | {'Test MAE (Mean ± Std)':>23s} | {'Test MdAE (Mean ± Std)':>23s} | {'Test RMSE (Mean ± Std)':>23s}")
    print("-" * 120)
    for k in ['deployment_model', 'type_priority_only', 'sbert_lgbm']:
        m = data['item1_seed_noise'][k]
        name = m['display_name']
        mae_s = f"{m['MAE']['mean']:.4f} ± {m['MAE']['std']:.4f}"
        mdae_s = f"{m['MdAE']['mean']:.4f} ± {m['MdAE']['std']:.4f}"
        rmse_s = f"{m['RMSE']['mean']:.4f} ± {m['RMSE']['std']:.4f}"
        print(f"{name:45s} | {mae_s:>23s} | {mdae_s:>23s} | {rmse_s:>23s}")
        individual_maes = [round(x['MAE'], 4) for x in m['per_seed_runs']]
        print(f"   -> Individual Seed MAEs: {individual_maes}")

    # 2. Paired Bootstrap
    print("\n" + "-" * 90)
    print("2. PAIRED BOOTSTRAP (1,000 DRAWS, SEED 42, TICKETS & PROJECTS)")
    print("-" * 90)
    b = data['item2_paired_bootstrap']
    for comp_key, comp_title in [
        ('text_vs_type_priority', '(a) MAE(text + Type + Priority) - MAE(Type + Priority only)'),
        ('sbert_vs_type_priority', '(b) MAE(S-BERT + Type + Priority) - MAE(Type + Priority only)')
    ]:
        print(f"\n{comp_title}:")
        t = b[comp_key]['ticket_level']
        p = b[comp_key]['project_level']
        print(f"  Ticket Resampling:  Mean Diff = {t['mean_difference']:+.4f} | 95% CI = [{t['ci_95'][0]:.4f}, {t['ci_95'][1]:.4f}] | Excludes 0: {t['excludes_zero']}")
        print(f"  Project Resampling: Mean Diff = {p['mean_difference']:+.4f} | 95% CI = [{p['ci_95'][0]:.4f}, {p['ci_95'][1]:.4f}] | Excludes 0: {p['excludes_zero']}")
    
    print("\n" + b['plain_statement'])

    # 3. Taskove-Compatible Variants
    print("\n" + "-" * 90)
    print("3. TASKOVE-COMPATIBLE VARIANTS (NO PRIORITY AVAILABLE)")
    print("-" * 90)
    task_eval = data['item3_taskove_variants']['models_evaluation']
    print(f"{'Model Variant':35s} | {'MAE':>7s} | {'MdAE':>7s} | {'RMSE':>7s} | {'SA':>7s} | {'Snapped':>7s} | {'vs Type MAE':>11s} | {'Rel Imp %':>9s}")
    print("-" * 105)
    for m_name, v in task_eval.items():
        print(f"{m_name:35s} | {v['MAE']:7.4f} | {v['MdAE']:7.4f} | {v['RMSE']:7.4f} | {v['Standardized_Accuracy']:7.4f} | {v['Snapped_MAE']:7.4f} | {v['diff_vs_per_type_median']:+11.4f} | {v['rel_improvement_vs_type_pct']:8.2f}%")

    print("\nDistinct Priority Values per Top-10 Test Project (Do they differ by project?):")
    for p in data['item3_taskove_variants']['top10_projects_priority_vocabulary']:
        prios = p['distinct_priorities_in_clean']
        p_str = str(prios) if prios else "[] (100% NaN/Missing)"
        print(f"  Project {p['project_id']:2d} (Test N={p['test_ticket_count']:4d}, Clean N={p['total_clean_ticket_count']:4d}): Priorities = {p_str}")

    print("\nDistinct Type values in TAWOS (Raw N=458,232 | Clean N=42,724):")
    raw_types = data['item3_taskove_variants']['tawos_raw_type_counts']
    clean_types = data['item3_taskove_variants']['tawos_clean_type_counts']
    all_type_keys = sorted(list(raw_types.keys()), key=lambda x: raw_types[x], reverse=True)
    print(f"{'Issue Type':32s} | {'Raw TAWOS Count':>16s} | {'Clean TAWOS Count':>18s}")
    print("-" * 72)
    for t_name in all_type_keys:
        r_c = raw_types.get(t_name, 0)
        c_c = clean_types.get(t_name, 0)
        print(f"{str(t_name):32s} | {r_c:16d} | {c_c:18d}")

    # 4. Population Definition (Epic-Excluded)
    print("\n" + "-" * 90)
    print("4. POPULATION DEFINITION (TYPE == 'EPIC' EXCLUDED)")
    print("-" * 90)
    drop = data['item4_epic_excluded_population']['dropped_counts']
    print(f"Counts dropped per split:")
    print(f"  Train Split: dropped {drop['train_dropped_epics']} Epics (kept {drop['train_remaining']:,})")
    print(f"  Val Split:   dropped {drop['val_dropped_epics']} Epics (kept {drop['val_remaining']:,})")
    print(f"  Test Split:  dropped {drop['test_dropped_epics']} Epics (kept {drop['test_remaining']:,})")
    print(f"  Total Clean: dropped {drop['total_dropped_epics']} Epics")

    print("\nMain Table on Epic-Excluded Population (Test N=6,167):")
    epic_table = data['item4_epic_excluded_population']['models_evaluation']
    print(f"{'Model Architecture':36s} | {'MAE':>7s} | {'MdAE':>7s} | {'RMSE':>7s} | {'SA':>7s} | {'Snapped':>7s} | {'vs Type MAE':>11s} | {'Rel Imp %':>9s}")
    print("-" * 110)
    for m_name, v in epic_table.items():
        print(f"{m_name:36s} | {v['MAE']:7.4f} | {v['MdAE']:7.4f} | {v['RMSE']:7.4f} | {v['Standardized_Accuracy']:7.4f} | {v['Snapped_MAE']:7.4f} | {v['diff_vs_per_type_median']:+11.4f} | {v['rel_improvement_vs_type_pct']:8.2f}%")

    p28 = data['item4_epic_excluded_population']['project_28_error_share']
    print(f"\nProject 28 Share of Error on Epic-Excluded Population:")
    print(f"  Project 28 Test Tickets: {p28['test_tickets_p28']} / {p28['total_test_tickets']} ({p28['ticket_share_pct']}%)")
    print(f"  Project 28 Absolute Error Sum: {p28['total_abs_error_p28']} / {p28['total_abs_error_all']}")
    print(f"  Project 28 Share of Total Absolute Error: {p28['error_share_pct']}%")

    print("\n10 Largest Absolute Errors on Epic-Excluded Population:")
    print(f"{'Issue ID':>10s} | {'Proj':>5s} | {'Type':>15s} | {'True SP':>8s} | {'Pred SP':>8s} | {'Abs Err':>8s} | {'Signed Err':>10s}")
    print("-" * 75)
    for r in data['item4_epic_excluded_population']['top10_largest_errors']:
        print(f"{r['Issue_ID']:10d} | {r['Project_ID']:5d} | {r['Type']:15s} | {r['True_SP']:8.2f} | {r['Pred_SP']:8.2f} | {r['Absolute_Error']:8.2f} | {r['Signed_Error']:+10.2f}")

    # 5. Intervals
    print("\n" + "-" * 90)
    print("5. INTERVALS ([P25, P75] COVERAGE & MEAN WIDTH)")
    print("-" * 90)
    print("Top 10 Test Projects - Full Population (N=6,409):")
    print(f"{'Proj ID':>8s} | {'Tickets':>8s} | {'Raw Cov %':>10s} | {'Raw Width':>10s} | {'Adj Cov %':>10s} | {'Adj Width':>10s}")
    print("-" * 65)
    for r in data['item5_intervals']['top10_test_projects_full_population']:
        print(f"{r['project_id']:8d} | {r['test_tickets']:8d} | {r['raw_coverage_pct']:9.2f}% | {r['raw_mean_width']:10.4f} | {r['conformal_coverage_pct']:9.2f}% | {r['conformal_mean_width']:10.4f}")

    print("\nEpic-Excluded Overall Population (N=6,167):")
    ne_tot = data['item5_intervals']['epic_excluded_overall']
    print(f"  Raw Central [P25, P75]:       Coverage = {ne_tot['raw_coverage_pct']:.2f}% | Mean Width = {ne_tot['raw_mean_width']:.4f} story points")
    print(f"  Conformal Calibrated Interval: Coverage = {ne_tot['conformal_coverage_pct']:.2f}% | Mean Width = {ne_tot['conformal_mean_width']:.4f} story points (c = {ne_tot['adjustment_constant']:.4f})")

    print("\nTop 10 Test Projects - Epic-Excluded Population (N=6,167):")
    print(f"{'Proj ID':>8s} | {'Tickets':>8s} | {'Raw Cov %':>10s} | {'Raw Width':>10s} | {'Adj Cov %':>10s} | {'Adj Width':>10s}")
    print("-" * 65)
    for r in data['item5_intervals']['top10_test_projects_epic_excluded']:
        print(f"{r['project_id']:8d} | {r['test_tickets']:8d} | {r['raw_coverage_pct']:9.2f}% | {r['raw_mean_width']:10.4f} | {r['conformal_coverage_pct']:9.2f}% | {r['conformal_mean_width']:10.4f}")

    # 6. Explanation Fix
    print("\n" + "-" * 90)
    print("6. FIX THE EXPLANATION: MODEL PREDICTION DISTRIBUTION FOR EPICS")
    print("-" * 90)
    ed = data['item6_epic_prediction_distribution']
    print(f"Test Epics Count: {ed['count']}")
    print(f"True SP: Mean = {ed['true_sp']['mean']:.2f} | Median = {ed['true_sp']['median']:.2f} | Std = {ed['true_sp']['std']:.2f} | Min = {ed['true_sp']['min']} | Max = {ed['true_sp']['max']} | P10 = {ed['true_sp']['p10']} | P25 = {ed['true_sp']['p25']} | P75 = {ed['true_sp']['p75']} | P90 = {ed['true_sp']['p90']} | P95 = {ed['true_sp']['p95']} | P99 = {ed['true_sp']['p99']}")
    print(f"Pred SP: Mean = {ed['pred_sp']['mean']:.2f} | Median = {ed['pred_sp']['median']:.2f} | Std = {ed['pred_sp']['std']:.2f} | Min = {ed['pred_sp']['min']} | Max = {ed['pred_sp']['max']} | P10 = {ed['pred_sp']['p10']} | P25 = {ed['pred_sp']['p25']} | P75 = {ed['pred_sp']['p75']} | P90 = {ed['pred_sp']['p90']} | P95 = {ed['pred_sp']['p95']} | P99 = {ed['pred_sp']['p99']}")

    # 7. Metadata Verification
    print("\n" + "-" * 90)
    print("7. METADATA.JSON VERIFICATION")
    print("-" * 90)
    m = data['item7_metadata']
    print(f"trained_on: '{m['trained_on']}'")
    print(f"clean_unfiltered_info: {m['clean_unfiltered_info']}")
    print(f"encoder_known_types ({len(m['encoder_known_types'])} types): {m['encoder_known_types']}")

if __name__ == "__main__":
    main()

