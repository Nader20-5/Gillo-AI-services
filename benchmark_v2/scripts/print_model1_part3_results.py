import os
import json
import pandas as pd

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
JSON_PATH = os.path.join(BASE_DIR, "work", "model1_part3_metrics.json")
P2_JSON_PATH = os.path.join(BASE_DIR, "work", "model1_part2_metrics.json")

def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    with open(P2_JSON_PATH, "r") as f:
        data_p2 = json.load(f)

    print("=" * 105)
    print("AUDIT PART 3: SUMMARY OF ALL BENCHMARK METRICS (FROM model1_part3_metrics.json)")
    print("=" * 105)

    # 1. Final Candidates on Epic-Excluded Population
    print("\n" + "-" * 105)
    print("1. FINAL CANDIDATES (TASKOVE-COMPATIBLE, NO PRIORITY, EPIC-EXCLUDED POPULATION)")
    print("-" * 105)
    print(f"Scope decision note: {data['item1_final_candidates_epic_excluded']['scope_decision_note']}\n")
    
    cand_summary = data['item1_final_candidates_epic_excluded']['summary']
    print(f"{'Candidate Architecture':25s} | {'Test MAE':>8s} | {'Test MdAE':>9s} | {'Test RMSE':>9s} | {'5-Seed MAE (Mean ± Std)':>23s} | {'5-Seed MdAE':>15s} | {'5-Seed RMSE':>15s}")
    print("-" * 115)
    for k in ['type_only', 'tfidf_type', 'sbert_type']:
        c = cand_summary[k]
        name = c['display_name']
        s42 = c['seed_42_single_run']
        mae_str = f"{c['MAE']['mean']:.4f} ± {c['MAE']['std']:.4f}"
        mdae_str = f"{c['MdAE']['mean']:.4f} ± {c['MdAE']['std']:.4f}"
        rmse_str = f"{c['RMSE']['mean']:.4f} ± {c['RMSE']['std']:.4f}"
        print(f"{name:25s} | {s42['MAE']:8.4f} | {s42['MdAE']:9.4f} | {s42['RMSE']:9.4f} | {mae_str:>23s} | {mdae_str:>15s} | {rmse_str:>15s}")
        print(f"   -> Individual Seed MAEs ({c['seeds']}): {[round(x['MAE'], 4) for x in c['per_seed_runs']]}")

    print("\nFull Population vs Epic-Excluded Population (Side-by-Side Comparison):")
    print(f"{'Population':22s} | {'Candidate / Baseline':30s} | {'Test MAE':>8s} | {'Test MdAE':>9s} | {'Test RMSE':>9s}")
    print("-" * 88)
    full_table = data_p2['item3_taskove_variants']['models_evaluation']
    for m_name, v in full_table.items():
        print(f"{'Full Population (N=6409)':22s} | {m_name:30s} | {v['MAE']:8.4f} | {v['MdAE']:9.4f} | {v['RMSE']:9.4f}")
    print("-" * 88)
    print(f"{'Epic-Excluded (N=6167)':22s} | {'Baseline: Per-Type Median':30s} | {1.9729:8.4f} | {1.0000:9.4f} | {3.3185:9.4f}")
    print(f"{'Epic-Excluded (N=6167)':22s} | {'Type Only':30s} | {cand_summary['type_only']['seed_42_single_run']['MAE']:8.4f} | {cand_summary['type_only']['seed_42_single_run']['MdAE']:9.4f} | {cand_summary['type_only']['seed_42_single_run']['RMSE']:9.4f}")
    print(f"{'Epic-Excluded (N=6167)':22s} | {'TF-IDF + Type':30s} | {cand_summary['tfidf_type']['seed_42_single_run']['MAE']:8.4f} | {cand_summary['tfidf_type']['seed_42_single_run']['MdAE']:9.4f} | {cand_summary['tfidf_type']['seed_42_single_run']['RMSE']:9.4f}")
    print(f"{'Epic-Excluded (N=6167)':22s} | {'S-BERT + Type':30s} | {cand_summary['sbert_type']['seed_42_single_run']['MAE']:8.4f} | {cand_summary['sbert_type']['seed_42_single_run']['MdAE']:9.4f} | {cand_summary['sbert_type']['seed_42_single_run']['RMSE']:9.4f}")

    print("\nPaired Bootstrap on Epic-Excluded Population (1,000 Draws, Seed 42, Tickets AND Projects):")
    boot_ne = data['item1_final_candidates_epic_excluded']['paired_bootstrap_1000']
    for b_key in ['type_only_vs_per_type_median', 'tfidf_type_vs_per_type_median', 'sbert_type_vs_per_type_median', 'tfidf_type_vs_type_only']:
        b_info = boot_ne[b_key]
        t = b_info['ticket_level']
        p = b_info['project_level']
        print(f"  {b_info['contrast']}:")
        print(f"    Ticket Resampling:  Mean Diff = {t['mean_difference']:+.4f} | 95% CI = [{t['ci_95'][0]:.4f}, {t['ci_95'][1]:.4f}] | Excludes 0: {t['excludes_zero']}")
        print(f"    Project Resampling: Mean Diff = {p['mean_difference']:+.4f} | 95% CI = [{p['ci_95'][0]:.4f}, {p['ci_95'][1]:.4f}] | Excludes 0: {p['excludes_zero']}")

    # 2. Cold-Start Test
    print("\n" + "-" * 105)
    print("2. COLD-START TEST: GROUPED 5-FOLD CV BY PROJECT (EPIC-EXCLUDED POPULATION)")
    print("-" * 105)
    cs = data['item2_cold_start_test']['grouped_5_fold_cv_by_project']
    print(f"{'Fold':5s} | {'Test Tickets':>12s} | {'Global Med MAE':>14s} | {'Per-Type MAE':>12s} | {'Type Only MAE':>13s} | {'TF-IDF+Type MAE':>15s} | {'S-BERT+Type MAE':>15s}")
    print("-" * 95)
    for f_row in cs['folds']:
        print(f"Fold {f_row['fold']:1d} | {f_row['test_tickets']:12d} | {f_row['MAE_global_median']:14.4f} | {f_row['MAE_per_type_median']:12.4f} | {f_row['MAE_type_only']:13.4f} | {f_row['MAE_tfidf_type']:15.4f} | {f_row['MAE_sbert_type']:15.4f}")
        print(f"   -> Held-Out Projects in Fold {f_row['fold']}: {f_row['test_projects']}")
    print("-" * 95)
    m = cs['metrics_mean_std']
    print(f"{'Mean ± Std':19s} | {m['global_median']['mean']:.4f} ± {m['global_median']['std']:.4f} | {m['per_type_median']['mean']:.4f} ± {m['per_type_median']['std']:.4f} | {m['type_only']['mean']:.4f} ± {m['type_only']['std']:.4f} | {m['tfidf_type']['mean']:.4f} ± {m['tfidf_type']['std']:.4f} | {m['sbert_type']['mean']:.4f} ± {m['sbert_type']['std']:.4f}")
    print(f"Rel Imp vs Global Median:      {'Baseline':>14s} | {'+'+str(round((m['global_median']['mean']-m['per_type_median']['mean'])/m['global_median']['mean']*100, 2))+'%':>12s} | {'+'+str(m['type_only']['rel_imp_vs_global_pct'])+'%':>13s} | {'+'+str(m['tfidf_type']['rel_imp_vs_global_pct'])+'%':>15s} | {'+'+str(m['sbert_type']['rel_imp_vs_global_pct'])+'%':>15s}")
    print(f"Rel Imp vs Per-Type Median:    {'—':>14s} | {'Baseline':>12s} | {'+'+str(m['type_only']['rel_imp_vs_per_type_pct'])+'%':>13s} | {'+'+str(m['tfidf_type']['rel_imp_vs_per_type_pct'])+'%':>15s} | {'+'+str(m['sbert_type']['rel_imp_vs_per_type_pct'])+'%':>15s}")

    print("\nEarlier Held-Out 5 Projects [2, 8, 15, 16, 18] (Epic-Excluded Population, N=1,241):")
    h5 = data['item2_cold_start_test']['heldout_5_projects_epic_excluded']
    print(f"  Global Median MAE:   {h5['MAE_global_median']:.4f}")
    print(f"  Per-Type Median MAE: {h5['MAE_per_type_median']:.4f}")
    print(f"  Type Only MAE:       {h5['MAE_type_only']:.4f}")
    print(f"  TF-IDF + Type MAE:   {h5['MAE_tfidf_type']:.4f} (Rel Imp vs Global: +{h5['rel_imp_tfidf_vs_global_pct']}%, vs Per-Type: +{h5['rel_imp_tfidf_vs_per_type_pct']}%)")

    # 3. Project Identity Control
    print("\n" + "-" * 105)
    print("3. PROJECT-IDENTITY CONTROL (CHRONOLOGICAL SPLIT, EPIC-EXCLUDED)")
    print("-" * 105)
    pic = data['item3_project_identity_control']
    print(f"{'Model Architecture':35s} | {'Test MAE':>8s} | {'Test MdAE':>9s} | {'Test RMSE':>9s}")
    print("-" * 65)
    print(f"{'Type Only':35s} | {pic['eval_type_only']['MAE']:8.4f} | {pic['eval_type_only']['MdAE']:9.4f} | {pic['eval_type_only']['RMSE']:9.4f}")
    print(f"{'Type + Project-ID One-Hot':35s} | {pic['eval_type_plus_project_id']['MAE']:8.4f} | {pic['eval_type_plus_project_id']['MdAE']:9.4f} | {pic['eval_type_plus_project_id']['RMSE']:9.4f}")
    print(f"{'TF-IDF + Type':35s} | {pic['eval_tfidf_plus_type']['MAE']:8.4f} | {pic['eval_tfidf_plus_type']['MdAE']:9.4f} | {pic['eval_tfidf_plus_type']['RMSE']:9.4f}")
    print(f"\n{pic['plain_statement']}")

    # 4. Scale Check
    print("\n" + "-" * 105)
    print("4. SCALE CHECK FOR TOP 10 TEST PROJECTS (EPIC-EXCLUDED POPULATION)")
    print("-" * 105)
    print(f"{'Proj ID':>8s} | {'Tickets':>8s} | {'Median SP':>10s} | {'P90 SP':>8s} | {'Max SP':>8s} | {'Share SP > 13':>14s}")
    print("-" * 65)
    for r in data['item4_scale_check']['top_10_projects_scale']:
        print(f"{r['project_id']:8d} | {r['test_tickets']:8d} | {r['median']:10.2f} | {r['p90']:8.2f} | {r['max']:8.2f} | {r['share_sp_gt_13_pct']:13.2f}%")
    print(f"\nDoes Project 28 scale differ? {data['item4_scale_check']['project_28_comparison']['does_project_28_scale_differ']}")
    print(f"{data['item4_scale_check']['project_28_comparison']['plain_explanation']}")

    # 5. Boosting Shrinkage Explanation
    print("\n" + "-" * 105)
    print("5. BOOSTING SHRINKAGE & INITIAL SCORE EXPERIMENT (EPIC PREDICTIONS)")
    print("-" * 105)
    bs = data['item5_boosting_shrinkage_explanation']
    print(f"Global Train Median (Init Score): {bs['initial_score']}")
    print(f"Training Epic Median:            {bs['training_epic_median']}")
    print(f"Test Epic Median:                {bs['test_epic_median']}")
    print("\nIteration Progression (learning_rate = 0.05):")
    for r in bs['n_estimators_sweep']:
        print(f"  n_estimators = {r['n_estimators']:3d} | Test Epic Mean = {r['test_epic_mean']:5.2f} | Test Epic Median = {r['test_epic_median']:5.2f}")
    print("\nLearning Rate Sweep (n_estimators = 200):")
    for r in bs['learning_rate_sweep']:
        print(f"  learning_rate = {r['learning_rate']:.2f} | Test Epic Mean = {r['test_epic_mean']:5.2f} | Test Epic Median = {r['test_epic_median']:5.2f}")
    print(f"\nEmpirical Finding:\n{bs['finding']}")

    # 6. Model Selection Rule & Artifact Packaging
    print("\n" + "-" * 105)
    print("6. FINAL MODEL SELECTION & VERIFICATION (V2_TASKOVE)")
    print("-" * 105)
    sel = data['item6_model_selection']['selection_log']
    print(f"Declared Selection Rule: {sel['rule']}")
    print(f"Validation MAEs:         {sel['candidates_val_mae']}")
    print(f"Selected Winner:         {sel['selected_winner']}")
    print(f"Selection Rationale:     {sel['rationale']}")
    meta_v2 = data['metadata_v2']
    print(f"\nFinal Test Metrics (v2_taskove): {meta_v2['test_metrics']}")
    print(f"Conformal Interval:              {meta_v2['conformal_interval']}")
    print(f"Artifact Directory:              benchmark_v2/artifacts/model1_sp/v2_taskove/")
    print(f"Taskove Type Mappings Count:     {len(meta_v2['taskove_to_tawos_type_mapping'])}")
    for t_map in meta_v2['taskove_to_tawos_type_mapping']:
        print(f"  {t_map['taskove_type']:10s} -> {t_map['mapped_tawos_type']:15s} ({t_map['mapping_nature']})")

if __name__ == "__main__":
    main()

