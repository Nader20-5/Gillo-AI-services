import os
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAWOS_DIR = os.path.join(ROOT_DIR, "tawos_csv")

cols = [
    "ID", "Title", "Description_Text", "Type", "Priority",
    "Story_Point", "Estimation_Date", "Creation_Date", "Project_ID",
    "Story_Point_Changed_After_Estimation",
    "Title_Changed_After_Estimation",
    "Description_Changed_After_Estimation",
    "Resolution"
]
df_raw = pd.read_csv(os.path.join(TAWOS_DIR, "issue.csv"), usecols=cols, low_memory=False)
df_raw.columns = [c.lstrip('\ufeff') for c in df_raw.columns]

df_sp = df_raw[df_raw['Story_Point'].notna() & (df_raw['Story_Point'] > 0) & (df_raw['Story_Point'] <= 100.0)].copy()
excluded_resolutions = {"won't fix", "wont fix", "wontfix", "invalid", "duplicate", "cannot reproduce"}
res_clean = df_sp['Resolution'].fillna("").str.lower().str.strip()
df_valid = df_sp[~res_clean.isin(excluded_resolutions)].copy()

clean_mask = (
    (df_valid['Story_Point_Changed_After_Estimation'] == 0) &
    (df_valid['Title_Changed_After_Estimation'] == 0) &
    (df_valid['Description_Changed_After_Estimation'] == 0)
)
df_clean = df_valid[clean_mask].copy()
df_clean['Timestamp_Str'] = df_clean['Estimation_Date'].fillna(df_clean['Creation_Date'])
df_clean['Timestamp_dt'] = pd.to_datetime(df_clean['Timestamp_Str'], errors='coerce')
df_clean = df_clean.dropna(subset=['Timestamp_dt']).sort_values('Timestamp_dt').reset_index(drop=True)

N_all = len(df_clean)
idx_tr = int(N_all * 0.70)
df_train_all = df_clean.iloc[:idx_tr].copy().reset_index(drop=True)
df_train = df_train_all[df_train_all['Type'] != 'Epic'].copy().reset_index(drop=True)

print(f"Train set (Epic-excluded): N={len(df_train)}")

global_q25 = float(df_train['Story_Point'].quantile(0.25))
global_q50 = float(df_train['Story_Point'].quantile(0.50))
global_q75 = float(df_train['Story_Point'].quantile(0.75))

print(f"Global quantiles: Q25={global_q25}, Q50={global_q50}, Q75={global_q75}")

type_stats = {}
type_medians = {}
type_ranges = {}

for t, grp in df_train.groupby('Type'):
    q25 = float(grp['Story_Point'].quantile(0.25))
    q50 = float(grp['Story_Point'].quantile(0.50))
    q75 = float(grp['Story_Point'].quantile(0.75))
    n = len(grp)
    type_stats[t] = {
        'count': n,
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'interval': [round(q25, 1), round(q75, 1)]
    }
    type_medians[t] = round(q50, 1)
    type_ranges[t] = [round(q25, 1), round(q75, 1)]
    print(f"{t:<22} | N={n:<6} | Q25={q25:4.1f} | Q50={q50:4.1f} | Q75={q75:4.1f} | Range: [{round(q25, 1)}, {round(q75, 1)}]")

output = {
    "description": "Per-type median story points and descriptive [Q25, Q75] range computed from TAWOS training data (Epic-excluded, N=29,330)",
    "global_median": float(global_q50),
    "global_interval": [round(global_q25, 1), round(global_q75, 1)],
    "type_medians": type_medians,
    "type_intervals": type_ranges,
    "details": type_stats
}

with open(os.path.join(ROOT_DIR, "reports", "computed_type_quantiles.json"), "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nSaved computed quantiles to reports/computed_type_quantiles.json")
