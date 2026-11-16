import os
import sys
import pandas as pd

BASE_DIR = r"d:\Gillo AI Service\benchmark_v2"
work_dir = os.path.join(BASE_DIR, "work")
tawos_dir = r"d:\Gillo AI Service\tawos_csv"
sys.path.insert(0, BASE_DIR)

from replay import TODO_SET, DONE_SET, IN_PROGRESS_SET, categorize_status

changelog = pd.read_parquet(os.path.join(work_dir, "changelog_slim.parquet"))
status_rows = changelog[changelog['Field'] == 'status']
all_statuses = sorted(list(set(status_rows['From_String'].dropna().unique()) | set(status_rows['To_String'].dropna().unique())))

exact_set_matches = {}
substring_matches = {}
fallback_matches = {}

for s in all_statuses:
    clean = ' '.join(s.strip().lower().split())
    cat = categorize_status(s)
    if clean in DONE_SET:
        exact_set_matches[s] = ('done', 'DONE_SET')
    elif clean in TODO_SET:
        exact_set_matches[s] = ('todo', 'TODO_SET')
    elif clean in IN_PROGRESS_SET:
        exact_set_matches[s] = ('in_progress', 'IN_PROGRESS_SET')
    elif any(w in clean for w in ['resolved', 'closed', 'done', 'complete', 'verified', 'shipped']):
        substring_matches[s] = (cat, 'done_substring')
    elif any(w in clean for w in ['to do', 'backlog', 'reopen', 'new', 'triage', 'deferred']):
        substring_matches[s] = (cat, 'todo_substring')
    elif any(w in clean for w in ['progress', 'review', 'testing', 'test', 'qa', 'accepted', 'active', 'pr', 'coding', 'deploy', 'investigat', 'working', 'merge']):
        substring_matches[s] = (cat, 'in_progress_substring')
    else:
        fallback_matches[s] = (cat, 'default_fallback')

print(f"Total unique statuses: {len(all_statuses)}")
print(f"Exact set matches: {len(exact_set_matches)}")
print(f"Substring matches: {len(substring_matches)}")
print(f"Fallback matches: {len(fallback_matches)}")

print("\n=== 35 SUBSTRING MATCHES AUDIT ===")
negative_words = ['not', 'un', "won't", 'wont', 'cannot', 'no']

flagged = []
for idx, (s, (cat, rule)) in enumerate(sorted(substring_matches.items()), 1):
    clean = s.strip().lower()
    has_neg = any(w in clean.split() or clean.startswith(w) for w in negative_words)
    is_done = (cat == 'done')
    marker = []
    if has_neg:
        marker.append("NEGATION")
    if is_done:
        marker.append("MAPPED_TO_DONE")
    marker_str = f" *** [{', '.join(marker)}] ***" if marker else ""
    print(f"{idx:2d}. '{s}': category='{cat}' (rule: {rule}){marker_str}")
    if marker:
        flagged.append((s, cat, rule, marker))

print(f"\nTotal flagged substring matches: {len(flagged)}")
for item in flagged:
    print("  Flagged:", item)

