"""
Exports unfilled human rating sheet for Model 3 evaluation.
Output: reports/model3_eval/human_rating.csv
Columns: id, provider, endpoint, input_summary, output_text, usefulness, clarity, factual_errors, notes
Selects 20 random deterministic outputs (seed=42) ready for team review.
"""

import csv
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "model3_eval"
TEMPLATE_RESULTS_PATH = REPORTS_DIR / "results_template.json"
OUTPUT_CSV_PATH = REPORTS_DIR / "human_rating.csv"


def export_human_rating_sheet():
    if not TEMPLATE_RESULTS_PATH.exists():
        print(f"Results file {TEMPLATE_RESULTS_PATH} does not exist.")
        return

    with open(TEMPLATE_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("per_scenario", [])
    if not scenarios:
        print("No scenarios found in results.")
        return

    # Select 20 random deterministic samples
    random.seed(42)
    selected = random.sample(scenarios, min(20, len(scenarios)))
    selected.sort(key=lambda s: s["id"])

    fieldnames = [
        "id",
        "provider",
        "endpoint",
        "input_summary",
        "output_text",
        "usefulness",       # 1-5, blank
        "clarity",          # 1-5, blank
        "factual_errors",   # yes/no, blank
        "notes",            # blank
    ]

    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for s in selected:
            # Build clean input summary
            desc = s.get("description", s.get("name", ""))
            edge = s.get("edge_case", "")
            summary = f"[{edge}] {desc}"

            # Format readable output text
            parsed = s.get("parsed_output", {})
            if isinstance(parsed, dict) and "answer" in parsed:
                out_text = parsed["answer"]
            else:
                out_text = json.dumps(parsed, indent=2, ensure_ascii=False)

            writer.writerow({
                "id": s["id"],
                "provider": s.get("method", "template"),
                "endpoint": s.get("endpoint", ""),
                "input_summary": summary,
                "output_text": out_text,
                "usefulness": "",
                "clarity": "",
                "factual_errors": "",
                "notes": "",
            })

    print(f"Exported {len(selected)} samples to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    export_human_rating_sheet()
