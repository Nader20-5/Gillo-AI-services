"""
Dataset Generation Script for Gillo AI Service (Phase 1 / Phase 2)
Generates schema-aligned Agile datasets for:
1. Model 1 (Task Effort Regressor - XGBoost)
2. Model 2 (Slippage Risk Classifier - LightGBM + SHAP)
3. Model 3 (Instruction-Tuning JSONL for QLoRA Agile Drafts)
"""

import os
import random
import json
import numpy as np
import pandas as pd

# Set seed for reproducible academic evaluation
np.random.seed(42)
random.seed(42)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Dataset Generation for Model 1 (Effort Estimation)
# ---------------------------------------------------------------------------
TASK_TITLES = [
    ("Implement JWT authentication with refresh token rotation", "feature", 5.0),
    ("Fix race condition in database connection pool", "bug", 3.0),
    ("Create responsive sprint Kanban board component", "feature", 8.0),
    ("Optimize slow MongoDB aggregate pipeline on time entries", "refactor", 5.0),
    ("Write unit tests for suggestions inbox state machine", "test", 3.0),
    ("Add rate limiting middleware with Redis token bucket", "infra", 5.0),
    ("Fix CSS overflow on task card tags in mobile view", "bug", 1.0),
    ("Design schema for multi-tenant project roles", "feature", 5.0),
    ("Migrate Kafka consumer group to async event dispatcher", "refactor", 8.0),
    ("Configure Docker multi-stage build for minimal image size", "infra", 3.0),
    ("Add webhook listener for GitHub PR merge events", "feature", 5.0),
    ("Fix NullPointer when task assignee is unassigned", "bug", 2.0),
    ("Implement batch export of sprint metrics to CSV/PDF", "feature", 5.0),
    ("Refactor user profile estimation ratio calculator", "refactor", 3.0),
    ("Set up automated end-to-end Cypress tests for auth", "test", 8.0),
    ("Investigate memory leak in WebSocket notification server", "bug", 8.0),
    ("Build early risk warning indicator with SHAP badge", "feature", 5.0),
    ("Write release request pre-fill prompt template", "feature", 3.0),
    ("Update dependency vulnerabilities in package.json", "infra", 2.0),
    ("Document API endpoints with OpenAPI Swagger schemas", "docs", 2.0),
]

def generate_model1_dataset(n_samples=2500):
    records = []
    task_types = ["feature", "bug", "infra", "refactor", "test", "docs"]
    story_points_list = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]

    for i in range(n_samples):
        base_title, base_type, base_sp = random.choice(TASK_TITLES)
        task_type = base_type if random.random() > 0.15 else random.choice(task_types)
        story_points = base_sp if random.random() > 0.2 else random.choice(story_points_list)
        
        # Dependencies: higher points often have more dependencies
        dep_count = np.random.poisson(lam=story_points * 0.3)
        dep_count = min(dep_count, 6)

        # Developer estimation ratio (0.8 = fast/under-estimator, 1.4 = slower/over-promiser)
        assignee_ratio = float(np.clip(np.random.normal(loc=1.05, scale=0.18), 0.70, 1.55))

        # Baseline median hours roughly 4-6 hours per story point
        type_multiplier = {"feature": 5.5, "bug": 4.0, "infra": 6.0, "refactor": 4.5, "test": 3.5, "docs": 2.5}
        base_hours = story_points * type_multiplier.get(task_type, 4.5)
        historical_median = round(base_hours * np.random.uniform(0.9, 1.1), 1)

        # True actual hours includes log-normal noise + developer ratio effect + dependency delay
        noise = np.random.lognormal(mean=0.0, sigma=0.25)
        dep_overhead = 1.0 + (dep_count * 0.08)
        actual_hours = round(historical_median * assignee_ratio * dep_overhead * noise, 1)
        actual_hours = max(0.5, actual_hours)

        records.append({
            "task_id": f"task-ml1-{i:05d}",
            "title": f"{base_title} (variant #{i})",
            "task_type": task_type,
            "story_points": story_points,
            "dependency_count": dep_count,
            "historical_median_hours": historical_median,
            "assignee_user_ratio": round(assignee_ratio, 2),
            "actual_hours": actual_hours
        })

    df = pd.DataFrame(records)
    # 80% Train, 20% Test
    train_df = df.iloc[:int(0.8 * len(df))]
    test_df = df.iloc[int(0.8 * len(df)):]

    train_df.to_csv(os.path.join(PROCESSED_DIR, "model1_effort_train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "model1_effort_test.csv"), index=False)
    print(f" Model 1 datasets generated: {len(train_df)} train, {len(test_df)} test.")


# ---------------------------------------------------------------------------
# 2. Dataset Generation for Model 2 (Slippage Risk Classifier)
# ---------------------------------------------------------------------------
def generate_model2_dataset(n_samples=3000):
    records = []
    task_types = ["feature", "bug", "infra", "refactor"]

    for i in range(n_samples):
        task_type = random.choice(task_types)
        story_points = random.choice([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
        
        # Risk factors
        stalled_days = int(np.random.exponential(scale=2.2))
        stalled_days = min(stalled_days, 15)

        dep_depth = int(np.random.poisson(lam=1.0))
        dep_depth = min(dep_depth, 5)

        active_blockers = 0
        if random.random() < 0.22:
            active_blockers = random.choice([1, 2, 3])

        # Assignee load ratio: 1.0 = 100% capacity, >1.0 = overloaded
        assignee_load_ratio = float(np.clip(np.random.normal(loc=1.0, scale=0.25), 0.4, 1.8))

        # Logged hours vs original estimated hours
        logged_vs_est_ratio = float(np.clip(np.random.normal(loc=0.75, scale=0.35), 0.05, 2.5))

        # Logistic function to determine realistic slippage probability
        # Strongest drivers: stalled_days, active_blockers, assignee overload, high logged ratio with task unfinished
        linear_score = (
            -3.2
            + (0.45 * stalled_days)
            + (1.20 * active_blockers)
            + (1.40 * max(0.0, assignee_load_ratio - 1.0))
            + (0.60 * max(0.0, logged_vs_est_ratio - 1.0))
            + (0.15 * dep_depth)
            + (0.05 * story_points)
        )
        prob_slip = 1.0 / (1.0 + np.exp(-linear_score))
        is_slipped = 1 if random.random() < prob_slip else 0

        records.append({
            "task_id": f"task-ml2-{i:05d}",
            "task_type": task_type,
            "story_points": story_points,
            "stalled_days": stalled_days,
            "dependency_depth": dep_depth,
            "active_blockers": active_blockers,
            "assignee_load_ratio": round(assignee_load_ratio, 2),
            "logged_vs_estimated_ratio": round(logged_vs_est_ratio, 2),
            "is_slipped": is_slipped
        })

    df = pd.DataFrame(records)
    train_df = df.iloc[:int(0.8 * len(df))]
    test_df = df.iloc[int(0.8 * len(df)):]

    train_df.to_csv(os.path.join(PROCESSED_DIR, "model2_risk_train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "model2_risk_test.csv"), index=False)
    
    slipped_pct = (df['is_slipped'].sum() / len(df)) * 100
    print(f" Model 2 datasets generated: {len(train_df)} train, {len(test_df)} test (Slipped rate: {slipped_pct:.1f}%).")


# ---------------------------------------------------------------------------
# 3. Instruction Tuning Dataset for Model 3 (JSON Schema Prompts)
# ---------------------------------------------------------------------------
def generate_model3_jsonl(n_samples=250):
    jsonl_path = os.path.join(PROCESSED_DIR, "model3_instruction_tuning.jsonl")
    entries = []

    for i in range(n_samples):
        kind = random.choice(["standup", "retro", "release_summary"])
        if kind == "standup":
            system_prompt = "You are an Agile assistant. Output a personal daily standup in strict JSON schema format."
            user_prompt = f"Generate standup draft for developer Nader on Sprint 4 with completed task 'API Endpoints' (6h logged) and active task 'Docker setup'."
            assistant_reply = {
                "kind": "standup",
                "yesterday": [
                    "Completed API endpoints for AI suggestions inbox decisions",
                    "Logged 6 hours and pushed branch feat/inbox-api"
                ],
                "today": [
                    "Continue Docker compose setup and configure volume persistence",
                    "Coordinate with Sama on ML seed models"
                ],
                "blockers": []
            }
        elif kind == "retro":
            system_prompt = "You are an Agile assistant. Output a sprint retrospective summary in strict JSON schema format."
            user_prompt = "Generate sprint retrospective summary for Sprint 3. Velocity: 42 points. 2 tasks slipped due to unmerged PRs."
            assistant_reply = {
                "kind": "retro",
                "sprint_name": "Sprint 3",
                "velocity_achieved": 42,
                "what_went_well": [
                    "Core model training scripts executed with 0 build errors",
                    "Good collaboration between frontend and AI service teams"
                ],
                "what_needs_improvement": [
                    "2 tasks slipped due to late PR reviews and blocked CI pipeline",
                    "Initial effort estimates were underestimated by 18%"
                ],
                "action_items": [
                    "Introduce daily PR review triage at 10 AM",
                    "Use Model 1 effort regressor before committing to sprint backlog"
                ]
            }
        else:
            system_prompt = "You are an Agile assistant. Output release documentation in strict JSON schema format."
            user_prompt = "Generate release summary and rollback plan for version v1.2.0 including auth and suggestion features."
            assistant_reply = {
                "kind": "release_summary",
                "version": "v1.2.0",
                "highlights": [
                    "AI Suggestions Inbox with accept/reject state machine",
                    "Automated daily early risk warning alerts with SHAP evidence"
                ],
                "rollback_plan": [
                    "Step 1: Revert service deployment to image tag v1.1.8",
                    "Step 2: Run rollback MongoDB migration script to restore suggestion status index",
                    "Step 3: Verify /health endpoint returns HTTP 200 OK"
                ]
            }

        entries.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(assistant_reply, indent=2)}
            ]
        })

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f" Model 3 JSONL instruction dataset generated: {len(entries)} prompt-completion turns.")


if __name__ == "__main__":
    print("Generating schema-aligned Agile ML datasets...")
    generate_model1_dataset()
    generate_model2_dataset()
    generate_model3_jsonl()
    print("All Phase 2 datasets ready in data/processed/")
