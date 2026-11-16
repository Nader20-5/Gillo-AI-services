"""
Taskove Demo Dataset Seeder (scripts/seed_data.py)

Generates a realistic demo dataset that exercises Model 1 (Story Point Estimation)
and Model 2 (Sprint Carry-Over Risk Scoring) honestly inside their validated operating ranges.

RULES & BOUNDARIES:
1. Seed data is strictly DEMO data tagged with source="seed_demo".
   Never use this demo data for training or for accuracy claims.
2. Does NOT fabricate model outputs, outcomes, or "ground truth" (no hardcoded AI suggestions or estimates).
3. All dates are computed relative to --now (default: current UTC) so the demo remains valid whenever run.
4. Idempotent: running twice updates documents in-place by deterministic IDs without duplicating.
   --reset wipes existing documents with source="seed_demo" only.
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.config import settings

PROJECT_ID = "proj-gillo-core"

# TODO: Replace with the official production Taskove task type enum once confirmed by the backend team.
TASK_TYPES: List[str] = [
    "feature", "bug", "infra", "refactor", "test",
    "docs", "backend", "frontend", "ml", "qa"
]

USERS = [
    {"user_id": "usr-nader", "name": "Nader", "role": "TL", "ratio": 1.05},
    {"user_id": "usr-sama", "name": "Sama", "role": "Developer", "ratio": 0.95},
    {"user_id": "usr-alex", "name": "Alex", "role": "PM", "ratio": 1.00},
    {"user_id": "usr-kareem", "name": "Kareem", "role": "Developer", "ratio": 1.35},
    {"user_id": "usr-layla", "name": "Layla", "role": "Developer", "ratio": 0.90},
]


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def seed_data(
    db: Any,
    now: Optional[datetime] = None,
    reset: bool = False
) -> Dict[str, Any]:
    """
    Populates or updates the demo dataset relative to `now`.
    If reset=True, removes all existing documents matching {"source": "seed_demo"} first.
    """
    now = ensure_utc(now) or datetime.now(timezone.utc)
    print(f"[seed_data] Starting demo seed relative to reference time (now): {now.isoformat()}")

    collections_to_manage = [
        "ai_profiles", "sprint_snapshots", "capacity", "task_snapshots",
        "task_events", "time_entries", "standup_blockers", "pull_requests",
        "release_requests", "ai_suggestions", "task_estimates", "sprint_fit_checks"
    ]

    if reset:
        print("[seed_data] --reset enabled: Wiping all documents with source='seed_demo'...")
        for col in collections_to_manage:
            res = await db[col].delete_many({"source": "seed_demo"})
            if res.deleted_count > 0:
                print(f"  - Cleared {res.deleted_count} demo documents from {col}")

    # =========================================================================
    # 1. Developer AI Profiles (No developer ranking, No mood data)
    # =========================================================================
    profiles = []
    for u in USERS:
        profiles.append({
            "user_id": u["user_id"],
            "name": u["name"],
            "role": u["role"],
            "estimation": {
                "ratio": u["ratio"],
                "sample_size": 24 if u["role"] == "Developer" else 15
            },
            "skills": ["Python", "FastAPI", "MongoDB", "PyTorch"] if u["name"] in ["Sama", "Nader"] else ["React", "TypeScript", "Node.js"],
            "source": "seed_demo",
            "updated_at": now - timedelta(days=2)
        })

    for p in profiles:
        await db.ai_profiles.update_one(
            {"user_id": p["user_id"]},
            {"$set": p},
            upsert=True
        )

    # =========================================================================
    # 2. Sprints
    # - 3 Closed Historical Sprints
    # - 2 Active Sprints in validated range (f in [0.30, 0.70]), each with >= 12 open tasks
    # - 1 Active Sprint with < 10 open tasks (tests eligibility gate)
    # - 1 Active Sprint with f < 0.20 (tests elapsed fraction gate)
    # =========================================================================
    sprints = [
        # Historical Closed Sprints
        {
            "sprint_id": "sprint-01",
            "project_id": PROJECT_ID,
            "name": "Sprint 1: Architecture & Foundation",
            "start_date": now - timedelta(days=56),
            "end_date": now - timedelta(days=42),
            "status": "completed",
            "source": "seed_demo"
        },
        {
            "sprint_id": "sprint-02",
            "project_id": PROJECT_ID,
            "name": "Sprint 2: Core Read Models & Auth",
            "start_date": now - timedelta(days=42),
            "end_date": now - timedelta(days=28),
            "status": "completed",
            "source": "seed_demo"
        },
        {
            "sprint_id": "sprint-03",
            "project_id": PROJECT_ID,
            "name": "Sprint 3: ML Pipeline & Feature Extractor",
            "start_date": now - timedelta(days=28),
            "end_date": now - timedelta(days=14),
            "status": "completed",
            "source": "seed_demo"
        },
        # Active Sprint 1 (In Validated Range: f = 7/14 = 0.50, >= 12 open tasks)
        {
            "sprint_id": "sprint-04",
            "project_id": PROJECT_ID,
            "name": "Sprint 4: Active Core Features & Inference",
            "start_date": now - timedelta(days=7),
            "end_date": now + timedelta(days=7),
            "status": "active",
            "source": "seed_demo"
        },
        # Active Sprint 2 (In Validated Range: f = 6/14 = 0.4286, >= 12 open tasks)
        {
            "sprint_id": "sprint-05",
            "project_id": PROJECT_ID,
            "name": "Sprint 5: Integrations, Reporting & Web UI",
            "start_date": now - timedelta(days=6),
            "end_date": now + timedelta(days=8),
            "status": "active",
            "source": "seed_demo"
        },
        # Ineligible Sprint 1 (< 10 open tasks)
        {
            "sprint_id": "sprint-small-active",
            "project_id": PROJECT_ID,
            "name": "Sprint Ineligible: Small Task Count",
            "start_date": now - timedelta(days=7),
            "end_date": now + timedelta(days=7),
            "status": "active",
            "source": "seed_demo"
        },
        # Ineligible Sprint 2 (f < 0.20: elapsed 1 day of 14 -> f = 0.0714)
        {
            "sprint_id": "sprint-early-active",
            "project_id": PROJECT_ID,
            "name": "Sprint Ineligible: Just Started",
            "start_date": now - timedelta(days=1),
            "end_date": now + timedelta(days=13),
            "status": "active",
            "source": "seed_demo"
        }
    ]

    for s in sprints:
        await db.sprint_snapshots.update_one(
            {"sprint_id": s["sprint_id"]},
            {"$set": s},
            upsert=True
        )

    # =========================================================================
    # 3. User Capacity
    # =========================================================================
    capacities = []
    for sp in sprints:
        for u in USERS:
            capacities.append({
                "user_id": u["user_id"],
                "sprint_id": sp["sprint_id"],
                "capacity_hours": 60.0 if u["role"] == "Developer" else 40.0,
                "source": "seed_demo"
            })

    for c in capacities:
        await db.capacity.update_one(
            {"user_id": c["user_id"], "sprint_id": c["sprint_id"]},
            {"$set": c},
            upsert=True
        )

    # =========================================================================
    # 4. Tasks and Event History Specifications
    # =========================================================================
    # Historical Tasks (Sprints 1-3)
    hist_task_defs = [
        ("task-hist-01", "Setup Docker and CI/CD pipelines", "infra", 3.0, "usr-nader", "sprint-01", 18.0, 50, 48),
        ("task-hist-02", "Design schema for MongoDB-AI read models", "backend", 5.0, "usr-sama", "sprint-01", 28.0, 48, 45),
        ("task-hist-03", "Implement JWT Auth and Role Validation", "backend", 3.0, "usr-kareem", "sprint-01", 24.0, 46, 43),
        ("task-hist-04", "Kafka Consumer event dispatcher", "backend", 5.0, "usr-nader", "sprint-02", 35.0, 38, 35),
        ("task-hist-05", "Scrape and curate Agile dataset", "ml", 8.0, "usr-sama", "sprint-02", 45.0, 36, 32),
        ("task-hist-06", "Build Suggestion Inbox frontend mock", "frontend", 5.0, "usr-layla", "sprint-02", 30.0, 35, 30),
        ("task-hist-07", "Train Effort Regressor Model 1", "ml", 8.0, "usr-sama", "sprint-03", 50.0, 24, 20),
        ("task-hist-08", "Train Slippage Risk Classifier Model 2", "ml", 8.0, "usr-nader", "sprint-03", 42.0, 22, 18),
        ("task-hist-09", "Integrate SHAP TreeExplainer for reasoning", "ml", 5.0, "usr-nader", "sprint-03", 32.0, 20, 16),
    ]

    # Active Tasks for Sprint 4 (16 tasks: 13 open, 3 done; ~25% mid-sprint additions; ~18% unestimated; 3 empty descriptions; 1 non-English)
    s4_start = now - timedelta(days=7)
    s4_task_defs = [
        # tid, title, desc, t_type, sp, assignee, status, stalled_days, deps, created_days_ago, assigned_after_start
        ("task-s4-01", "Implement async MongoDB connection pooling with health check", "Refactor database client to use bounded connection pool and add automatic ping on reconnection.", "backend", 5.0, "usr-sama", "in_progress", 2, [], 20, False),
        ("task-s4-02", "Fix token expiry race condition in refresh endpoint", "When two concurrent requests hit refresh token endpoint simultaneously, second request triggers 401 instead of graceful rotation.", "bug", 3.0, "usr-kareem", "in_progress", 5, [], 15, False),
        ("task-s4-03", "Add Prometheus metrics exporter for HTTP endpoint latencies", "Expose request count, duration histograms, and error counters at /metrics for Grafana ingestion.", "infra", 3.0, "usr-nader", "todo", 7, [], 18, False),
        ("task-s4-04", "Build responsive sprint burndown chart with date range picker", "", "frontend", 5.0, "usr-layla", "in_progress", 1, [], 10, False),  # Empty description 1
        ("task-s4-05", "Migrate legacy user profile queries to projection aggregations", "Replace find() full document fetch with aggregate $project pipeline to decrease network payload size.", "refactor", None, "usr-sama", "todo", 3, ["task-s4-01"], 4, True),  # Unestimated 1, Mid-sprint 1
        ("task-s4-06", "Configure automated end-to-end Playwright test suite in GitHub Actions", "Add headless browser tests covering login flow, sprint creation, and suggestion acceptance.", "test", 5.0, "usr-kareem", "in_progress", 4, ["task-s4-02"], 14, False),
        ("task-s4-07", "Write deployment runbook and disaster recovery guide for MongoDB replica set", "", "docs", 2.0, "usr-alex", "todo", 6, [], 12, False),  # Empty description 2
        ("task-s4-08", "Implement model inference telemetry logging to ai_runs collection", "Capture real measured latency_ms, provider token counts, model_version, and provenance method.", "ml", 8.0, "usr-sama", "in_progress", 3, ["task-s4-01"], 16, False),
        ("task-s4-09", "Automated regression test suite for role-based access control matrix", "Verify PM, TL, Developer, and Admin roles across all API route permissions and 403 enforcement.", "qa", None, "usr-kareem", "todo", 1, [], 3, True),  # Unestimated 2, Mid-sprint 2
        ("task-s4-10", "Resolve WebSocket disconnect on mobile browsers during background tab switch", "Handle visibilitychange event to cleanly reconnect socket and replay missing board delta events.", "bug", 3.0, "usr-layla", "todo", 2, [], 2, True),  # Mid-sprint 3
        ("task-s4-11", "Setup Redis cache for project developer profile read models", "", "infra", 2.0, "usr-nader", "todo", 4, [], 9, False),  # Empty description 3
        ("task-s4-12", "Optimize suggestion inbox query with compound index on project, status, and created_at", "Explain query shows COLLSCAN on /suggestions; add compound index to support filtered pagination.", "backend", 1.0, "usr-sama", "in_progress", 1, [], 11, False),
        ("task-s4-13", "إصلاح خطأ مصادقة الدخول عبر رمز التوكن وإعادة التوجيه", "تحسين معالجة الأخطاء عند انتهاء صلاحية التوكن وإعادة توجيه المستخدم لصفحة تسجيل الدخول", "bug", None, "usr-layla", "todo", 2, [], 5, True),  # Non-English, Unestimated 3, Mid-sprint 4
        # Done tasks (3 tasks)
        ("task-s4-14", "Design and seed database schema for task snapshots", "Created Pydantic read models and MongoDB schema with validation rules.", "backend", 3.0, "usr-sama", "done", 0, [], 14, False),
        ("task-s4-15", "Configure Docker Compose for local development environment", "Containerized FastAPI service and local MongoDB instance with persistent volume mount.", "infra", 2.0, "usr-nader", "done", 0, [], 12, False),
        ("task-s4-16", "Create OpenAPI Swagger documentation for suggestion endpoints", "Annotated Pydantic models with example payloads and HTTP error response contracts.", "docs", 1.0, "usr-alex", "done", 0, [], 10, False),
    ]

    # Active Tasks for Sprint 5 (15 tasks: 12 open, 3 done; ~20% mid-sprint additions; ~13% unestimated; 1 empty description)
    s5_start = now - timedelta(days=6)
    s5_task_defs = [
        ("task-s5-01", "Design notification badge component for suggestion alerts", "Shows unread suggestion count badge with polling refresh in application header.", "frontend", 2.0, "usr-layla", "in_progress", 2, [], 12, False),
        ("task-s5-02", "Implement export endpoint for sprint audit report in CSV format", "Provides downloadable CSV dump of all sprint tasks, estimations, and carry-over telemetry.", "backend", 3.0, "usr-sama", "todo", 4, [], 14, False),
        ("task-s5-03", "Kubernetes Helm charts for high availability deployment", "Define Helm values for multi-replica FastAPI deployment with autoscaling and secret management.", "infra", 5.0, "usr-nader", "in_progress", 3, [], 16, False),
        ("task-s5-04", "Audit dead letter queue processing logic in Kafka event consumer", "Ensure poison-pill events with invalid JSON schema route to dead_letter_events without stalling consumer.", "qa", 3.0, "usr-kareem", "todo", 5, [], 8, False),
        ("task-s5-05", "Refactor sprint fit check logic into reusable domain service", "Extract capacity comparison and developer load distribution into isolated pure functions.", "refactor", None, "usr-sama", "in_progress", 1, [], 3, True),  # Unestimated 1, Mid-sprint 1
        ("task-s5-06", "Validate French and German localization for UI labels", "", "docs", 2.0, "usr-alex", "todo", 3, [], 10, False),  # Empty description 4
        ("task-s5-07", "Fine-tune Ollama prompt templates for grounded standup summaries", "Constrain LLM generation to strict facts extracted from sprint snapshot tasks without hallucinations.", "ml", 8.0, "usr-sama", "in_progress", 4, [], 15, False),
        ("task-s5-08", "Address memory leak in background scheduler job execution", "Garbage collect completed APScheduler job execution contexts to avoid creeping RSS memory growth.", "bug", 5.0, "usr-nader", "todo", 6, ["task-s5-03"], 11, False),
        ("task-s5-09", "End-to-end integration tests for suggestion decision state transitions", "Test pending -> accepted/rejected/dismissed with atomic concurrency conflict assertions.", "test", 3.0, "usr-kareem", "in_progress", 2, [], 2, True),  # Mid-sprint 2
        ("task-s5-10", "Fix CSS layout collapse on mobile table view", "Responsive CSS flexbox adjustments for small viewport screens below 768px.", "bug", 1.0, "usr-layla", "todo", 1, [], 1, True),  # Mid-sprint 3
        ("task-s5-11", "Add rate limiting headers to all public AI endpoints", "Return X-RateLimit-Limit and X-RateLimit-Remaining headers from Redis token bucket middleware.", "backend", 2.0, "usr-kareem", "in_progress", 1, [], 7, False),
        ("task-s5-12", "Benchmark LightGBM inference latency under concurrent thread load", "Measure p50, p95, and p99 inference latency with 50 concurrent requests.", "ml", None, "usr-nader", "todo", 2, [], 9, False),  # Unestimated 2
        # Done tasks (3 tasks)
        ("task-s5-13", "Setup central Winston logger format with JSON output", "Structured JSON logging for Datadog log aggregation.", "backend", 2.0, "usr-sama", "done", 0, [], 8, False),
        ("task-s5-14", "Initial CSS theme tokens and dark mode palette", "Tailwind CSS configuration with dark theme variables.", "frontend", 3.0, "usr-layla", "done", 0, [], 7, False),
        ("task-s5-15", "Document REST API endpoint response schemas", "Markdown documentation for API consumers.", "docs", 1.0, "usr-alex", "done", 0, [], 6, False),
    ]

    # Ineligible Sprint: small task count (5 tasks: 4 open, 1 done)
    s_small_defs = [
        ("task-small-01", "Audit Redis cache invalidation keys for stale sessions", "Verify TTL eviction policies and purge hanging authentication keys.", "backend", 2.0, "usr-sama", "todo", 2, [], 10, False),
        ("task-small-02", "Add unit tests for Kafka dead letter consumer replay", "Cover corrupted message payloads and exponential backoff retry logic.", "test", 1.0, "usr-kareem", "todo", 3, [], 9, False),
        ("task-small-03", "Accessibility review of modal dialog focus trap", "Verify keyboard navigation and ARIA attributes in team settings modal.", "qa", 2.0, "usr-layla", "in_progress", 1, [], 8, False),
        ("task-small-04", "Draft architecture retrospective notes for Q4 planning", "Document lessons learned from asynchronous event queue refactoring.", "docs", 1.0, "usr-alex", "in_progress", 2, [], 7, False),
        ("task-small-05", "Provision staging PostgreSQL read replica in cluster", "Configure streaming replication and automated failover health checks.", "infra", 3.0, "usr-nader", "done", 0, [], 11, False),
    ]

    # Ineligible Sprint: early sprint (f < 0.20, 13 tasks: 12 open, 1 done)
    s_early_defs = [
        ("task-early-01", "Fix null pointer exception in webhook delivery retry", "Add validation check on attempt timestamp before queueing retry job.", "bug", 2.0, "usr-sama", "todo", 1, [], 5, False),
        ("task-early-02", "Provision isolated VPC subnet for staging cluster", "Configure route tables, security groups, and NAT gateways for private worker nodes.", "infra", 3.0, "usr-nader", "in_progress", 0, [], 5, False),
        ("task-early-03", "Refactor auth middleware to support rotating JWT tokens", "Extract token verification logic into isolated stateless security service.", "refactor", 4.0, "usr-kareem", "todo", 1, [], 5, False),
        ("task-early-04", "Automated stress tests for WebSocket event broadcast", "Benchmark server latency and memory under 1000 concurrent client socket connections.", "test", 5.0, "usr-sama", "in_progress", 0, [], 5, False),
        ("task-early-05", "Write onboarding guide for third-party webhook integrations", "Document HMAC signature headers and payload schemas with runnable code examples.", "docs", 1.0, "usr-alex", "todo", 1, [], 5, False),
        ("task-early-06", "Implement batch task assignment endpoint with transactional rollback", "Support bulk assignment of sprint tasks with atomic MongoDB multi-document transactions.", "backend", 2.0, "usr-nader", "in_progress", 0, [], 5, False),
        ("task-early-07", "Build interactive burndown chart widget for sprint overview", "SVG visualization component showing ideal vs actual work progress.", "frontend", 3.0, "usr-layla", "todo", 1, [], 5, False),
        ("task-early-08", "Evaluate LightGBM tree depth limits for inference latency", "Profile inference memory footprint and p99 latency tradeoffs with max_depth=6.", "ml", 4.0, "usr-sama", "in_progress", 0, [], 5, False),
        ("task-early-09", "Verify cross-browser layout on Safari and mobile viewports", "Test responsive navigation drawers and data tables across WebKit engines.", "qa", 5.0, "usr-kareem", "todo", 1, [], 5, False),
        ("task-early-10", "Implement automated database migration runner in CI/CD pipeline", "Run MongoDB collection schema validations and index creations during deploy phase.", "feature", 1.0, "usr-alex", "in_progress", 0, [], 5, False),
        ("task-early-11", "Resolve intermittent race condition in async notification worker", "Introduce distributed locking using Redis Redlock algorithm during email dispatch.", "bug", 2.0, "usr-nader", "todo", 1, [], 5, False),
        ("task-early-12", "Configure Prometheus alert rules for API p99 latency spikes", "Trigger alerts when p99 request duration exceeds 500ms for over five minutes.", "infra", 3.0, "usr-layla", "in_progress", 0, [], 5, False),
        ("task-early-13", "Optimize Mongo aggregation pipeline for completed sprint metrics", "Use indexed match stages before group operations to minimize memory usage.", "refactor", 4.0, "usr-sama", "done", 0, [], 5, False),
    ]

    # =========================================================================
    # Process Tasks & Events and build consistent Snapshots
    # =========================================================================
    all_task_snapshots: List[Dict[str, Any]] = []
    all_task_events: List[Dict[str, Any]] = []
    all_time_entries: List[Dict[str, Any]] = []

    # 1. Historical Tasks
    for tid, title, t_type, sp_pts, assignee, sp_id, logged_h, c_days, u_days in hist_task_defs:
        c_at = now - timedelta(days=c_days)
        u_at = now - timedelta(days=u_days)
        assigned_at = c_at + timedelta(days=1)
        done_at = u_at

        # Events
        evt_assign = {
            "event_id": f"evt-{tid}-sprint",
            "task_id": tid,
            "sprint_id": sp_id,
            "type": "sprint_assigned",
            "from": None,
            "to": sp_id,
            "at": assigned_at,
            "source": "seed_demo"
        }
        evt_status_init = {
            "event_id": f"evt-{tid}-st-0",
            "task_id": tid,
            "sprint_id": sp_id,
            "type": "status_change",
            "from": None,
            "to": "todo",
            "at": c_at,
            "source": "seed_demo"
        }
        evt_status_done = {
            "event_id": f"evt-{tid}-st-1",
            "task_id": tid,
            "sprint_id": sp_id,
            "type": "status_change",
            "from": "in_progress",
            "to": "done",
            "at": done_at,
            "source": "seed_demo"
        }
        evt_usr = {
            "event_id": f"evt-{tid}-user",
            "task_id": tid,
            "sprint_id": sp_id,
            "type": "assignee_change",
            "from": None,
            "to": assignee,
            "at": assigned_at,
            "source": "seed_demo"
        }
        all_task_events.extend([evt_assign, evt_status_init, evt_status_done, evt_usr])

        all_task_snapshots.append({
            "task_id": tid,
            "project_id": PROJECT_ID,
            "sprint_id": sp_id,
            "title": title,
            "description": f"Implementation details for {title}",
            "task_type": t_type,
            "story_points": sp_pts,
            "status": "done",
            "assignee_id": assignee,
            "stalled_days": 0,
            "dependencies": [],
            "created_at": c_at,
            "updated_at": u_at,
            "sprint_assigned_at": evt_assign["at"],
            "status_changed_at": evt_status_done["at"],
            "source": "seed_demo"
        })

        all_time_entries.append({
            "entry_id": f"time-{tid}",
            "task_id": tid,
            "user_id": assignee,
            "hours": logged_h,
            "logged_at": done_at - timedelta(days=1),
            "source": "seed_demo"
        })

    # Helper for active task definition processing
    def process_active_defs(task_defs, sp_id, sp_start):
        for item in task_defs:
            tid, title, desc, t_type, sp_pts, assignee, status, stalled_d, deps, c_days_ago, is_mid_sprint = item
            c_at = now - timedelta(days=c_days_ago)

            if is_mid_sprint:
                # Assigned 1 to 3 days after sprint start
                assigned_at = sp_start + timedelta(days=2, hours=4)
            else:
                # Assigned at sprint start (or 1 day prior)
                assigned_at = sp_start - timedelta(hours=12)

            status_changed_at = now - timedelta(days=stalled_d)
            # updated_at can be slightly later after comments or commits
            updated_at = status_changed_at + timedelta(hours=2) if status_changed_at + timedelta(hours=2) <= now else now

            # Events
            evt_assign = {
                "event_id": f"evt-{tid}-sprint",
                "task_id": tid,
                "sprint_id": sp_id,
                "type": "sprint_assigned",
                "from": None,
                "to": sp_id,
                "at": assigned_at,
                "source": "seed_demo"
            }
            evt_st_init = {
                "event_id": f"evt-{tid}-st-0",
                "task_id": tid,
                "sprint_id": sp_id,
                "type": "status_change",
                "from": None,
                "to": "todo",
                "at": c_at,
                "source": "seed_demo"
            }
            events_for_task = [evt_assign, evt_st_init]

            if status != "todo":
                evt_st_curr = {
                    "event_id": f"evt-{tid}-st-1",
                    "task_id": tid,
                    "sprint_id": sp_id,
                    "type": "status_change",
                    "from": "todo",
                    "to": status,
                    "at": status_changed_at,
                    "source": "seed_demo"
                }
                events_for_task.append(evt_st_curr)

            if assignee:
                evt_usr = {
                    "event_id": f"evt-{tid}-user",
                    "task_id": tid,
                    "sprint_id": sp_id,
                    "type": "assignee_change",
                    "from": None,
                    "to": assignee,
                    "at": assigned_at,
                    "source": "seed_demo"
                }
                events_for_task.append(evt_usr)

            all_task_events.extend(events_for_task)

            # Snapshot derived strictly from events
            latest_status_evt = events_for_task[-1] if events_for_task[-1]["type"] == "status_change" else (
                events_for_task[2] if len(events_for_task) > 2 and events_for_task[2]["type"] == "status_change" else evt_st_init
            )
            derived_status_changed_at = latest_status_evt["at"]
            derived_stalled_days = max(0, int((now - derived_status_changed_at).total_seconds() / 86400))

            all_task_snapshots.append({
                "task_id": tid,
                "project_id": PROJECT_ID,
                "sprint_id": sp_id,
                "title": title,
                "description": desc,
                "task_type": t_type,
                "story_points": sp_pts,
                "status": status,
                "assignee_id": assignee,
                "stalled_days": derived_stalled_days,
                "dependencies": deps,
                "created_at": c_at,
                "updated_at": updated_at,
                "sprint_assigned_at": evt_assign["at"],
                "status_changed_at": derived_status_changed_at,
                "source": "seed_demo"
            })

            # Time entry for in_progress/done tasks
            if status in ["in_progress", "done"]:
                all_time_entries.append({
                    "entry_id": f"time-{tid}",
                    "task_id": tid,
                    "user_id": assignee,
                    "hours": 12.0 if status == "in_progress" else 24.0,
                    "logged_at": now - timedelta(days=1),
                    "source": "seed_demo"
                })

    process_active_defs(s4_task_defs, "sprint-04", s4_start)
    process_active_defs(s5_task_defs, "sprint-05", s5_start)
    process_active_defs(s_small_defs, "sprint-small-active", now - timedelta(days=7))
    process_active_defs(s_early_defs, "sprint-early-active", now - timedelta(days=1))

    # Upsert all task snapshots
    for t in all_task_snapshots:
        await db.task_snapshots.update_one(
            {"task_id": t["task_id"]},
            {"$set": t},
            upsert=True
        )

    # Upsert all task events (audit log mirror of Kafka event stream)
    for e in all_task_events:
        await db.task_events.update_one(
            {"event_id": e["event_id"]},
            {"$set": e},
            upsert=True
        )

    # Upsert time entries
    for te in all_time_entries:
        await db.time_entries.update_one(
            {"entry_id": te["entry_id"]},
            {"$set": te},
            upsert=True
        )

    # =========================================================================
    # 5. Standup Blockers (Privacy check: NO mood data)
    # =========================================================================
    blockers = [
        {
            "blocker_id": "blk-01",
            "user_id": "usr-kareem",
            "task_id": "task-s4-02",
            "description": "Waiting for Redis socket cluster credentials from DevOps. Integration tests failing.",
            "reported_at": now - timedelta(days=3),
            "status": "active",
            "source": "seed_demo"
        },
        {
            "blocker_id": "blk-02",
            "user_id": "usr-layla",
            "task_id": "task-s4-10",
            "description": "Waiting for UX team approval on mobile drawer navigation spec.",
            "reported_at": now - timedelta(days=1),
            "status": "active",
            "source": "seed_demo"
        }
    ]
    for b in blockers:
        await db.standup_blockers.update_one(
            {"blocker_id": b["blocker_id"]},
            {"$set": b},
            upsert=True
        )

    # =========================================================================
    # 6. Pull Requests & Release Requests
    # =========================================================================
    prs = [
        {
            "pr_id": "pr-101",
            "task_ids": ["task-s4-01"],
            "title": "feat(backend): async connection pooling and reconnection health check",
            "status": "open",
            "branch": "feat/connection-pool",
            "created_at": now - timedelta(days=2),
            "source": "seed_demo"
        },
        {
            "pr_id": "pr-099",
            "task_ids": ["task-hist-07", "task-hist-08"],
            "title": "feat(ml): LightGBM production artifacts for story points and risk",
            "status": "merged",
            "branch": "feat/ml-artifacts",
            "created_at": now - timedelta(days=16),
            "source": "seed_demo"
        }
    ]
    for pr in prs:
        await db.pull_requests.update_one(
            {"pr_id": pr["pr_id"]},
            {"$set": pr},
            upsert=True
        )

    releases = [
        {
            "release_request_id": "rel-v1.0",
            "project_id": PROJECT_ID,
            "version": "v1.0.0",
            "target_date": now - timedelta(days=14),
            "status": "released",
            "task_ids": [f"task-hist-{i:02d}" for i in range(1, 10)],
            "source": "seed_demo"
        },
        {
            "release_request_id": "rel-v1.1",
            "project_id": PROJECT_ID,
            "version": "v1.1.0-RC",
            "target_date": now + timedelta(days=5),
            "status": "blocked",
            "task_ids": ["task-s4-01", "task-s4-02", "task-s4-03", "task-s4-04"],
            "source": "seed_demo"
        }
    ]
    for rel in releases:
        await db.release_requests.update_one(
            {"release_request_id": rel["release_request_id"]},
            {"$set": rel},
            upsert=True
        )

    summary = {
        "reference_time": now.isoformat(),
        "profiles": len(profiles),
        "sprints": len(sprints),
        "capacities": len(capacities),
        "tasks": len(all_task_snapshots),
        "task_events": len(all_task_events),
        "time_entries": len(all_time_entries),
        "blockers": len(blockers),
        "pull_requests": len(prs),
        "release_requests": len(releases),
        "source": "seed_demo"
    }

    print("[seed_data] Demo seed completed successfully!")
    print(f"  - Reference Time:   {summary['reference_time']}")
    print(f"  - Sprints Seeded:   {summary['sprints']} (Active in range: sprint-04, sprint-05)")
    print(f"  - Tasks Seeded:     {summary['tasks']} total snapshots")
    print(f"  - Events Mirrored:  {summary['task_events']} Kafka audit events")
    print(f"  - Developer Users:  {summary['profiles']}")
    print(f"  - Privacy check:    No mood fields, zero developer ranking, source='seed_demo' tagged.\n")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Taskove Demo Dataset Seeder")
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="ISO UTC timestamp to use as reference now (e.g. 2026-11-14T12:00:00Z). Default: current UTC."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe all existing seed_demo documents before seeding."
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=None,
        help="MongoDB connection URI override."
    )
    parser.add_argument(
        "--mongo-db",
        type=str,
        default=None,
        help="MongoDB database name override."
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    ref_now = None
    if args.now:
        clean_str = args.now.replace("Z", "+00:00")
        ref_now = datetime.fromisoformat(clean_str)

    mongo_uri = args.mongo_uri or settings.MONGO_URI
    mongo_db = args.mongo_db or settings.MONGO_DB

    print(f"[seed_data] Connecting to MongoDB at {mongo_uri} (db: {mongo_db})...")
    client = AsyncIOMotorClient(mongo_uri)
    db = client[mongo_db]

    try:
        await seed_data(db, now=ref_now, reset=args.reset)
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
