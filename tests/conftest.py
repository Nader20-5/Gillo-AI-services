import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
import pytest
from mongomock_motor import AsyncMongoMockClient

from app.main import app
from app.database import get_database, db, init_indexes
from app.config import settings

# Test mock user constants
HEADERS_DEV = {
    "x-mock-user-id": "usr-sama",
    "x-mock-role": "Developer",
    "x-mock-project-id": "proj-gillo-core"
}

HEADERS_TL = {
    "x-mock-user-id": "usr-nader",
    "x-mock-role": "TL",
    "x-mock-project-id": "proj-gillo-core"
}

HEADERS_PM = {
    "x-mock-user-id": "usr-alex",
    "x-mock-role": "PM",
    "x-mock-project-id": "proj-gillo-core"
}

HEADERS_ADMIN = {
    "x-mock-user-id": "usr-admin",
    "x-mock-role": "Admin",
    "x-mock-project-id": "proj-gillo-core"
}


async def seed_test_database(database):
    """Seeds in-memory database with realistic test agile data."""
    now = datetime.now(timezone.utc)
    project_id = "proj-gillo-core"

    # Profiles
    await database.ai_profiles.insert_many([
        {"user_id": "usr-nader", "name": "Nader", "role": "TL", "estimation": {"ratio": 1.05, "sample_size": 20}, "skills": ["Python", "FastAPI"]},
        {"user_id": "usr-sama", "name": "Sama", "role": "Developer", "estimation": {"ratio": 0.95, "sample_size": 25}, "skills": ["Python", "PyTorch"]},
        {"user_id": "usr-alex", "name": "Alex", "role": "PM", "estimation": {"ratio": 1.0, "sample_size": 15}, "skills": ["Agile", "Scrum"]},
        {"user_id": "usr-kareem", "name": "Kareem", "role": "Developer", "estimation": {"ratio": 1.30, "sample_size": 18}, "skills": ["Docker", "Kafka"]},
        {"user_id": "usr-layla", "name": "Layla", "role": "Developer", "estimation": {"ratio": 0.90, "sample_size": 22}, "skills": ["React", "UI"]},
    ])

    # Sprints
    await database.sprint_snapshots.insert_many([
        {
            "sprint_id": "sprint-01",
            "project_id": project_id,
            "name": "Sprint 1",
            "start_date": now - timedelta(days=30),
            "end_date": now - timedelta(days=16),
            "status": "completed",
        },
        {
            "sprint_id": "sprint-04",
            "project_id": project_id,
            "name": "Sprint 4",
            "start_date": now - timedelta(days=7),
            "end_date": now + timedelta(days=7),
            "status": "active",
        }
    ])

    # Capacity
    await database.capacity.insert_many([
        {"user_id": "usr-nader", "sprint_id": "sprint-04", "capacity_hours": 60.0},
        {"user_id": "usr-sama", "sprint_id": "sprint-04", "capacity_hours": 60.0},
        {"user_id": "usr-kareem", "sprint_id": "sprint-04", "capacity_hours": 40.0},
        {"user_id": "usr-layla", "sprint_id": "sprint-04", "capacity_hours": 60.0},
    ])

    # Tasks
    await database.task_snapshots.insert_many([
        # Historical completed tasks for baseline calculation
        {
            "task_id": "task-hist-01",
            "project_id": project_id,
            "sprint_id": "sprint-01",
            "title": "Historical Backend Task",
            "task_type": "backend",
            "story_points": 3.0,
            "status": "done",
            "assignee_id": "usr-sama",
            "created_at": now - timedelta(days=28),
            "updated_at": now - timedelta(days=18),
        },
        {
            "task_id": "task-hist-02",
            "project_id": project_id,
            "sprint_id": "sprint-01",
            "title": "Historical ML Task",
            "task_type": "ml",
            "story_points": 5.0,
            "status": "done",
            "assignee_id": "usr-sama",
            "created_at": now - timedelta(days=27),
            "updated_at": now - timedelta(days=17),
        },
        # Active tasks
        {
            "task_id": "task-act-01",
            "project_id": project_id,
            "sprint_id": "sprint-04",
            "title": "Build Suggestion Inbox API",
            "task_type": "backend",
            "story_points": 5.0,
            "status": "in_progress",
            "assignee_id": "usr-sama",
            "stalled_days": 1,
            "dependencies": [],
            "created_at": now - timedelta(days=5),
            "updated_at": now - timedelta(hours=2),
        },
        {
            "task_id": "task-act-02",
            "project_id": project_id,
            "sprint_id": "sprint-04",
            "title": "Model 1 Calibration",
            "task_type": "ml",
            "story_points": 8.0,
            "status": "in_progress",
            "assignee_id": "usr-kareem",
            "stalled_days": 4,
            "dependencies": ["task-act-01"],
            "created_at": now - timedelta(days=6),
            "updated_at": now - timedelta(hours=3),
        },
        {
            "task_id": "task-act-03",
            "project_id": project_id,
            "sprint_id": "sprint-04",
            "title": "WebSocket Blocker Service",
            "task_type": "backend",
            "story_points": 5.0,
            "status": "blocked",
            "assignee_id": "usr-kareem",
            "stalled_days": 6,
            "dependencies": ["task-act-01"],
            "created_at": now - timedelta(days=6),
            "updated_at": now - timedelta(hours=1),
        }
    ])

    # Time entries
    await database.time_entries.insert_many([
        {"entry_id": "te-01", "task_id": "task-hist-01", "user_id": "usr-sama", "hours": 15.0, "logged_at": now - timedelta(days=20)},
        {"entry_id": "te-02", "task_id": "task-hist-02", "user_id": "usr-sama", "hours": 24.0, "logged_at": now - timedelta(days=19)},
        {"entry_id": "te-03", "task_id": "task-act-01", "user_id": "usr-sama", "hours": 10.0, "logged_at": now - timedelta(days=1)},
        {"entry_id": "te-04", "task_id": "task-act-03", "user_id": "usr-kareem", "hours": 25.0, "logged_at": now - timedelta(days=1)},
    ])

    # Blockers
    await database.standup_blockers.insert_many([
        {
            "blocker_id": "blk-01",
            "user_id": "usr-kareem",
            "task_id": "task-act-03",
            "description": "Waiting for Redis socket credentials from DevOps.",
            "status": "active",
            "reported_at": now - timedelta(days=3),
        }
    ])

    # Releases & PRs
    await database.release_requests.insert_many([
        {
            "release_request_id": "rel-v1.0",
            "project_id": project_id,
            "version": "v1.0.0",
            "status": "released",
            "task_ids": ["task-hist-01", "task-hist-02"],
        },
        {
            "release_request_id": "rel-v1.1",
            "project_id": project_id,
            "version": "v1.1.0-RC",
            "status": "blocked",
            "task_ids": ["task-act-01", "task-act-02", "task-act-03"],
        }
    ])
    await database.pull_requests.insert_many([
        {
            "pr_id": "pr-101",
            "task_ids": ["task-act-01"],
            "title": "feat: API endpoints for suggestion decisions",
            "status": "open",
            "branch": "feat/inbox-api",
        }
    ])


@pytest.fixture(autouse=True)
async def setup_test_db():
    """Initializes in-memory mock client and seeds it for each test."""
    mock_client = AsyncMongoMockClient()
    mock_database = mock_client["gillo_ai_test"]

    # Seed
    await seed_test_database(mock_database)

    # Patch database singletons
    db.client = mock_client
    db.db = mock_database
    await init_indexes()
    app.dependency_overrides[get_database] = lambda: mock_database

    # Enable mock authentication for test fixtures
    orig_allow_mock_auth = settings.ALLOW_MOCK_AUTH
    settings.ALLOW_MOCK_AUTH = True

    yield mock_database

    settings.ALLOW_MOCK_AUTH = orig_allow_mock_auth
    app.dependency_overrides.clear()
    mock_client.close()
