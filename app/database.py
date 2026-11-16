import asyncio
import logging
import re
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

logger = logging.getLogger(__name__)


def sanitize_mongo_uri(uri: str) -> str:
    """Masks credentials in MongoDB URI for safe logging."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", uri)


class Database:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None


db = Database()


async def connect_to_mongo():
    """Connect to MongoDB and initialize collections and indexes."""
    if db.client is not None and db.db is not None:
        try:
            await init_indexes()
        except Exception:
            pass
        return

    logger.info(f"Connecting to MongoDB at {sanitize_mongo_uri(settings.MONGO_URI)}...")
    try:
        db.client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=1000)
        db.db = db.client[settings.MONGO_DB]
        await init_indexes()
        logger.info("Connected to MongoDB successfully and verified indexes.")
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB at {sanitize_mongo_uri(settings.MONGO_URI)}: {e}")


async def close_mongo_connection():
    """Close MongoDB connection pool."""
    if db.client:
        try:
            if hasattr(db.client, "close"):
                db.client.close()
        except Exception:
            pass


async def init_indexes():
    """
    Initialize indexes as specified in the Taskove / Gillo Architecture:
    - ai_suggestions: {project_id, status, created_at}, {task_id, type, status},
      and a unique partial index on dedupe_key where status == "pending".
    - task_estimates: {task_id, created_at} and {sprint_id, status}.
    - chat_sessions: TTL index on last_active_at (e.g. 7 days).
    - processed_events: unique index on event_id for idempotency.
    """
    database = db.db

    # 1. ai_suggestions indexes
    await database.ai_suggestions.create_index(
        [("project_id", 1), ("status", 1), ("created_at", -1)],
        name="idx_suggestions_project_status_created"
    )
    await database.ai_suggestions.create_index(
        [("task_id", 1), ("type", 1), ("status", 1)],
        name="idx_suggestions_task_type_status"
    )
    # Unique partial index on dedupe_key when status is "pending"
    await database.ai_suggestions.create_index(
        [("dedupe_key", 1)],
        unique=True,
        partialFilterExpression={"status": "pending"},
        name="uniq_pending_dedupe_key"
    )

    # 2. task_estimates indexes
    await database.task_estimates.create_index(
        [("task_id", 1), ("created_at", -1)],
        name="idx_estimates_task_created"
    )
    await database.task_estimates.create_index(
        [("sprint_id", 1), ("status", 1)],
        name="idx_estimates_sprint_status"
    )

    # 3. chat_sessions TTL index (expire after 7 days of inactivity = 604800 seconds)
    await database.chat_sessions.create_index(
        [("last_active_at", 1)],
        expireAfterSeconds=604800,
        name="ttl_chat_sessions_last_active"
    )

    # 4. processed_events idempotency index
    await database.processed_events.create_index(
        [("event_id", 1)],
        unique=True,
        name="uniq_processed_event_id"
    )

    # 5. Read models helpful query indexes
    await database.task_snapshots.create_index([("task_id", 1)], unique=True, name="uniq_task_snapshot_id")
    await database.task_snapshots.create_index([("sprint_id", 1)], name="idx_tasks_sprint_id")
    await database.sprint_snapshots.create_index([("sprint_id", 1)], unique=True, name="uniq_sprint_snapshot_id")
    await database.time_entries.create_index([("task_id", 1)], name="idx_time_entries_task_id")
    await database.task_events.create_index([("event_id", 1)], unique=True, name="uniq_task_event_id")
    await database.task_events.create_index([("task_id", 1), ("at", 1)], name="idx_task_events_task_at")


def get_database() -> AsyncIOMotorDatabase:
    """Dependency helper to get the active database instance."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    needs_new = False
    if db.client is None or db.db is None:
        needs_new = True
    elif loop and hasattr(db.client, "io_loop") and db.client.io_loop.is_closed():
        needs_new = True

    if needs_new:
        db.client = AsyncIOMotorClient(settings.MONGO_URI)
        db.db = db.client[settings.MONGO_DB]
    return db.db
