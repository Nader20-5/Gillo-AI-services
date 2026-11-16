import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError
from pymongo.errors import DuplicateKeyError

from app.utils.privacy import strip_mood_fields, contains_mood
from app.jobs.sprint_jobs import refresh_developer_ratios

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgileEvent(BaseModel):
    event_id: str = Field(..., description="Unique event identifier for idempotency")
    event_type: str = Field(..., description="Type of event, e.g. StandupSubmitted, SprintClosed")
    aggregate_id: Optional[str] = None
    project_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)
    payload: Dict[str, Any] = Field(default_factory=dict)


class EventConsumer:
    """
    Kafka / Event Consumer with strict idempotency and fault tolerance:
    1. Validates each incoming event with Pydantic schema.
    2. Enforces idempotency via db.processed_events (unique event_id).
    3. Traps malformed events into db.dead_letter_events (no crash loop).
    4. Strips all mood keys before writing to DB.
    """

    def __init__(self, db):
        self.db = db

    async def process_raw_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point for consuming a raw event dict."""
        # 1. Schema Validation
        try:
            event = AgileEvent.model_validate(raw_event)
        except ValidationError as val_err:
            logger.error(f"[DeadLetter] Malformed event: {val_err}")
            await self.db.dead_letter_events.insert_one({
                "raw_event": raw_event,
                "error": str(val_err),
                "failed_at": utc_now(),
            })
            return {"status": "dead_letter", "reason": "schema_validation_error"}

        # 2. Idempotency Check via processed_events unique index
        try:
            await self.db.processed_events.insert_one({
                "event_id": event.event_id,
                "event_type": event.event_type,
                "processed_at": utc_now(),
            })
        except DuplicateKeyError:
            logger.info(f"[Idempotency] Event {event.event_id} already processed. Skipping duplicate.")
            return {"status": "duplicate_skipped", "event_id": event.event_id}

        # 3. Dispatch by event_type
        try:
            if event.event_type == "StandupSubmitted":
                await self._handle_standup_submitted(event)
            elif event.event_type == "SprintClosed":
                await self._handle_sprint_closed(event)
            else:
                logger.info(f"Unhandled event type '{event.event_type}'. Marked processed.")

            return {"status": "processed", "event_id": event.event_id}
        except Exception as exc:
            logger.error(f"[Consumer Error] Failed processing event {event.event_id}: {exc}", exc_info=True)
            # Route to dead letter rather than crashing
            await self.db.dead_letter_events.insert_one({
                "event_id": event.event_id,
                "event_type": event.event_type,
                "payload": event.payload,
                "error": str(exc),
                "failed_at": utc_now(),
            })
            return {"status": "error_routed_to_dead_letter", "error": str(exc)}

    async def _handle_standup_submitted(self, event: AgileEvent):
        """Processes submitted standup; ensures mood is stripped before DB write."""
        raw_payload = event.payload
        # Guardrail: strip mood fields before any DB write
        cleaned_payload = strip_mood_fields(raw_payload)

        # Store blockers if present
        blockers = cleaned_payload.get("blockers", [])
        for b in blockers:
            if isinstance(b, dict):
                cleaned_b = strip_mood_fields(b)
                cleaned_b["event_id"] = event.event_id
                cleaned_b["recorded_at"] = utc_now()
                await self.db.standup_blockers.insert_one(cleaned_b)

    async def _handle_sprint_closed(self, event: AgileEvent):
        """Sprint close event triggers developer estimation ratio refresh."""
        logger.info(f"SprintClosed event {event.event_id} received. Refreshing developer ratios.")
        await refresh_developer_ratios(self.db)

