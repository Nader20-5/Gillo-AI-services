from pathlib import Path
import pytest
from app.utils.privacy import FORBIDDEN_RANKING_TERMS, strip_mood_fields, contains_mood
from app.events.consumer import EventConsumer, AgileEvent
from app.models.suggestions import AISuggestionBase
from app.models.estimates import SprintFitCheckModel

PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


def test_no_ranking_in_prompt_templates():
    """
    Ethical Guardrail:
    Prompt templates must never instruct the model to rank, score, or compare developers.
    """
    assert PROMPTS_ROOT.exists(), f"Prompts directory {PROMPTS_ROOT} must exist"
    for md_file in PROMPTS_ROOT.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_RANKING_TERMS:
            assert term not in content, (
                f"Ethical violation: prohibited term '{term}' found in prompt template {md_file}!"
            )


def test_no_ranking_fields_in_response_schemas():
    """
    Ethical Guardrail:
    Response models must never include developer score or ranking fields.
    """
    schemas_to_check = [AISuggestionBase, SprintFitCheckModel]
    for schema in schemas_to_check:
        fields = schema.model_fields.keys()
        for term in FORBIDDEN_RANKING_TERMS:
            assert term not in fields, f"Ethical violation: prohibited field '{term}' in {schema.__name__}"


@pytest.mark.asyncio
async def test_mood_fields_stripped_by_consumer(setup_test_db):
    """
    Privacy Guardrail:
    Event consumer must strip all mood, morale, and sentiment fields before writing to DB.
    """
    consumer = EventConsumer(setup_test_db)

    raw_event = {
        "event_id": "evt-privacy-01",
        "event_type": "StandupSubmitted",
        "payload": {
            "developer": "usr-kareem",
            "mood": "frustrated",
            "happiness_score": 2,
            "blockers": [
                {
                    "blocker_id": "blk-priv-01",
                    "description": "Waiting on credentials",
                    "user_mood": "stressed",
                    "sentiment": "negative"
                }
            ]
        }
    }

    res = await consumer.process_raw_event(raw_event)
    assert res["status"] == "processed"

    # Verify standup_blockers collection
    saved_blocker = await setup_test_db.standup_blockers.find_one({"blocker_id": "blk-priv-01"})
    assert saved_blocker is not None
    assert "user_mood" not in saved_blocker
    assert "sentiment" not in saved_blocker
    assert not contains_mood(saved_blocker), "Database document contains forbidden mood keys!"
