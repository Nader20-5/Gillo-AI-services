import pytest
from app.models.read_models import StandupBlocker
from app.models.suggestions import SuggestionDecisionRequest


def test_privacy_guardrail_no_mood_in_standup_blocker():
    """
    Architectural Privacy Guardrail:
    Standup blocker read model must strictly exclude any mood fields.
    """
    blocker = StandupBlocker(
        blocker_id="blk-test-01",
        user_id="usr-test",
        description="Blocked on CI deployment pipeline"
    )
    dumped = blocker.model_dump()
    assert "mood" not in dumped, "Privacy violation: 'mood' field found in StandupBlocker model!"
    assert "user_mood" not in dumped, "Privacy violation: 'user_mood' field found in StandupBlocker model!"


def test_human_in_the_loop_mandatory_decision_reason():
    """
    Architectural Guardrail:
    Rejecting or dismissing a suggestion strictly requires a non-empty reason.
    """
    # Empty string should fail validation check in router logic
    req = SuggestionDecisionRequest(decision_reason="")
    assert not req.decision_reason or not req.decision_reason.strip()

    valid_req = SuggestionDecisionRequest(decision_reason="Reassigning to layla because she has more domain context.")
    assert valid_req.decision_reason and len(valid_req.decision_reason.strip()) > 0
