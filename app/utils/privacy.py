from typing import Any, Dict, List
import re

MOOD_KEYS = {"mood", "mood_rating", "user_mood", "sentiment", "morale", "emotional_state", "happiness_score"}

FORBIDDEN_RANKING_TERMS = [
    "developer_rank",
    "developer_score",
    "performance_score",
    "rank_developer",
    "best_developer",
    "worst_developer",
    "developer_rating",
    "compare_performance",
]


def strip_mood_fields(obj: Any) -> Any:
    """
    Recursively strips any mood, sentiment, or emotional state fields
    from dictionaries and lists before persistence or prompt construction.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k.lower() in MOOD_KEYS:
                continue
            cleaned[k] = strip_mood_fields(v)
        return cleaned
    elif isinstance(obj, list):
        return [strip_mood_fields(item) for item in obj]
    return obj


def contains_mood(obj: Any) -> bool:
    """Checks whether an object contains any prohibited mood keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in MOOD_KEYS:
                return True
            if contains_mood(v):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if contains_mood(item):
                return True
    return False
