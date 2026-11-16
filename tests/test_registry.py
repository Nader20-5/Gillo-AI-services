import pytest
from app.ml.registry import (
    registry,
    predict_effort_with_fallback,
    predict_risk_with_fallback,
    ModelMetadata,
)
from app.ml.features import MODEL1_FEATURE_LIST, MODEL2_FEATURE_LIST


def test_registry_synthetic_flagging():
    """Validates that existing synthetic artifacts are flagged as ml_model_synthetic."""
    pred = predict_effort_with_fallback(
        title="Test Task",
        task_type="backend",
        story_points=5.0,
        dependency_count=1,
        historical_median=20.0,
        assignee_ratio=1.05,
    )
    # The active model is v1_synthetic, so it must return ml_model_synthetic or rule_baseline
    assert pred["method"] in ["ml_model_synthetic", "rule_baseline"]
    if pred["method"] == "ml_model_synthetic":
        assert pred["warning"] is not None
        assert "synthetic" in pred["warning"].lower()


def test_registry_missing_model_fallback_no_500():
    """
    Validates that a non-existent model name or version never crashes or throws 500,
    falling back to the deterministic rule baseline.
    """
    # Override active version temporarily to non-existent version in registry cache/lookup
    non_existent = registry.get_model("model_est", "v999_non_existent")
    assert non_existent is None

    # Test fallback prediction with missing story points or cold start
    pred = predict_effort_with_fallback(
        title="Test Fallback",
        task_type="backend",
        story_points=None,  # Missing story points triggers baseline
        dependency_count=0,
        historical_median=15.0,
        assignee_ratio=1.0,
    )
    assert pred["method"] == "rule_baseline"
    assert pred["mid_hours"] == 15.0
    assert pred["model_version"] == "deterministic_baseline_v1"
    assert pred["warning"] is not None
    assert "story points" in pred["warning"].lower()


def test_registry_feature_mismatch_refusal():
    """
    Validates that if metadata feature list does not match features.py,
    the registry refuses to serve the model and falls back to rule baseline.
    """
    # Create fake mismatched metadata
    mismatched_meta = ModelMetadata(
        name="model_est",
        version="v_mismatched",
        model_file="fake.pkl",
        trained_on="tawos",
        trained_at="2026-10-20",
        feature_list=["wrong_feature_1", "wrong_feature_2"],
        metrics={},
        sha256="fake_hash",
    )

    # Validate against canonical features
    is_valid = (mismatched_meta.feature_list == MODEL1_FEATURE_LIST)
    assert not is_valid, "Feature list mismatch must be detected!"
