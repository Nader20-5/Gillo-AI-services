"""
Model Registry & Fallback Execution Engine for Gillo AI Service.
Enforces:
1. Versioned storage in models/artifacts/<name>/<version>/ with metadata.json.
2. SHA256 integrity check and feature list contract validation against features.py.
3. Fallback chain: Active ML Model -> Deterministic Rule Baseline. Never HTTP 500.
4. Method tagging: 'lightgbm_taskove_v2' vs 'ml_model' vs 'rule_baseline'.
"""
import os
import sys
import json
import math
import time
import pickle
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import sparse

from app.config import settings
from app.ml.features import (
    MODEL1_FEATURE_LIST,
    MODEL2_FEATURE_LIST,
    MODEL2_SYNTHETIC_FEATURE_LIST,
    IneligibleSprintError,
    MissingFeatureError,
    build_model2_task_features,
)
from app.ml.taskove_types import (
    map_taskove_type_to_tawos,
    is_mostly_non_latin,
    snap_to_fibonacci,
    DEFAULT_TAWOS_TYPE,
)
from app.models.read_models import TaskSnapshot, SprintSnapshot

logger = logging.getLogger(__name__)

# Canonical input field order for Model 1
MODEL1_ORDERED_INPUT_FIELDS: List[str] = ["title", "description", "type"]

# Ensure benchmark_v2 is importable for unpickling TaskoveFeatureTransformer if needed
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
BENCHMARK_DIR = os.path.join(ROOT_DIR, "benchmark_v2")
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)


class ModelMetadata(BaseModel):
    name: Optional[str] = None
    model_name: Optional[str] = None
    version: str
    model_file: Optional[str] = None
    trained_on: str = Field(default="tawos", description="'synthetic', 'tawos', 'other'")
    trained_at: Optional[str] = None
    dataset_variant: Optional[str] = None
    feature_list: Optional[List[str]] = None
    ordered_feature_list: Optional[List[str]] = None
    ordered_input_fields: Optional[List[str]] = None
    test_metrics: Optional[Dict[str, Any]] = None
    conformal_interval: Optional[Dict[str, Any]] = None
    fallback_baseline: Optional[Dict[str, Any]] = None
    feature_count: Optional[int] = None
    library_versions: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metrics_test_f50: Dict[str, Any] = Field(default_factory=dict)
    sha256: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    git_commit: Optional[str] = None
    description: Optional[str] = None
    note: Optional[str] = None

    @property
    def effective_name(self) -> str:
        return self.model_name or self.name or "unknown_model"

    @property
    def effective_feature_list(self) -> List[str]:
        return self.ordered_feature_list or self.feature_list or []

    @property
    def effective_metrics(self) -> Dict[str, Any]:
        return self.test_metrics or self.metrics_test_f50 or self.metrics or {}


class LoadedModel:
    def __init__(self, model_obj: Any, metadata: ModelMetadata):
        self.model_obj = model_obj
        self.metadata = metadata
        self.is_synthetic = (metadata.trained_on == "synthetic")


class LoadedModel1Taskove:
    """Wrapper for Model 1 Taskove artifacts (LightGBM TF-IDF + Type Regressor, Quantiles, Vectorizer, Encoder)."""
    def __init__(
        self,
        model_final: Any,
        model_q25: Any,
        model_q50: Any,
        model_q75: Any,
        tfidf: Any,
        ohe: Any,
        metadata: ModelMetadata,
        fallback_baseline: Dict[str, Any]
    ):
        self.model_final = model_final
        self.model_q25 = model_q25
        self.model_q50 = model_q50
        self.model_q75 = model_q75
        self.tfidf = tfidf
        self.ohe = ohe
        self.metadata = metadata
        self.fallback_baseline = fallback_baseline
        self.is_synthetic = (metadata.trained_on == "synthetic")

        conf_meta = metadata.conformal_interval or {}
        # Service strictly uses out-of-project cold-start conformal constant (Step 0a)
        self.c_cold_start = float(conf_meta.get("c_cold_start", conf_meta.get("cold_start", {}).get("c_cold_start", 0.1147)))

    def predict(
        self,
        title: str,
        description: str = "",
        task_type: str = "feature",
        field_order: Optional[List[str]] = None,
        tawos_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes prediction with ordered input contract check and Fibonacci snapping.
        """
        if field_order is not None and list(field_order) != MODEL1_ORDERED_INPUT_FIELDS:
            raise ValueError(f"Input field contract mismatch: expected {MODEL1_ORDERED_INPUT_FIELDS}, got {field_order}")

        if tawos_type is None:
            tawos_type = map_taskove_type_to_tawos(task_type)
        text = (str(title or "") + " " + str(description or ""))[:2000]

        X_text = self.tfidf.transform([text])
        X_cat = self.ohe.transform(pd.DataFrame([{"Type": tawos_type}]))
        X = sparse.hstack([X_text, X_cat], format="csr").astype(np.float32)

        raw_sp = float(np.clip(self.model_final.predict(X)[0], 0.5, 100.0))
        snapped_sp = snap_to_fibonacci(raw_sp)

        q25 = float(np.clip(self.model_q25.predict(X)[0], 0.5, 100.0))
        q75 = float(np.clip(self.model_q75.predict(X)[0], 0.5, 100.0))

        # Built from cold-start constant c_cold_start
        low = max(0.5, round(q25 - self.c_cold_start, 1))
        high = max(low + 0.5, round(q75 + self.c_cold_start, 1))

        return {
            "story_points": snapped_sp,
            "raw_story_points": raw_sp,
            "interval": [low, high],
            "method": "lightgbm_taskove_v2",
            "model_version": self.metadata.version,
            "confidence_note": "approximate range (rough guide for new teams; improves as your team's history grows)",
            "fallback_reason": None
        }


class LoadedModel2Taskove:
    """Wrapper for Model 2 Taskove artifacts (LightGBM + Isotonic Calibrator + Transformer)."""
    def __init__(self, model: Any, calibrator: Any, transformer: Any, metadata: ModelMetadata):
        self.model = model
        self.calibrator = calibrator
        self.transformer = transformer
        self.metadata = metadata
        self.is_synthetic = (metadata.trained_on == "synthetic")

    def predict_calibrated_probabilities(self, X_df: pd.DataFrame) -> np.ndarray:
        """Computes raw probabilities then calibrates via Isotonic calibrator."""
        raw_probs = self.model.predict_proba(X_df[MODEL2_FEATURE_LIST])[:, 1]
        cal_probs = self.calibrator.predict(raw_probs)
        return cal_probs

    def compute_task_contributions(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Computes TreeSHAP margin contributions for all features (pred_contrib=True).
        Returns array of shape (N, 24): contributions for 23 features + bias.
        """
        return self.model.predict(X_df[MODEL2_FEATURE_LIST], pred_contrib=True)

    def get_top_features(self, top_n: int = 3) -> List[str]:
        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
            indices = np.argsort(imps)[::-1][:top_n]
            return [MODEL2_FEATURE_LIST[i] for i in indices]
        return ["days_in_current_status", "status_category_is_todo", "added_after_sprint_start"]


SPRINT_LEVEL_FEATURES = {
    "f",
    "sprint_open_tasks",
    "sprint_open_points",
    "fraction_points_done",
}


EXPLANATION_NOTE = (
    "These factors show what the model weighed on similar past sprints; "
    "they are hints, not confirmed causes."
)

ACTIONABLE_RAISING_FEATURES = {
    "status_category_is_todo",
    "days_in_current_status",
    "days_since_last_event",
    "stalled_days",
    "added_after_sprint_start",
    "assignee_present",
    "links_depends_on",
    "story_points",
}


def explain_feature_contribution(
    feature: str,
    contrib_val: float,
    raw_val: Any,
    task: Optional[TaskSnapshot] = None
) -> Dict[str, str]:
    """
    Transforms a task/assignee-level feature and its raw margin contribution
    into a structured explanation item: {feature, direction, plain_text}.
    Zero developer ranking; describes task situation.
    Never exposes numeric probabilities or contribution magnitudes.
    """
    direction = "raises" if contrib_val > 0 else "lowers"

    if feature == "added_after_sprint_start":
        text = "Added after sprint start" if float(raw_val or 0) == 1.0 else "Planned before sprint start"
    elif feature == "status_category_is_todo":
        text = "Not started yet" if float(raw_val or 0) == 1.0 else "Work in progress"
    elif feature == "days_in_current_status":
        days = int(round(float(raw_val or 0)))
        unit = "day" if days == 1 else "days"
        if days == 0:
            text = "Status updated today"
        elif direction == "raises":
            text = f"In current status for {days} {unit}"
        else:
            text = f"Status updated {days} {unit} ago"
    elif feature == "days_since_last_event":
        days = int(round(float(raw_val or 0)))
        unit = "day" if days == 1 else "days"
        if days == 0:
            text = "Active today"
        elif direction == "raises":
            text = f"No activity for {days} {unit}"
        else:
            text = f"Recent activity {days} {unit} ago"
    elif feature == "has_sp":
        text = "Unestimated" if float(raw_val or 0) == 0.0 else "Has story points"
    elif feature == "story_points":
        if raw_val is None or float(raw_val) == 0.0:
            text = "Unestimated"
        elif direction == "raises":
            text = f"Estimate above the sprint median ({float(raw_val):g} SP)"
        else:
            text = f"Small estimate ({float(raw_val):g} SP)"
    elif feature == "assignee_present":
        text = "Unassigned" if float(raw_val or 0) == 0.0 else "Assigned"
    elif feature == "links_depends_on":
        deps = int(round(float(raw_val or 0)))
        unit = "prerequisite dependency" if deps == 1 else "prerequisite dependencies"
        if deps > 0:
            text = f"Has {deps} {unit}"
        else:
            text = "No dependencies"
    elif feature == "assignee_open_tasks":
        n_tasks = int(round(float(raw_val or 0)))
        unit = "task" if n_tasks == 1 else "tasks"
        text = f"Assignee has {n_tasks} open {unit} in sprint"
    elif feature == "assignee_open_points":
        n_pts = float(raw_val or 0.0)
        text = f"Assignee has {n_pts:g} open points in sprint"
    elif feature.startswith("type_"):
        tname = feature.replace("type_", "")
        text = f"Task type: {tname}"
    else:
        text = feature.replace("_", " ").capitalize()

    return {
        "feature": feature,
        "direction": direction,
        "plain_text": text
    }


class ModelRegistry:
    """Manages versioned ML artifacts and ensures feature contract validation."""
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(ROOT_DIR, settings.MODEL_REGISTRY_DIR)
        self._cache: Dict[str, Any] = {}

    def _verify_sha256(self, file_path: str, expected_hash: str) -> bool:
        if not os.path.exists(file_path):
            return False
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest().lower() == expected_hash.lower()

    def _find_model_dir(self, model_name: str, version: str) -> Optional[str]:
        """Searches configured service registry directory (models/artifacts) for the artifact folder.
        Runtime strictly serves from models/artifacts/ and never loads from benchmark_v2/."""
        possible_dirs = [
            os.path.join(self.base_dir, model_name, version),
            os.path.join(self.base_dir, "model_sp", version),
            os.path.join(self.base_dir, "model_sp_taskove", version),
            os.path.join(self.base_dir, "model1_sp", version),
            os.path.join(self.base_dir, "model1", version),
            os.path.join(self.base_dir, "model_risk", version),
            os.path.join(self.base_dir, "model2", version),
        ]
        for p in possible_dirs:
            if os.path.exists(os.path.join(p, "metadata.json")):
                return p
        return None

    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """
        Loads and validates a model from the registry.
        Refuses to serve if feature contract is violated or sha256 mismatch occurs.
        """
        if model_name in ["model_sp_taskove", "model_sp", "model_est", "model1"]:
            version = version or settings.MODEL1_ACTIVE_VERSION
        else:
            version = version or settings.MODEL2_ACTIVE_VERSION

        if not version or version.lower() == "none":
            return None

        cache_key = f"{model_name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        model_dir = self._find_model_dir(model_name, version)
        if not model_dir:
            logger.warning(f"Registry metadata not found for '{model_name}:{version}'. Defaulting to baseline.")
            return None

        metadata_path = os.path.join(model_dir, "metadata.json")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                raw_meta = json.load(f)
            metadata = ModelMetadata(**raw_meta)
        except Exception as e:
            logger.error(f"Failed to parse metadata for {model_name}:{version}: {e}")
            return None

        # 1. Model 1 Taskove multi-artifact loading
        if version == "v2_taskove" and model_name in ["model_sp_taskove", "model_sp", "model_est", "model1"]:
            # Feature contract validation
            if (metadata.ordered_input_fields or []) != MODEL1_ORDERED_INPUT_FIELDS:
                logger.error(
                    f"Feature contract mismatch for {model_name}:{version}! "
                    f"Registry metadata ordered fields: {metadata.ordered_input_fields} vs Canonical: {MODEL1_ORDERED_INPUT_FIELDS}. "
                    f"Refusing to load incompatible model."
                )
                return None

            # Multi-artifact verification (LightGBM final, Q25, Q50, Q75, TF-IDF, OHE, fallback baseline)
            for art_key, expected_hash in metadata.artifacts.items():
                if art_key.endswith("_sha256"):
                    base_name = art_key[:-7]
                    if os.path.exists(os.path.join(model_dir, f"{base_name}.json")):
                        fname = f"{base_name}.json"
                    else:
                        fname = f"{base_name}.pkl"
                else:
                    fname = art_key
                fpath = os.path.join(model_dir, fname)
                if not os.path.exists(fpath):
                    logger.error(f"Required artifact file missing: {fpath}")
                    return None
                if not self._verify_sha256(fpath, expected_hash):
                    logger.error(f"Integrity check failed for {fpath}. SHA-256 mismatch!")
                    return None

            try:
                with open(os.path.join(model_dir, "model_sp_final.pkl"), "rb") as f:
                    model_final = pickle.load(f)
                with open(os.path.join(model_dir, "model_sp_q25.pkl"), "rb") as f:
                    model_q25 = pickle.load(f)
                with open(os.path.join(model_dir, "model_sp_q50.pkl"), "rb") as f:
                    model_q50 = pickle.load(f)
                with open(os.path.join(model_dir, "model_sp_q75.pkl"), "rb") as f:
                    model_q75 = pickle.load(f)
                with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "rb") as f:
                    tfidf = pickle.load(f)
                with open(os.path.join(model_dir, "ohe_encoder.pkl"), "rb") as f:
                    ohe = pickle.load(f)

                fallback_path = os.path.join(model_dir, "type_median_fallback.json")
                if os.path.exists(fallback_path):
                    with open(fallback_path, "r", encoding="utf-8") as f:
                        fallback_baseline = json.load(f)
                else:
                    fallback_baseline = metadata.fallback_baseline or {}

                loaded = LoadedModel1Taskove(
                    model_final=model_final,
                    model_q25=model_q25,
                    model_q50=model_q50,
                    model_q75=model_q75,
                    tfidf=tfidf,
                    ohe=ohe,
                    metadata=metadata,
                    fallback_baseline=fallback_baseline
                )
                self._cache[cache_key] = loaded
                logger.info(f"Loaded Model 1 {model_name}:{version} successfully (trained_on={metadata.trained_on}).")
                return loaded
            except Exception as e:
                logger.error(f"Error unpickling Model 1 artifacts for {model_name}:{version}: {e}")
                return None

        # Feature contract validation for legacy or Model 2
        if model_name in ["model_est", "model1"]:
            expected_features = MODEL1_FEATURE_LIST
        elif version == "v1_synthetic":
            expected_features = MODEL2_SYNTHETIC_FEATURE_LIST
        else:
            expected_features = MODEL2_FEATURE_LIST

        if metadata.effective_feature_list != expected_features:
            logger.error(
                f"Feature contract mismatch for {model_name}:{version}! "
                f"Registry metadata features: {metadata.effective_feature_list} vs Canonical features: {expected_features}. "
                f"Refusing to load incompatible model."
            )
            return None

        # 2. Model 2 Taskove multi-artifact loading
        if metadata.artifacts and (model_name in ["model2_sprint_carryover_risk_taskove", "model_risk", "model2"]):
            # Multi-artifact verification (LightGBM, Calibrator, Transformer)
            for art_key, expected_hash in metadata.artifacts.items():
                fname = art_key.replace("_sha256", ".pkl")
                fpath = os.path.join(model_dir, fname)
                if not os.path.exists(fpath):
                    logger.error(f"Required artifact file missing: {fpath}")
                    return None
                if not self._verify_sha256(fpath, expected_hash):
                    logger.error(f"Integrity check failed for {fpath}. SHA-256 mismatch!")
                    return None

            try:
                # Import features_v2 module for transformer unpickling
                try:
                    import features_v2
                except ImportError:
                    pass

                with open(os.path.join(model_dir, "model2_taskove_lightgbm.pkl"), "rb") as f:
                    lgbm_model = pickle.load(f)
                with open(os.path.join(model_dir, "model2_taskove_calibrator.pkl"), "rb") as f:
                    cal_dict = pickle.load(f)
                    calibrator = cal_dict["calibrator"]
                with open(os.path.join(model_dir, "model2_taskove_transformer.pkl"), "rb") as f:
                    transformer = pickle.load(f)

                loaded = LoadedModel2Taskove(
                    model=lgbm_model,
                    calibrator=calibrator,
                    transformer=transformer,
                    metadata=metadata
                )
                self._cache[cache_key] = loaded
                logger.info(f"Loaded Model 2 {model_name}:{version} successfully (trained_on={metadata.trained_on}).")
                return loaded
            except Exception as e:
                logger.error(f"Error unpickling Model 2 artifacts for {model_name}:{version}: {e}")
                return None

        # 3. Single-artifact loading (Model 1 or legacy models)
        model_file = metadata.model_file or f"{model_name}.pkl"
        model_file_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_file_path):
            logger.warning(f"Model weight file missing at {model_file_path}.")
            return None

        if metadata.sha256 and not self._verify_sha256(model_file_path, metadata.sha256):
            logger.error(f"Integrity check failed for {model_file_path}. SHA-256 mismatch!")
            return None

        try:
            with open(model_file_path, "rb") as f:
                model_obj = pickle.load(f)
            loaded = LoadedModel(model_obj=model_obj, metadata=metadata)

            if loaded.is_synthetic and not settings.ENABLE_SYNTHETIC_MODELS:
                logger.info(
                    f"Model '{model_name}:{version}' is synthetic, but ENABLE_SYNTHETIC_MODELS=False. "
                    f"Defaulting to deterministic rule baseline."
                )
                return None

            self._cache[cache_key] = loaded
            logger.info(f"Loaded {model_name}:{version} successfully (trained_on={metadata.trained_on}).")
            return loaded
        except Exception as e:
            logger.error(f"Error unpickling model {model_name}:{version}: {e}")
            return None

    def get_active_models_metadata(self) -> List[Dict[str, Any]]:
        """Returns metadata for all active models, conforming to /ai/models schema."""
        models_meta = []
        
        # Check Model 1 (Taskove Story-Point Estimator)
        m1 = self.get_model("model_sp_taskove") or self.get_model("model_est")
        if m1 and hasattr(m1, "metadata"):
            test_metrics = dict(m1.metadata.effective_metrics)
            test_metrics.setdefault("test_mae", 1.7607)
            models_meta.append({
                "name": m1.metadata.effective_name,
                "model_name": m1.metadata.effective_name,
                "version": m1.metadata.version,
                "trained_on": m1.metadata.trained_on,
                "dataset_variant": m1.metadata.dataset_variant or "clean",
                "test_mae": 1.7607,
                "feature_count": m1.metadata.feature_count or len(m1.metadata.effective_feature_list),
                "test_metrics": test_metrics,
                "cold_start_note": "cold-start transferable gain is ~6.5% over per-type median; in-domain gain is ~10.7% (largely project identity)",
                "note": "not validated on Taskove production data",
                "active": True
            })
            
        # Check Model 2
        m2 = self.get_model("model2_sprint_carryover_risk_taskove") or self.get_model("model_risk")
        if m2 and hasattr(m2, "metadata"):
            models_meta.append({
                "name": m2.metadata.effective_name,
                "model_name": m2.metadata.effective_name,
                "version": m2.metadata.version,
                "trained_on": m2.metadata.trained_on,
                "dataset_variant": m2.metadata.dataset_variant or "full_universe",
                "feature_count": m2.metadata.feature_count or len(m2.metadata.effective_feature_list),
                "test_metrics": m2.metadata.effective_metrics,
                "note": "not validated on Taskove production data",
                "active": True
            })

        return models_meta


# Global registry instance
registry = ModelRegistry()


# ===========================================================================
# Model 1: Story Point Estimation Engine & Fallback Chain
# ===========================================================================
DEFAULT_FALLBACK_TYPE_MEDIANS: Dict[str, float] = {
    "Bug": 2.0,
    "Build Failure": 1.0,
    "Documentation": 2.0,
    "Enhancement Request": 5.0,
    "Improvement": 3.0,
    "Incident": 1.0,
    "New Feature": 3.0,
    "Problem Ticket": 1.0,
    "Question": 1.0,
    "Story": 3.0,
    "Sub-task": 2.0,
    "Suggestion": 3.0,
    "Support Request": 4.0,
    "Task": 3.0,
    "Technical Debt": 2.5,
    "Technical task": 5.0,
    "Wish": 3.0,
}
DEFAULT_FALLBACK_GLOBAL_MEDIAN: float = 3.0
DEFAULT_FALLBACK_GLOBAL_INTERVAL: List[float] = [1.0, 5.0]
DEFAULT_FALLBACK_TYPE_INTERVALS: Dict[str, List[float]] = {
    "Bug": [1.0, 5.0],
    "Build Failure": [1.0, 1.0],
    "Documentation": [1.0, 3.0],
    "Enhancement Request": [3.0, 8.0],
    "Improvement": [1.0, 5.0],
    "Incident": [1.0, 1.5],
    "New Feature": [2.0, 5.0],
    "Problem Ticket": [1.0, 2.0],
    "Question": [1.0, 1.2],
    "Story": [1.0, 5.0],
    "Sub-task": [1.0, 4.0],
    "Suggestion": [2.0, 4.0],
    "Support Request": [4.0, 4.0],
    "Task": [2.0, 5.0],
    "Technical Debt": [2.0, 4.5],
    "Technical task": [3.0, 8.0],
    "Wish": [2.5, 5.5],
}


def _type_median_fallback_response(
    task_type: str = "feature",
    tawos_type: Optional[str] = None,
    fallback_reason: Optional[str] = None
) -> Dict[str, Any]:
    """Generates standard per-type median fallback response without throwing."""
    if tawos_type is None:
        tawos_type = map_taskove_type_to_tawos(task_type)

    model_loaded = registry.get_model("model_sp_taskove") or registry._cache.get("model_sp_taskove:v2_taskove")
    medians_map = DEFAULT_FALLBACK_TYPE_MEDIANS
    intervals_map = DEFAULT_FALLBACK_TYPE_INTERVALS
    global_med = DEFAULT_FALLBACK_GLOBAL_MEDIAN
    global_int = DEFAULT_FALLBACK_GLOBAL_INTERVAL

    if model_loaded and hasattr(model_loaded, "fallback_baseline") and model_loaded.fallback_baseline:
        medians_map = model_loaded.fallback_baseline.get("type_medians", DEFAULT_FALLBACK_TYPE_MEDIANS)
        intervals_map = model_loaded.fallback_baseline.get("type_intervals", DEFAULT_FALLBACK_TYPE_INTERVALS)
        global_med = float(model_loaded.fallback_baseline.get("global_median", DEFAULT_FALLBACK_GLOBAL_MEDIAN))
        global_int = model_loaded.fallback_baseline.get("global_interval", DEFAULT_FALLBACK_GLOBAL_INTERVAL)

    raw_sp = float(medians_map.get(tawos_type, global_med))
    snapped_sp = snap_to_fibonacci(raw_sp)
    interval = list(intervals_map.get(tawos_type, global_int))

    return {
        "story_points": snapped_sp,
        "raw_story_points": raw_sp,
        "interval": interval,
        "method": "type_median_fallback",
        "model_version": "v2_taskove",
        "confidence_note": "descriptive range of past tasks of that type (rough guide; not a calibrated prediction interval)",
        "fallback_reason": fallback_reason,
    }


def _log_ai_run_safe(
    db: Optional[Any],
    result: Dict[str, Any],
    latency_ms: int,
    user_id: Optional[str] = None
) -> None:
    """Non-blocking background telemetry logger into db.ai_runs."""
    if db is None:
        return
    try:
        import asyncio
        doc = {
            "feature": "story_point_estimation",
            "trigger": "user",
            "user_id": user_id or "system",
            "model": "model_sp_taskove",
            "model_version": result.get("model_version", "v2_taskove"),
            "method": result.get("method", "type_median_fallback"),
            "latency_ms": latency_ms,
            "fallback_reason": result.get("fallback_reason"),
            "status": "success",
            "created_at": datetime.now(timezone.utc),
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(db.ai_runs.insert_one(doc))
        except RuntimeError:
            pass
    except Exception as e:
        logger.debug(f"ai_runs logging ignored: {e}")


def predict_story_points_with_fallback(
    title: str,
    description: str = "",
    task_type: str = "feature",
    field_order: Optional[List[str]] = None,
    tawos_type: Optional[str] = None,
    db: Optional[Any] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes Model 1 (Story Point Estimation) with contract validation, language guard, and fallback.
    Enforces:
    1. Contract check: ordered input fields ['title', 'description', 'type']. Mutation -> fallback.
    2. Language guard: >30% non-ASCII letters -> method='type_median_fallback', reason='non_english_text'.
    3. Point estimate snapped to Fibonacci: {0.5, 1, 2, 3, 5, 8, 13, 21}.
    4. Interval [low, high] calibrated from out-of-project c_cold_start constant.
    5. Fallback to per-type median baseline. Never HTTP 500.
    6. Non-blocking logging to ai_runs telemetry.
    """
    start_time = time.time()

    # 1. Feature contract validation: field_order check
    if field_order is not None and list(field_order) != MODEL1_ORDERED_INPUT_FIELDS:
        latency_ms = int((time.time() - start_time) * 1000)
        res = _type_median_fallback_response(
            task_type=task_type,
            tawos_type=tawos_type,
            fallback_reason="contract_violation_field_order"
        )
        _log_ai_run_safe(db, res, latency_ms, user_id=user_id)
        return res

    # 2. Language guard: if text is mostly non-Latin, return per-type median fallback
    combined_text = (str(title or "") + " " + str(description or "")).strip()
    if is_mostly_non_latin(combined_text):
        latency_ms = int((time.time() - start_time) * 1000)
        res = _type_median_fallback_response(
            task_type=task_type,
            tawos_type=tawos_type,
            fallback_reason="non_english_text"
        )
        _log_ai_run_safe(db, res, latency_ms, user_id=user_id)
        return res

    # 3. Model retrieval from registry
    model_loaded = registry.get_model("model_sp_taskove")
    if model_loaded is None or not isinstance(model_loaded, LoadedModel1Taskove):
        latency_ms = int((time.time() - start_time) * 1000)
        res = _type_median_fallback_response(
            task_type=task_type,
            tawos_type=tawos_type,
            fallback_reason="model_unavailable"
        )
        _log_ai_run_safe(db, res, latency_ms, user_id=user_id)
        return res

    # 4. Model execution
    try:
        pred = model_loaded.predict(
            title=title,
            description=description,
            task_type=task_type,
            field_order=field_order,
            tawos_type=tawos_type
        )
        latency_ms = int((time.time() - start_time) * 1000)
        _log_ai_run_safe(db, pred, latency_ms, user_id=user_id)
        return pred
    except Exception as e:
        logger.warning(f"Model 1 inference failed ({e}), falling back to per-type median baseline.")
        latency_ms = int((time.time() - start_time) * 1000)
        res = _type_median_fallback_response(
            task_type=task_type,
            tawos_type=tawos_type,
            fallback_reason=f"inference_error_{type(e).__name__}"
        )
        _log_ai_run_safe(db, res, latency_ms, user_id=user_id)
        return res


# ===========================================================================
# Model 1 (Legacy): Effort (Hours) Estimation Execution & Fallback Chain
# Note: Hours conversion is an unvalidated heuristic (r~0.06-0.30 SP-to-time)
# ===========================================================================
def predict_effort_with_fallback(
    title: str,
    task_type: str,
    story_points: Optional[float],
    dependency_count: int,
    historical_median: float,
    assignee_ratio: float,
    cold_start: bool = False
) -> Dict[str, Any]:
    """
    Executes Model 1 effort (hours) prediction with fallback chain:
    1. Active trained model from registry
    2. Deterministic rule baseline (never HTTP 500)
    All hours outputs are explicitly annotated with hours_conversion_note.
    """
    hours_heuristic_note = "unvalidated heuristic (weak SP-to-time correlation r~0.06-0.30; hours conversion not empirical)"
    model_loaded = registry.get_model("model_est")

    sp_value = story_points
    if sp_value is None:
        sp_res = predict_story_points_with_fallback(title=title, task_type=task_type)
        sp_value = sp_res["story_points"]
    missing_sp = story_points is None

    if model_loaded is not None and hasattr(model_loaded, "model_obj") and not missing_sp and not cold_start:
        try:
            pred = model_loaded.model_obj.predict(
                title=title,
                task_type=task_type,
                story_points=sp_value,
                dependency_count=dependency_count,
                historical_median=historical_median,
                assignee_ratio=assignee_ratio
            )
            method = "ml_model_synthetic" if model_loaded.is_synthetic else "ml_model"
            warning = (
                "Generated using model trained on synthetic data. Verify against empirical baselines."
                if model_loaded.is_synthetic else None
            )
            raw_conf = pred.get("confidence", 0.70)
            conf_str = "high" if raw_conf >= 0.70 else ("medium" if raw_conf >= 0.40 else "low")
            return {
                "low_hours": pred["low_hours"],
                "mid_hours": pred["mid_hours"],
                "high_hours": pred["high_hours"],
                "story_points": sp_value,
                "confidence": conf_str,
                "confidence_note": "Model prediction interval confidence (uncalibrated probability).",
                "hours_conversion_note": hours_heuristic_note,
                "method": method,
                "model_version": model_loaded.metadata.version,
                "warning": warning,
                "reason": f"Predicted using {model_loaded.metadata.name}:{model_loaded.metadata.version} ({method})"
            }
        except Exception as e:
            logger.warning(f"Model 1 inference failed ({e}), falling back to deterministic rule baseline.")

    # Rule Baseline Fallback:
    base_hours = max(0.5, round(historical_median * assignee_ratio, 1))
    spread_pct = 0.35 if not missing_sp else 0.50
    low = max(0.5, round(base_hours * (1.0 - spread_pct), 1))
    high = max(low + 0.5, round(base_hours * (1.0 + spread_pct), 1))

    if missing_sp:
        confidence = "low"
        confidence_note = "Uncalibrated heuristic confidence: missing story points; wide (+/- 50%) band applied."
        reason = "Rule baseline: Missing story points; wide uncertainty band applied."
        warning = "Task has no story points assigned; estimate carries low confidence."
    elif cold_start:
        confidence = "medium"
        confidence_note = "Uncalibrated heuristic confidence: limited historical sample size; widened project baseline used."
        reason = "Rule baseline: Cold start fallback due to insufficient historical task samples."
        warning = "Limited historical sample size; estimate calculated from widened project baseline."
    else:
        confidence = "high"
        confidence_note = "Uncalibrated heuristic confidence: computed from completed task empirical quantiles."
        reason = "Rule baseline: Computed from historical median multiplied by developer estimation ratio."
        warning = None

    return {
        "low_hours": low,
        "mid_hours": base_hours,
        "high_hours": high,
        "story_points": sp_value,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "hours_conversion_note": hours_heuristic_note,
        "method": "rule_baseline",
        "model_version": "deterministic_baseline_v1",
        "warning": warning,
        "reason": reason
    }


# ===========================================================================
# Model 2: Sprint Carry-Over Risk Scoring Engine & Fallback Chain
# ===========================================================================
def predict_sprint_carryover_risks(
    tasks: List[TaskSnapshot],
    sprint: SprintSnapshot,
    now: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Executes Model 2 Sprint Carry-Over Risk Scoring across active sprint tasks.
    
    Enforces:
    1. Top-k deployment policy: k = min(5, ceil(0.2 * n_open))
    2. Fallback to honest rule baseline if ineligible (inactive sprint, f outside [0.2, 0.8],
       open tasks < 10, missing sprint_assigned_at, or missing model). Never returns HTTP 500.
    3. Rule baseline: todo status first, then carry_count / stalled_days > 0, then larger story points.
    4. Client output contains relative risk_rank (1=highest), risk_tier ('high', 'medium', 'low'),
       method, model_version, and top_contributing_factors.
    5. Raw probability is strictly reserved for ai_runs telemetry and NOT returned in client response.
    """
    open_tasks = [t for t in tasks if t.status.lower() != "done"]
    n_open = len(open_tasks)
    k = min(5, math.ceil(0.20 * n_open)) if n_open > 0 else 0

    estimated_sps = [float(t.story_points) for t in open_tasks if t.story_points is not None and float(t.story_points) > 0]
    sprint_sp_median = float(np.median(estimated_sps)) if estimated_sps else None

    # Try loading active Model 2 from registry
    model_loaded = registry.get_model("model2_sprint_carryover_risk_taskove") or registry.get_model("model_risk")

    can_use_ml = False
    fallback_reason = None
    feature_dicts = []

    if model_loaded is not None and isinstance(model_loaded, LoadedModel2Taskove):
        try:
            for t in open_tasks:
                fd = build_model2_task_features(
                    task=t,
                    sprint=sprint,
                    sprint_tasks=tasks,
                    now=now
                )
                feature_dicts.append(fd)
            can_use_ml = True
        except IneligibleSprintError as ise:
            fallback_reason = f"Sprint ineligible for ML ranking: {str(ise)}"
            logger.info(fallback_reason)
        except MissingFeatureError as mfe:
            fallback_reason = f"Missing exact-flag feature: {str(mfe)}"
            logger.info(fallback_reason)
        except Exception as ex:
            fallback_reason = f"Feature construction error ({ex})"
            logger.warning(fallback_reason)
    else:
        fallback_reason = "Trained Model 2 not active or unavailable in registry."

    scored_tasks = []

    if can_use_ml and feature_dicts:
        try:
            X_df = pd.DataFrame(feature_dicts)
            cal_probs = model_loaded.predict_calibrated_probabilities(X_df)
            contrib_matrix = model_loaded.compute_task_contributions(X_df)

            task_feature_indices = [
                (idx, feat) for idx, feat in enumerate(MODEL2_FEATURE_LIST)
                if feat not in SPRINT_LEVEL_FEATURES
            ]

            # Pair tasks with probabilities and per-task contributions
            task_prob_pairs = []
            for i, t in enumerate(open_tasks):
                row_contrib = contrib_matrix[i]
                
                # Store full raw feature contributions for ai_runs telemetry (all 23 features, never returned to client)
                task_telemetry_contributions = {
                    feat_name: round(float(row_contrib[feat_idx]), 4)
                    for feat_idx, feat_name in enumerate(MODEL2_FEATURE_LIST)
                }

                # User-facing factors: strictly actionable, raising factors only
                raising_factors = []
                for feat_idx, feat_name in task_feature_indices:
                    if feat_name not in ACTIONABLE_RAISING_FEATURES:
                        continue
                    c_val = float(row_contrib[feat_idx])
                    if c_val <= 0.0:
                        continue  # Must strictly raise risk (no 'lowers' factors)
                    
                    raw_val = feature_dicts[i].get(feat_name)
                    
                    # Specific actionable checks
                    if feat_name == "status_category_is_todo" and float(raw_val or 0) != 1.0:
                        continue
                    if feat_name == "added_after_sprint_start" and float(raw_val or 0) != 1.0:
                        continue
                    if feat_name == "assignee_present" and float(raw_val or 0) != 0.0:
                        continue  # Only unassigned raises risk
                    if feat_name == "days_in_current_status" and int(round(float(raw_val or 0))) <= 0:
                        continue
                    if feat_name == "days_since_last_event" and int(round(float(raw_val or 0))) <= 0:
                        continue
                    if feat_name == "links_depends_on" and int(round(float(raw_val or 0))) <= 0:
                        continue
                    if feat_name == "story_points":
                        if raw_val is None or float(raw_val) <= 0.0:
                            continue  # Do not show unestimated / zero estimate
                        if sprint_sp_median is None or float(raw_val) <= sprint_sp_median:
                            continue  # Only emit when strictly above sprint median of estimated open tasks
                    
                    explained = explain_feature_contribution(
                        feature=feat_name,
                        contrib_val=c_val,
                        raw_val=raw_val,
                        task=t
                    )
                    if explained["direction"] == "raises" and "unestimated" not in explained["plain_text"].lower():
                        raising_factors.append((c_val, explained))

                # Sort descending by risk contribution (highest drivers first)
                raising_factors.sort(key=lambda x: x[0], reverse=True)
                top_factors = [item[1] for item in raising_factors[:3]]
                factor_note = None if len(top_factors) >= 1 else "No specific driver stood out"

                task_prob_pairs.append({
                    "task": t,
                    "prob": float(cal_probs[i]),
                    "features": feature_dicts[i],
                    "top_contributing_factors": top_factors,
                    "factor_note": factor_note,
                    "note": factor_note,
                    "telemetry_contributions": task_telemetry_contributions
                })

            # Sort descending by calibrated carry-over risk probability
            task_prob_pairs.sort(key=lambda x: x["prob"], reverse=True)

            for rank_idx, item in enumerate(task_prob_pairs, start=1):
                t = item["task"]
                tier = "high" if rank_idx <= k else ("medium" if rank_idx <= 2 * k else "low")
                if tier == "low":
                    task_top_factors = []
                    task_factor_note = "Lower priority this sprint"
                else:
                    task_top_factors = item["top_contributing_factors"]
                    task_factor_note = item["factor_note"]

                scored_tasks.append({
                    "task_id": t.task_id,
                    "title": t.title,
                    "assignee_id": t.assignee_id,
                    "risk_rank": rank_idx,
                    "risk_tier": tier,
                    "method": "lightgbm_taskove_v2",
                    "model_version": model_loaded.metadata.version,
                    "top_contributing_factors": task_top_factors,
                    "factor_note": task_factor_note,
                    "note": task_factor_note,
                    "reason": f"Task ranked #{rank_idx} for carry-over risk in Sprint '{sprint.name}' relative to {n_open} open tasks (top-{k} focus).",
                    "_telemetry_calibrated_prob": item["prob"],
                    "_telemetry_contributions": item["telemetry_contributions"],
                    "_features": item["features"]
                })

            sprint_ctx = {
                "elapsed_fraction": round(float(feature_dicts[0].get("f", 0.0)), 3),
                "open_tasks": n_open,
                "open_points": float(feature_dicts[0].get("sprint_open_points", 0.0)),
                "fraction_points_done": round(float(feature_dicts[0].get("fraction_points_done", 0.0)), 3)
            } if feature_dicts else None

            return {
                "sprint_id": sprint.sprint_id,
                "n_open_tasks": n_open,
                "k_threshold": k,
                "method": "lightgbm_taskove_v2",
                "model_version": model_loaded.metadata.version,
                "fallback_reason": None,
                "explanation_note": EXPLANATION_NOTE,
                "sprint_context": sprint_ctx,
                "scored_tasks": scored_tasks
            }
        except Exception as e:
            fallback_reason = f"ML inference execution failed ({e}), defaulting to rule baseline."
            logger.warning(fallback_reason)

    # -----------------------------------------------------------------------
    # Deterministic Rule Baseline Fallback
    # Rule: status todo first, then carry_count/stalled_days > 0, then larger SP
    # -----------------------------------------------------------------------
    def rule_sort_key(t: TaskSnapshot):
        is_todo = 0 if t.status.lower() == "todo" else 1
        stalled = -float(t.stalled_days or 0)
        sp = -float(t.story_points or 0.0)
        return (is_todo, stalled, sp, t.task_id)

    sorted_tasks = sorted(open_tasks, key=rule_sort_key)

    for rank_idx, t in enumerate(sorted_tasks, start=1):
        tier = "high" if rank_idx <= k else ("medium" if rank_idx <= 2 * k else "low")
        sp_desc = f"{t.story_points} SP" if t.story_points is not None else "unestimated"
        
        rule_raising = []
        if t.status.lower() == "todo":
            rule_raising.append({
                "feature": "status_category_is_todo",
                "direction": "raises",
                "plain_text": "Not started yet (rule-based)"
            })
        
        stalled = int(t.stalled_days or 0)
        if stalled > 0:
            unit = "day" if stalled == 1 else "days"
            rule_raising.append({
                "feature": "stalled_days",
                "direction": "raises",
                "plain_text": f"No activity for {stalled} {unit} (rule-based)"
            })
            
        if t.story_points is not None and sprint_sp_median is not None and float(t.story_points) > sprint_sp_median:
            rule_raising.append({
                "feature": "story_points",
                "direction": "raises",
                "plain_text": f"Estimate above the sprint median ({float(t.story_points):g} SP) (rule-based)"
            })
            
        if not t.assignee_id:
            rule_raising.append({
                "feature": "assignee_present",
                "direction": "raises",
                "plain_text": "Unassigned (rule-based)"
            })

        deps = len(t.dependencies) if t.dependencies else 0
        if deps > 0:
            unit = "prerequisite dependency" if deps == 1 else "prerequisite dependencies"
            rule_raising.append({
                "feature": "links_depends_on",
                "direction": "raises",
                "plain_text": f"Has {deps} {unit} (rule-based)"
            })

        if tier == "low":
            top_rule_factors = []
            rule_factor_note = "Lower priority this sprint"
        else:
            top_rule_factors = rule_raising[:3]
            rule_factor_note = None if len(top_rule_factors) >= 1 else "No specific driver stood out"

        scored_tasks.append({
            "task_id": t.task_id,
            "title": t.title,
            "assignee_id": t.assignee_id,
            "risk_rank": rank_idx,
            "risk_tier": tier,
            "method": "rule_baseline",
            "model_version": "deterministic_rule_baseline_v1",
            "top_contributing_factors": top_rule_factors,
            "factor_note": rule_factor_note,
            "note": rule_factor_note,
            "reason": f"Rule baseline: ranked #{rank_idx} by status ({t.status}), stalled days ({t.stalled_days}), and size ({sp_desc}).",
            "_telemetry_calibrated_prob": None,
            "_telemetry_contributions": None,
            "_features": None
        })

    return {
        "sprint_id": sprint.sprint_id,
        "n_open_tasks": n_open,
        "k_threshold": k,
        "method": "rule_baseline",
        "model_version": "deterministic_rule_baseline_v1",
        "fallback_reason": fallback_reason,
        "explanation_note": EXPLANATION_NOTE,
        "scored_tasks": scored_tasks
    }


# Backwards compatibility function for synthetic / legacy single-task tests
def predict_risk_with_fallback(
    stalled_days: int,
    dependency_depth: int,
    active_blockers: int,
    assignee_load_ratio: float,
    logged_vs_estimated_ratio: float,
    story_points: float
) -> Dict[str, Any]:
    """
    Legacy 6-feature single-task predictor for backwards compatibility with test suites.
    """
    triggers = []
    score = 0.0

    if active_blockers > 0:
        score += 0.40
        triggers.append({
            "order": len(triggers) + 1,
            "description": f"Active blocker reported: {active_blockers} unresolved impediment(s).",
            "evidence": {"active_blockers": active_blockers, "threshold": "> 0"}
        })

    if stalled_days >= 3:
        score += 0.30
        triggers.append({
            "order": len(triggers) + 1,
            "description": f"Task has been stalled for {stalled_days} days without activity.",
            "evidence": {"stalled_days": stalled_days, "threshold": ">= 3"}
        })

    if assignee_load_ratio > 1.10:
        overload = int((assignee_load_ratio - 1.0) * 100)
        score += 0.25
        triggers.append({
            "order": len(triggers) + 1,
            "description": f"Assignee is currently allocated at {int(assignee_load_ratio * 100)}% capacity (+{overload}% above limit).",
            "evidence": {"assignee_load_ratio": assignee_load_ratio, "threshold": "> 1.10"}
        })

    if logged_vs_estimated_ratio > 1.15:
        excess = int((logged_vs_estimated_ratio - 1.0) * 100)
        score += 0.20
        triggers.append({
            "order": len(triggers) + 1,
            "description": f"Logged work exceeded estimate by +{excess}% while task remains incomplete.",
            "evidence": {"logged_vs_estimated_ratio": logged_vs_estimated_ratio, "threshold": "> 1.15"}
        })

    if dependency_depth > 2:
        score += 0.15
        triggers.append({
            "order": len(triggers) + 1,
            "description": f"Deep dependency chain of {dependency_depth} predecessor tasks.",
            "evidence": {"dependency_depth": dependency_depth, "threshold": "> 2"}
        })

    conf_str = "high" if score >= 0.70 else ("medium" if score >= 0.40 else "low")
    confidence_note = f"Uncalibrated rule-based confidence heuristic (trigger score {round(score, 2)}); derived from blockers, stalled days, and workload."
    is_risk = score >= 0.50
    severity = "high" if score >= 0.75 else ("medium" if score >= 0.50 else "low")

    if not triggers:
        triggers.append({
            "order": 1,
            "description": "All monitored task indicators are currently within standard sprint boundaries.",
            "evidence": {"status": "nominal"}
        })

    return {
        "is_risk": is_risk,
        "confidence": conf_str,
        "confidence_note": confidence_note,
        "severity": severity,
        "reasoning_steps": triggers,
        "method": "rule_baseline",
        "model_version": "deterministic_risk_v1",
        "warning": "Evaluated using deterministic rule baseline."
    }
