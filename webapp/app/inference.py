"""Inference layer for the Water Source Classifier.

The app currently supports two backends:

1. dummy
   A deterministic placeholder that returns fake source-type probabilities.
   Use this while the interface is being tested.

2. yolo
   An Ultralytics YOLO classification model that outputs probabilities over
   water-source types. The binary improved/non-improved prediction is derived
   from the top predicted source type using SOURCE_TYPE_TO_BINARY_CLASS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping
import hashlib
import re

import numpy as np
from PIL import Image

from config import (
    MODEL_BACKEND,
    MODEL_PATH,
    MODEL_VERSION,
    SOURCE_TYPE_TO_BINARY_CLASS,
    TOP_K,
)

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # Ultralytics is optional until the real model is added.
    YOLO = None


@dataclass(frozen=True)
class SourceTypePrediction:
    source_type: str
    confidence: float
    binary_class: str


@dataclass(frozen=True)
class PredictionResult:
    binary_prediction: str
    top_source_type: str
    confidence: float
    top_predictions: List[SourceTypePrediction]
    model_backend: str
    model_version: str
    explanation: str


_YOLO_MODEL = None


def normalize_class_name(name: str) -> str:
    """Normalize class names so model labels can match the mapping dictionary."""
    normalized = name.strip().lower()
    normalized = re.sub(r"[\s\-/]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def prettify_class_name(name: str) -> str:
    """Convert source-type labels into readable English for the interface."""
    clean = str(name).replace("_", " ").replace("-", " ").strip()
    return clean[:1].upper() + clean[1:]


def binary_class_from_source_type(source_type: str) -> str:
    normalized = normalize_class_name(source_type)
    return SOURCE_TYPE_TO_BINARY_CLASS.get(normalized, "unknown")


def get_top_k_predictions(
    class_names: Mapping[int, str],
    probabilities: np.ndarray,
    k: int = TOP_K,
) -> List[SourceTypePrediction]:
    """Return the top-k source-type predictions sorted by confidence."""
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        probabilities = probabilities.reshape(-1)

    if probabilities.size == 0:
        raise RuntimeError("The model returned an empty probability vector.")

    k = min(k, probabilities.size)
    top_indices = np.argsort(probabilities)[::-1][:k]

    predictions: List[SourceTypePrediction] = []
    for idx in top_indices:
        idx_int = int(idx)
        raw_name = class_names.get(idx_int, f"class_{idx_int}")
        source_type = str(raw_name)
        confidence = float(probabilities[idx_int])
        binary_class = binary_class_from_source_type(source_type)

        predictions.append(
            SourceTypePrediction(
                source_type=source_type,
                confidence=confidence,
                binary_class=binary_class,
            )
        )

    return predictions


def predict_with_dummy_model(image: Image.Image) -> PredictionResult:
    """Return deterministic fake predictions for UI testing.

    The dummy model uses image bytes only to make repeated uploads of the same
    image return the same fake prediction. These values are not meaningful.
    """
    class_names = {
        0: "tube_well_or_borehole",
        1: "protected_dug_well",
        2: "unprotected_dug_well",
        3: "surface_water",
        4: "rainwater_harvesting",
        5: "sand_or_subsurface_dam",
        6: "protected_spring",
        7: "unprotected_spring",
    }

    thumb = image.convert("RGB").resize((96, 96))
    digest = hashlib.sha256(thumb.tobytes()).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)

    rng = np.random.default_rng(seed)
    raw_scores = rng.random(len(class_names))
    probabilities = raw_scores / raw_scores.sum()

    top_predictions = get_top_k_predictions(class_names, probabilities, TOP_K)
    top_prediction = top_predictions[0]

    return PredictionResult(
        binary_prediction=top_prediction.binary_class,
        top_source_type=top_prediction.source_type,
        confidence=top_prediction.confidence,
        top_predictions=top_predictions,
        model_backend="dummy",
        model_version=MODEL_VERSION,
        explanation=(
            "This is a demo prediction produced by a placeholder model. "
            "Replace it with the trained YOLO classification model before real use."
        ),
    )


def _load_yolo_model():
    global _YOLO_MODEL

    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    if YOLO is None:
        raise RuntimeError(
            "The 'ultralytics' package is not installed. "
            "Install it after adding the real YOLO model."
        )

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"YOLO model file not found at {model_file}. "
            "Place the trained model there or update MODEL_PATH in config.py."
        )

    _YOLO_MODEL = YOLO(str(model_file))
    return _YOLO_MODEL


def predict_with_yolo_model(image: Image.Image) -> PredictionResult:
    """Predict using an Ultralytics YOLO classification model.

    Expected model behavior:
    - The model returns class probabilities over water-source types.
    - The top source-type prediction is mapped to improved/non-improved.
    """
    model = _load_yolo_model()
    results = model.predict(image, verbose=False)

    if not results:
        raise RuntimeError("YOLO returned no results.")

    result = results[0]
    if not hasattr(result, "probs") or result.probs is None:
        raise RuntimeError(
            "The loaded YOLO model did not return class probabilities. "
            "This app expects a YOLO classification model, not a detection model."
        )

    probabilities = result.probs.data.detach().cpu().numpy().astype(float)
    names = getattr(result, "names", None) or getattr(model, "names", {})

    # Ultralytics names are usually a dict such as {0: 'class_a', 1: 'class_b'}.
    class_names = {int(idx): str(name) for idx, name in dict(names).items()}

    top_predictions = get_top_k_predictions(class_names, probabilities, TOP_K)
    top_prediction = top_predictions[0]

    return PredictionResult(
        binary_prediction=top_prediction.binary_class,
        top_source_type=top_prediction.source_type,
        confidence=top_prediction.confidence,
        top_predictions=top_predictions,
        model_backend="yolo",
        model_version=MODEL_VERSION,
        explanation="Prediction produced by the trained YOLO classification model.",
    )


def predict(image: Image.Image) -> PredictionResult:
    backend = MODEL_BACKEND.strip().lower()

    if backend == "dummy":
        return predict_with_dummy_model(image)

    if backend == "yolo":
        return predict_with_yolo_model(image)

    raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")
