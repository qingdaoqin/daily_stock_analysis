# -*- coding: utf-8 -*-
"""Lightweight trainable calibration model for analysis feedback loops."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_MODEL_VERSION = 1
_VARIANCE_FLOOR = 1e-4


@dataclass
class SmallCalibrationPrediction:
    """Prediction payload returned by the small calibration model."""

    predicted_signal: str
    confidence: float
    probabilities: Dict[str, float]
    sample_count: int
    validation_accuracy: Optional[float] = None
    baseline_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_signal": self.predicted_signal,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "sample_count": self.sample_count,
            "validation_accuracy": self.validation_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
        }


class SmallCalibrationModel:
    """Simple Gaussian Naive Bayes model persisted as JSON."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        class_stats: Dict[str, Dict[str, Any]],
        sample_count: int,
        trained_at_ts: Optional[float] = None,
        validation_accuracy: Optional[float] = None,
        validation_count: int = 0,
        baseline_accuracy: Optional[float] = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.class_stats = class_stats
        self.sample_count = int(sample_count)
        self.trained_at_ts = float(trained_at_ts or time.time())
        self.validation_accuracy = validation_accuracy
        self.validation_count = int(validation_count)
        self.baseline_accuracy = baseline_accuracy

    @property
    def labels(self) -> List[str]:
        return list(self.class_stats.keys())

    @classmethod
    def fit(cls, samples: Sequence[Dict[str, Any]]) -> Optional["SmallCalibrationModel"]:
        ordered_samples = [sample for sample in samples if isinstance(sample, dict)]
        if len(ordered_samples) < 6:
            return None

        feature_names = sorted(
            {
                str(name)
                for sample in ordered_samples
                for name in (sample.get("features") or {}).keys()
            }
        )
        if not feature_names:
            return None

        split_index = int(len(ordered_samples) * 0.8) if len(ordered_samples) >= 20 else len(ordered_samples)
        split_index = max(1, min(split_index, len(ordered_samples)))

        train_samples = ordered_samples[:split_index]
        validation_samples = ordered_samples[split_index:]
        class_stats = cls._fit_class_stats(train_samples, feature_names)
        if len(class_stats) < 2:
            return None

        model = cls(
            feature_names=feature_names,
            class_stats=class_stats,
            sample_count=len(ordered_samples),
        )
        if validation_samples:
            validation_accuracy = model._evaluate(validation_samples)
            model.validation_accuracy = validation_accuracy
            model.validation_count = len(validation_samples)

        label_counts: Dict[str, int] = {}
        for sample in train_samples:
            label = str(sample.get("label") or "").strip().lower()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts:
            model.baseline_accuracy = max(label_counts.values()) / max(sum(label_counts.values()), 1)

        return model

    @classmethod
    def _fit_class_stats(
        cls,
        samples: Sequence[Dict[str, Any]],
        feature_names: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        label_groups: Dict[str, List[Dict[str, float]]] = {}
        for sample in samples:
            label = str(sample.get("label") or "").strip().lower()
            if not label:
                continue
            label_groups.setdefault(label, []).append(cls._vectorize(sample.get("features") or {}, feature_names))

        total = sum(len(rows) for rows in label_groups.values())
        if total <= 0:
            return {}

        class_stats: Dict[str, Dict[str, Any]] = {}
        for label, rows in label_groups.items():
            if not rows:
                continue
            means: Dict[str, float] = {}
            variances: Dict[str, float] = {}
            count = len(rows)
            for feature_name in feature_names:
                values = [float(row.get(feature_name, 0.0)) for row in rows]
                mean = sum(values) / count
                variance = sum((value - mean) ** 2 for value in values) / count
                means[feature_name] = mean
                variances[feature_name] = max(variance, _VARIANCE_FLOOR)
            class_stats[label] = {
                "count": count,
                "prior": count / total,
                "means": means,
                "variances": variances,
            }
        return class_stats

    @staticmethod
    def _vectorize(features: Dict[str, Any], feature_names: Sequence[str]) -> Dict[str, float]:
        vector: Dict[str, float] = {}
        for feature_name in feature_names:
            raw_value = features.get(feature_name, 0.0)
            try:
                vector[feature_name] = float(raw_value)
            except (TypeError, ValueError):
                vector[feature_name] = 0.0
        return vector

    def predict(self, features: Dict[str, Any]) -> Optional[SmallCalibrationPrediction]:
        if not self.class_stats:
            return None

        vector = self._vectorize(features, self.feature_names)
        log_scores: Dict[str, float] = {}
        for label, stats in self.class_stats.items():
            prior = max(float(stats.get("prior") or 0.0), 1e-8)
            score = math.log(prior)
            means = stats.get("means") or {}
            variances = stats.get("variances") or {}
            for feature_name in self.feature_names:
                variance = max(float(variances.get(feature_name, _VARIANCE_FLOOR)), _VARIANCE_FLOOR)
                mean = float(means.get(feature_name, 0.0))
                value = float(vector.get(feature_name, 0.0))
                score += -0.5 * math.log(2 * math.pi * variance)
                score += -((value - mean) ** 2) / (2 * variance)
            log_scores[label] = score

        max_log = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probabilities = {label: value / total for label, value in exp_scores.items()}
        predicted_signal = max(probabilities.items(), key=lambda item: item[1])[0]
        confidence = float(probabilities.get(predicted_signal, 0.0))
        return SmallCalibrationPrediction(
            predicted_signal=predicted_signal,
            confidence=confidence,
            probabilities=probabilities,
            sample_count=self.sample_count,
            validation_accuracy=self.validation_accuracy,
            baseline_accuracy=self.baseline_accuracy,
        )

    def _evaluate(self, samples: Sequence[Dict[str, Any]]) -> float:
        total = 0
        correct = 0
        for sample in samples:
            label = str(sample.get("label") or "").strip().lower()
            prediction = self.predict(sample.get("features") or {})
            if not label or prediction is None:
                continue
            total += 1
            if prediction.predicted_signal == label:
                correct += 1
        if total <= 0:
            return 0.0
        return correct / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": _MODEL_VERSION,
            "trained_at_ts": self.trained_at_ts,
            "sample_count": self.sample_count,
            "feature_names": list(self.feature_names),
            "class_stats": self.class_stats,
            "validation_accuracy": self.validation_accuracy,
            "validation_count": self.validation_count,
            "baseline_accuracy": self.baseline_accuracy,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> Optional["SmallCalibrationModel"]:
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version") or 0) != _MODEL_VERSION:
            return None
        feature_names = payload.get("feature_names") or []
        class_stats = payload.get("class_stats") or {}
        if not feature_names or not class_stats:
            return None
        return cls(
            feature_names=feature_names,
            class_stats=class_stats,
            sample_count=int(payload.get("sample_count") or 0),
            trained_at_ts=float(payload.get("trained_at_ts") or time.time()),
            validation_accuracy=payload.get("validation_accuracy"),
            validation_count=int(payload.get("validation_count") or 0),
            baseline_accuracy=payload.get("baseline_accuracy"),
        )

    def save(self, path: str) -> None:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> Optional["SmallCalibrationModel"]:
        model_path = Path(path)
        if not model_path.exists():
            return None
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[LearningLoop] 加载小型校准模型失败: %s", exc)
            return None
        return cls.from_dict(payload)
