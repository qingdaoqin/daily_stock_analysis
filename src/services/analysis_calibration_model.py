# -*- coding: utf-8 -*-
"""Trainable calibration model for analysis feedback loops.

Default strategy:
1. Prefer a tree-based classifier (HistGradientBoostingClassifier) when
   scikit-learn is available.
2. Train a global model plus per-market submodels when enough samples exist.
3. Fall back to the previous Gaussian Naive Bayes implementation only when the
   tree backend is unavailable or the scoped sample set is too small.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_MODEL_VERSION = 2
_VARIANCE_FLOOR = 1e-4
_TREE_BACKEND = "hist_gradient_boosting"
_NB_BACKEND = "naive_bayes"
_DEFAULT_SCOPE = "global"

try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when sklearn missing
    HistGradientBoostingClassifier = None
    _SKLEARN_AVAILABLE = False


@dataclass
class SmallCalibrationPrediction:
    """Prediction payload returned by the calibration model."""

    predicted_signal: str
    confidence: float
    probabilities: Dict[str, float]
    sample_count: int
    validation_accuracy: Optional[float] = None
    validation_count: int = 0
    baseline_accuracy: Optional[float] = None
    engine: str = _NB_BACKEND
    scope: str = _DEFAULT_SCOPE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_signal": self.predicted_signal,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "sample_count": self.sample_count,
            "validation_accuracy": self.validation_accuracy,
            "validation_count": self.validation_count,
            "baseline_accuracy": self.baseline_accuracy,
            "engine": self.engine,
            "scope": self.scope,
        }


class SmallCalibrationModel:
    """Tree-first trainable calibration model with JSON persistence."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        submodels: Dict[str, Dict[str, Any]],
        sample_count: int,
        trained_at_ts: Optional[float] = None,
        preferred_backend: str = _TREE_BACKEND,
    ) -> None:
        self.feature_names = list(feature_names)
        self.submodels = submodels
        self.sample_count = int(sample_count)
        self.trained_at_ts = float(trained_at_ts or time.time())
        self.preferred_backend = preferred_backend
        self._runtime_cache: Dict[str, Any] = {}

    @property
    def labels(self) -> List[str]:
        global_model = self.submodels.get(_DEFAULT_SCOPE) or next(iter(self.submodels.values()), {})
        return list(global_model.get("label_names") or [])

    @property
    def validation_accuracy(self) -> Optional[float]:
        global_model = self.submodels.get(_DEFAULT_SCOPE) or {}
        return global_model.get("validation_accuracy")

    @property
    def validation_count(self) -> int:
        global_model = self.submodels.get(_DEFAULT_SCOPE) or {}
        return int(global_model.get("validation_count") or 0)

    @property
    def baseline_accuracy(self) -> Optional[float]:
        global_model = self.submodels.get(_DEFAULT_SCOPE) or {}
        return global_model.get("baseline_accuracy")

    @classmethod
    def fit(
        cls,
        samples: Sequence[Dict[str, Any]],
        *,
        preferred_backend: str = "tree",
        market_split: bool = True,
        min_scope_samples: int = 18,
    ) -> Optional["SmallCalibrationModel"]:
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

        scope_samples: Dict[str, List[Dict[str, Any]]] = {_DEFAULT_SCOPE: list(ordered_samples)}
        if market_split:
            for market in ("cn", "hk", "us"):
                scoped_rows = [
                    sample
                    for sample in ordered_samples
                    if str(sample.get("market") or "").strip().lower() == market
                ]
                if len(scoped_rows) >= min_scope_samples:
                    scope_samples[market] = scoped_rows

        submodels: Dict[str, Dict[str, Any]] = {}
        for scope, scoped_rows in scope_samples.items():
            trained = cls._fit_scope_model(
                scoped_rows,
                feature_names,
                preferred_backend=preferred_backend,
            )
            if trained is not None:
                trained["scope"] = scope
                submodels[scope] = trained

        if _DEFAULT_SCOPE not in submodels:
            return None

        actual_backend = str(
            (submodels.get(_DEFAULT_SCOPE) or {}).get("engine")
            or preferred_backend
        )
        return cls(
            feature_names=feature_names,
            submodels=submodels,
            sample_count=len(ordered_samples),
            preferred_backend=actual_backend,
        )

    @classmethod
    def _fit_scope_model(
        cls,
        samples: Sequence[Dict[str, Any]],
        feature_names: Sequence[str],
        *,
        preferred_backend: str,
    ) -> Optional[Dict[str, Any]]:
        if len(samples) < 6:
            return None

        ordered = list(samples)
        split_index = int(len(ordered) * 0.8) if len(ordered) >= 20 else len(ordered)
        split_index = max(1, min(split_index, len(ordered)))
        train_samples = ordered[:split_index]
        validation_samples = ordered[split_index:]

        labels = sorted(
            {
                str(sample.get("label") or "").strip().lower()
                for sample in train_samples
                if str(sample.get("label") or "").strip()
            }
        )
        if len(labels) < 2:
            labels = sorted(
                {
                    str(sample.get("label") or "").strip().lower()
                    for sample in ordered
                    if str(sample.get("label") or "").strip()
                }
            )
            if len(labels) < 2:
                return None
            train_samples = ordered
            validation_samples = []

        backend = cls._resolve_backend(
            preferred_backend=preferred_backend,
            sample_count=len(samples),
        )

        if backend == _TREE_BACKEND:
            trained = cls._fit_tree_backend(train_samples, validation_samples, feature_names, labels)
            if trained is not None:
                return trained

        return cls._fit_nb_backend(train_samples, validation_samples, feature_names, labels)

    @staticmethod
    def _resolve_backend(*, preferred_backend: str, sample_count: int) -> str:
        normalized = str(preferred_backend or "tree").strip().lower()
        if normalized in {"tree", "auto", _TREE_BACKEND} and _SKLEARN_AVAILABLE and sample_count >= 12:
            return _TREE_BACKEND
        return _NB_BACKEND

    @staticmethod
    def _vectorize_dict(features: Dict[str, Any], feature_names: Sequence[str]) -> Dict[str, float]:
        vector: Dict[str, float] = {}
        for feature_name in feature_names:
            raw_value = features.get(feature_name, 0.0)
            try:
                vector[feature_name] = float(raw_value)
            except (TypeError, ValueError):
                vector[feature_name] = 0.0
        return vector

    @classmethod
    def _vectorize_list(cls, features: Dict[str, Any], feature_names: Sequence[str]) -> List[float]:
        vector = cls._vectorize_dict(features, feature_names)
        return [vector[feature_name] for feature_name in feature_names]

    @classmethod
    def _fit_tree_backend(
        cls,
        train_samples: Sequence[Dict[str, Any]],
        validation_samples: Sequence[Dict[str, Any]],
        feature_names: Sequence[str],
        label_names: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if not _SKLEARN_AVAILABLE or HistGradientBoostingClassifier is None:
            return None

        X_train = [cls._vectorize_list(sample.get("features") or {}, feature_names) for sample in train_samples]
        y_train = [str(sample.get("label") or "").strip().lower() for sample in train_samples]
        if len({label for label in y_train if label}) < 2:
            return None

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=240,
            max_depth=6,
            min_samples_leaf=max(4, len(train_samples) // 18),
            l2_regularization=0.08,
            random_state=42,
        )
        model.fit(X_train, y_train)

        validation_accuracy = cls._evaluate_tree_model(
            model,
            validation_samples,
            feature_names,
        )
        baseline_accuracy = cls._compute_baseline_accuracy(train_samples)

        payload = base64.b64encode(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
        return {
            "engine": _TREE_BACKEND,
            "sample_count": len(train_samples) + len(validation_samples),
            "validation_accuracy": validation_accuracy,
            "validation_count": len(validation_samples),
            "baseline_accuracy": baseline_accuracy,
            "label_names": list(getattr(model, "classes_", label_names)),
            "artifact_b64": payload,
        }

    @classmethod
    def _fit_nb_backend(
        cls,
        train_samples: Sequence[Dict[str, Any]],
        validation_samples: Sequence[Dict[str, Any]],
        feature_names: Sequence[str],
        label_names: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        class_stats = cls._fit_nb_class_stats(train_samples, feature_names)
        if len(class_stats) < 2:
            return None

        temp_model = cls(
            feature_names=feature_names,
            submodels={
                _DEFAULT_SCOPE: {
                    "engine": _NB_BACKEND,
                    "sample_count": len(train_samples) + len(validation_samples),
                    "validation_accuracy": None,
                    "validation_count": len(validation_samples),
                    "baseline_accuracy": cls._compute_baseline_accuracy(train_samples),
                    "label_names": list(label_names),
                    "class_stats": class_stats,
                }
            },
            sample_count=len(train_samples) + len(validation_samples),
            preferred_backend=_NB_BACKEND,
        )
        validation_accuracy = temp_model._evaluate(validation_samples, scope=_DEFAULT_SCOPE)
        return {
            "engine": _NB_BACKEND,
            "sample_count": len(train_samples) + len(validation_samples),
            "validation_accuracy": validation_accuracy,
            "validation_count": len(validation_samples),
            "baseline_accuracy": cls._compute_baseline_accuracy(train_samples),
            "label_names": list(label_names),
            "class_stats": class_stats,
        }

    @classmethod
    def _fit_nb_class_stats(
        cls,
        samples: Sequence[Dict[str, Any]],
        feature_names: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        label_groups: Dict[str, List[Dict[str, float]]] = {}
        for sample in samples:
            label = str(sample.get("label") or "").strip().lower()
            if not label:
                continue
            label_groups.setdefault(label, []).append(
                cls._vectorize_dict(sample.get("features") or {}, feature_names)
            )

        total = sum(len(rows) for rows in label_groups.values())
        if total <= 0:
            return {}

        class_stats: Dict[str, Dict[str, Any]] = {}
        for label, rows in label_groups.items():
            count = len(rows)
            means: Dict[str, float] = {}
            variances: Dict[str, float] = {}
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
    def _compute_baseline_accuracy(samples: Sequence[Dict[str, Any]]) -> Optional[float]:
        label_counts: Dict[str, int] = {}
        for sample in samples:
            label = str(sample.get("label") or "").strip().lower()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
        if not label_counts:
            return None
        return max(label_counts.values()) / max(sum(label_counts.values()), 1)

    @classmethod
    def _evaluate_tree_model(
        cls,
        model: Any,
        samples: Sequence[Dict[str, Any]],
        feature_names: Sequence[str],
    ) -> Optional[float]:
        if not samples:
            return None
        total = 0
        correct = 0
        for sample in samples:
            label = str(sample.get("label") or "").strip().lower()
            if not label:
                continue
            vector = [cls._vectorize_list(sample.get("features") or {}, feature_names)]
            prediction = model.predict(vector)[0]
            total += 1
            if prediction == label:
                correct += 1
        if total <= 0:
            return None
        return correct / total

    def _evaluate(self, samples: Sequence[Dict[str, Any]], *, scope: str = _DEFAULT_SCOPE) -> Optional[float]:
        if not samples:
            return None
        total = 0
        correct = 0
        for sample in samples:
            label = str(sample.get("label") or "").strip().lower()
            prediction = self.predict(sample.get("features") or {}, scope_hint=scope)
            if not label or prediction is None:
                continue
            total += 1
            if prediction.predicted_signal == label:
                correct += 1
        if total <= 0:
            return None
        return correct / total

    def _resolve_scopes(self, scope_hint: Optional[str]) -> List[str]:
        scopes: List[str] = []
        normalized_scope = str(scope_hint or "").strip().lower()
        if normalized_scope and normalized_scope in self.submodels:
            scopes.append(normalized_scope)
        if _DEFAULT_SCOPE not in scopes and _DEFAULT_SCOPE in self.submodels:
            scopes.append(_DEFAULT_SCOPE)
        for scope in self.submodels:
            if scope not in scopes:
                scopes.append(scope)
        return scopes

    def predict(
        self,
        features: Dict[str, Any],
        *,
        scope_hint: Optional[str] = None,
    ) -> Optional[SmallCalibrationPrediction]:
        for scope in self._resolve_scopes(scope_hint):
            submodel = self.submodels.get(scope)
            if not submodel:
                continue
            prediction = self._predict_with_submodel(features, scope=scope, submodel=submodel)
            if prediction is not None:
                return prediction
        return None

    def _predict_with_submodel(
        self,
        features: Dict[str, Any],
        *,
        scope: str,
        submodel: Dict[str, Any],
    ) -> Optional[SmallCalibrationPrediction]:
        engine = str(submodel.get("engine") or _NB_BACKEND)
        label_names = list(submodel.get("label_names") or [])
        if not label_names:
            return None

        if engine == _TREE_BACKEND:
            model = self._load_runtime_tree_model(scope, submodel)
            if model is None:
                return None
            vector = [self._vectorize_list(features, self.feature_names)]
            probabilities_list = model.predict_proba(vector)
            if not probabilities_list.size:  # pragma: no cover - defensive
                return None
            probabilities = {
                str(label): float(probabilities_list[0][idx])
                for idx, label in enumerate(getattr(model, "classes_", label_names))
            }
        else:
            class_stats = submodel.get("class_stats") or {}
            probabilities = self._predict_nb_probabilities(features, class_stats)
            if not probabilities:
                return None

        predicted_signal = max(probabilities.items(), key=lambda item: item[1])[0]
        confidence = float(probabilities.get(predicted_signal, 0.0))
        return SmallCalibrationPrediction(
            predicted_signal=predicted_signal,
            confidence=confidence,
            probabilities=probabilities,
            sample_count=int(submodel.get("sample_count") or self.sample_count),
            validation_accuracy=submodel.get("validation_accuracy"),
            validation_count=int(submodel.get("validation_count") or 0),
            baseline_accuracy=submodel.get("baseline_accuracy"),
            engine=engine,
            scope=scope,
        )

    def _load_runtime_tree_model(self, scope: str, submodel: Dict[str, Any]) -> Optional[Any]:
        if scope in self._runtime_cache:
            return self._runtime_cache[scope]

        artifact_b64 = str(submodel.get("artifact_b64") or "").strip()
        if not artifact_b64:
            return None
        try:
            model = pickle.loads(base64.b64decode(artifact_b64.encode("ascii")))
        except Exception as exc:  # pragma: no cover - corrupt artifact
            logger.warning("[LearningLoop] 加载树模型 artifact 失败(scope=%s): %s", scope, exc)
            return None
        self._runtime_cache[scope] = model
        return model

    def _predict_nb_probabilities(
        self,
        features: Dict[str, Any],
        class_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        if not class_stats:
            return {}

        vector = self._vectorize_dict(features, self.feature_names)
        log_scores: Dict[str, float] = {}
        for label, stats in class_stats.items():
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
        return {label: value / total for label, value in exp_scores.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": _MODEL_VERSION,
            "trained_at_ts": self.trained_at_ts,
            "sample_count": self.sample_count,
            "feature_names": list(self.feature_names),
            "preferred_backend": self.preferred_backend,
            "submodels": self.submodels,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> Optional["SmallCalibrationModel"]:
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version") or 0) != _MODEL_VERSION:
            return None

        feature_names = payload.get("feature_names") or []
        submodels = payload.get("submodels") or {}
        if not feature_names or not submodels:
            return None
        return cls(
            feature_names=feature_names,
            submodels=submodels,
            sample_count=int(payload.get("sample_count") or 0),
            trained_at_ts=float(payload.get("trained_at_ts") or time.time()),
            preferred_backend=str(payload.get("preferred_backend") or _TREE_BACKEND),
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
        except Exception as exc:  # pragma: no cover - corrupt json
            logger.warning("[LearningLoop] 加载校准模型失败: %s", exc)
            return None
        return cls.from_dict(payload)
