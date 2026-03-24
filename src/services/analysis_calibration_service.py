# -*- coding: utf-8 -*-
"""Backtest-driven calibration for the standard analysis pipeline.

The project already contains backtesting and agent memory primitives, but the
classic single-shot analysis path does not consume that feedback.  This
service closes the smallest useful loop:

1. Periodically run incremental backtests for older analysis records.
2. Aggregate historical signal accuracy by stock / market / model.
3. Use those metrics to dampen, downgrade, or occasionally reverse weak
   low-quality signals before the report is persisted and notified.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from sqlalchemy import desc, select

from src.core.trading_calendar import get_market_for_stock
from src.services.analysis_calibration_model import SmallCalibrationModel
from src.services.backtest_service import BacktestService
from src.storage import AnalysisHistory, BacktestResult, DatabaseManager

logger = logging.getLogger(__name__)

_DEFAULT_REFRESH_INTERVAL_MINUTES = 60
_DEFAULT_MIN_SAMPLES = 20
_DEFAULT_HISTORY_LIMIT = 800
_DEFAULT_AUTO_BACKTEST_LIMIT = 200


def _safe_ratio(value: Optional[float], default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signal_from_operation_advice(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "hold"

    buy_tokens = ("买入", "加仓", "建仓", "buy", "long", "bull")
    sell_tokens = ("卖出", "减仓", "清仓", "sell", "short", "bear")
    hold_tokens = ("持有", "观望", "等待", "wait", "watch", "hold", "neutral")

    if any(token in text for token in buy_tokens):
        return "buy"
    if any(token in text for token in sell_tokens):
        return "sell"
    if any(token in text for token in hold_tokens):
        return "hold"
    return "hold"


def _pct_to_fraction(value: Any) -> float:
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return 0.0


@dataclass
class SignalStats:
    signal: str
    samples: int = 0
    accuracy: float = 0.5
    avg_return: float = 0.0


@dataclass
class CalibrationProfile:
    enabled: bool = False
    market: str = "cn"
    model_used: Optional[str] = None
    source_scope: str = ""
    samples: int = 0
    threshold: int = _DEFAULT_MIN_SAMPLES
    current_signal: str = "hold"
    current_signal_stats: Optional[SignalStats] = None
    best_alternative_stats: Optional[SignalStats] = None
    suggested_signal: Optional[str] = None
    score_adjustment: int = 0
    applied: bool = False
    reason: str = ""
    model_prediction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.current_signal_stats is not None:
            payload["current_signal_stats"] = asdict(self.current_signal_stats)
        if self.best_alternative_stats is not None:
            payload["best_alternative_stats"] = asdict(self.best_alternative_stats)
        if self.model_prediction is not None:
            payload["model_prediction"] = dict(self.model_prediction)
        return payload


class AnalysisCalibrationService:
    """Use historical backtest outcomes to calibrate current signals."""

    def __init__(
        self,
        *,
        db_manager: Optional[DatabaseManager] = None,
        config: Optional[Any] = None,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager.get_instance()
        self.config = config
        self._time_fn = time_fn or time.time
        self._last_refresh_ts = 0.0
        self._small_model = self._load_small_model()
        self._last_model_train_ts = self._small_model.trained_at_ts if self._small_model else 0.0

    @property
    def enabled(self) -> bool:
        return isinstance(self.db, DatabaseManager) and bool(getattr(self.config, "backtest_enabled", True))

    @property
    def refresh_interval_seconds(self) -> int:
        minutes = int(getattr(self.config, "analysis_learning_refresh_interval_minutes", _DEFAULT_REFRESH_INTERVAL_MINUTES))
        return max(0, minutes) * 60

    @property
    def min_samples(self) -> int:
        return max(3, int(getattr(self.config, "analysis_learning_min_samples", _DEFAULT_MIN_SAMPLES)))

    @property
    def model_enabled(self) -> bool:
        return self.enabled and bool(getattr(self.config, "analysis_learning_model_enabled", True))

    @property
    def model_path(self) -> str:
        return str(
            getattr(
                self.config,
                "analysis_learning_model_path",
                "./data/models/analysis_calibration_model.json",
            )
        )

    @property
    def model_retrain_interval_seconds(self) -> int:
        minutes = int(
            getattr(
                self.config,
                "analysis_learning_model_retrain_interval_minutes",
                180,
            )
        )
        return max(0, minutes) * 60

    @property
    def model_train_min_samples(self) -> int:
        return max(
            12,
            int(getattr(self.config, "analysis_learning_model_train_min_samples", 60)),
        )

    @property
    def model_confidence_threshold(self) -> float:
        return max(
            0.5,
            min(
                0.95,
                float(
                    getattr(
                        self.config,
                        "analysis_learning_model_confidence_threshold",
                        0.62,
                    )
                ),
            ),
        )

    @property
    def learning_label_band_pct(self) -> float:
        return max(
            0.5,
            float(
                getattr(
                    self.config,
                    "analysis_learning_label_band_pct",
                    getattr(self.config, "backtest_neutral_band_pct", 2.0),
                )
            ),
        )

    def _load_small_model(self) -> Optional[SmallCalibrationModel]:
        if not self.model_enabled:
            return None
        return SmallCalibrationModel.load(self.model_path)

    def maybe_refresh_backtests(self) -> Dict[str, Any]:
        """Run incremental backtest refresh if the cooldown window has elapsed."""
        if not self.enabled:
            return {"success": False, "reason": "disabled"}

        now = self._time_fn()
        if self._last_refresh_ts and now - self._last_refresh_ts < self.refresh_interval_seconds:
            return {"success": True, "reason": "cooldown"}

        try:
            stats = BacktestService(self.db).run_backtest(
                code=None,
                force=False,
                limit=int(getattr(self.config, "analysis_learning_auto_backtest_limit", _DEFAULT_AUTO_BACKTEST_LIMIT)),
            )
            self._last_refresh_ts = now
            logger.info(
                "[LearningLoop] 增量回测刷新完成: processed=%s saved=%s completed=%s insufficient=%s errors=%s",
                stats.get("processed", 0),
                stats.get("saved", 0),
                stats.get("completed", 0),
                stats.get("insufficient", 0),
                stats.get("errors", 0),
            )
            return {"success": True, **stats}
        except Exception as exc:
            logger.warning("[LearningLoop] 增量回测刷新失败: %s", exc)
            return {"success": False, "reason": str(exc)}

    def maybe_refresh_model(self) -> Dict[str, Any]:
        """Retrain the lightweight calibration model when enough history exists."""
        if not self.model_enabled:
            return {"success": False, "reason": "disabled"}

        now = self._time_fn()
        if self._last_model_train_ts and now - self._last_model_train_ts < self.model_retrain_interval_seconds:
            return {"success": True, "reason": "cooldown"}

        rows = self._load_learning_rows()
        samples = self._build_training_samples(rows)
        if len(samples) < self.model_train_min_samples:
            return {
                "success": False,
                "reason": f"insufficient_samples:{len(samples)}",
                "sample_count": len(samples),
            }

        model = SmallCalibrationModel.fit(samples)
        if model is None:
            return {
                "success": False,
                "reason": "fit_failed",
                "sample_count": len(samples),
            }

        try:
            model.save(self.model_path)
        except Exception as exc:
            logger.warning("[LearningLoop] 保存小型校准模型失败: %s", exc)
            return {"success": False, "reason": str(exc), "sample_count": len(samples)}

        self._small_model = model
        self._last_model_train_ts = now
        logger.info(
            "[LearningLoop] 小型校准模型训练完成: samples=%s validation_accuracy=%s baseline_accuracy=%s path=%s",
            model.sample_count,
            f"{model.validation_accuracy:.3f}" if model.validation_accuracy is not None else "n/a",
            f"{model.baseline_accuracy:.3f}" if model.baseline_accuracy is not None else "n/a",
            self.model_path,
        )
        return {
            "success": True,
            "sample_count": model.sample_count,
            "validation_accuracy": model.validation_accuracy,
            "baseline_accuracy": model.baseline_accuracy,
        }

    def calibrate_result(self, result: Any, context_snapshot: Optional[Dict[str, Any]] = None) -> Any:
        """Calibrate one AnalysisResult in-place and return it."""
        if not self.enabled or result is None or not getattr(result, "success", True):
            return result

        decision_type = str(getattr(result, "decision_type", "") or "").strip().lower()
        if decision_type not in {"buy", "hold", "sell"}:
            decision_type = _signal_from_operation_advice(getattr(result, "operation_advice", ""))

        profile = self._build_profile(
            stock_code=str(getattr(result, "code", "") or "").strip(),
            current_signal=decision_type,
            model_used=str(getattr(result, "model_used", "") or "").strip() or None,
            score=int(getattr(result, "sentiment_score", 50) or 50),
            trend_prediction=str(getattr(result, "trend_prediction", "") or "").strip(),
            context_snapshot=context_snapshot,
        )
        result.calibration_info = profile.to_dict()

        if not profile.applied or not profile.suggested_signal:
            return result

        suggested_signal = profile.suggested_signal
        original_signal = decision_type
        updated_score = int(getattr(result, "sentiment_score", 50) or 50) + int(profile.score_adjustment or 0)
        updated_score = max(0, min(100, updated_score))
        if suggested_signal == "buy" and updated_score < 60:
            updated_score = 60
        elif suggested_signal == "sell" and updated_score > 39:
            updated_score = 39
        elif suggested_signal == "hold" and not 40 <= updated_score <= 59:
            updated_score = 50

        if suggested_signal == "buy":
            result.operation_advice = "买入"
            result.trend_prediction = "看多"
        elif suggested_signal == "sell":
            result.operation_advice = "减仓/卖出"
            result.trend_prediction = "看空"
        else:
            result.operation_advice = "观望"
            result.trend_prediction = "震荡"

        result.decision_type = suggested_signal
        result.sentiment_score = updated_score
        if suggested_signal != original_signal:
            result.confidence_level = "低"

        note = (
            f"回测校准：基于{profile.source_scope}近{profile.samples}笔样本，"
            f"{original_signal}历史准确率约{profile.current_signal_stats.accuracy * 100:.0f}%"
            if profile.current_signal_stats
            else f"回测校准：基于{profile.source_scope}历史样本调整本次信号。"
        )
        if profile.reason:
            note = f"{note} {profile.reason}"

        existing_summary = str(getattr(result, "analysis_summary", "") or "").strip()
        result.analysis_summary = f"{existing_summary} {note}".strip() if existing_summary else note

        existing_warning = str(getattr(result, "risk_warning", "") or "").strip()
        if suggested_signal != original_signal:
            warning_note = f"系统已根据历史回测表现，将本次信号从 {original_signal} 调整为 {suggested_signal}。"
            result.risk_warning = f"{existing_warning} {warning_note}".strip() if existing_warning else warning_note

        dashboard = getattr(result, "dashboard", None)
        if isinstance(dashboard, dict):
            dashboard["learning_calibration"] = profile.to_dict()

        from src.analyzer import normalize_analysis_result_signals

        return normalize_analysis_result_signals(result)

    def _build_profile(
        self,
        *,
        stock_code: str,
        current_signal: str,
        model_used: Optional[str],
        score: int,
        trend_prediction: str,
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> CalibrationProfile:
        market = get_market_for_stock(stock_code) or "cn"
        profile = CalibrationProfile(
            enabled=self.enabled,
            market=market,
            model_used=model_used,
            current_signal=current_signal,
            threshold=self.min_samples,
        )
        if not stock_code or current_signal not in {"buy", "hold", "sell"}:
            return profile

        rows = self._load_learning_rows()
        if rows:
            profile = self._build_heuristic_profile(
                stock_code=stock_code,
                current_signal=current_signal,
                model_used=model_used,
                score=score,
                market=market,
                rows=rows,
            )

        model_decision = self._build_model_profile(
            stock_code=stock_code,
            current_signal=current_signal,
            model_used=model_used,
            score=score,
            trend_prediction=trend_prediction,
            context_snapshot=context_snapshot,
        )
        if model_decision is None:
            return profile

        profile.model_prediction = model_decision["prediction_meta"]
        if profile.applied:
            if model_decision["applied"] and model_decision["suggested_signal"] == profile.suggested_signal:
                profile.source_scope = (
                    f"{profile.source_scope}+小型模型" if profile.source_scope else "小型模型"
                )
                profile.score_adjustment += int(model_decision["score_adjustment"] or 0)
                profile.reason = " ".join(
                    token for token in (profile.reason, model_decision["reason"]) if token
                ).strip()
            elif (
                model_decision["applied"]
                and profile.source_scope == "全局"
                and model_decision["confidence"] >= max(0.72, self.model_confidence_threshold + 0.08)
                and model_decision["suggested_signal"] != profile.suggested_signal
            ):
                profile.source_scope = "全局+小型模型"
                profile.suggested_signal = "hold"
                profile.score_adjustment = 0
                profile.reason = " ".join(
                    token
                    for token in (
                        profile.reason,
                        "小型校准模型与全局统计结论冲突，已保守降级为观望。",
                    )
                    if token
                ).strip()
            return profile

        if not model_decision["applied"]:
            return profile

        return CalibrationProfile(
            enabled=self.enabled,
            market=market,
            model_used=model_used,
            source_scope="小型校准模型",
            samples=int(model_decision["prediction_meta"].get("sample_count") or 0),
            threshold=self.model_train_min_samples,
            current_signal=current_signal,
            suggested_signal=model_decision["suggested_signal"],
            score_adjustment=int(model_decision["score_adjustment"] or 0),
            applied=True,
            reason=model_decision["reason"],
            model_prediction=model_decision["prediction_meta"],
        )

        return profile

    def _build_heuristic_profile(
        self,
        *,
        stock_code: str,
        current_signal: str,
        model_used: Optional[str],
        score: int,
        market: str,
        rows: List[Dict[str, Any]],
    ) -> CalibrationProfile:
        profile = CalibrationProfile(
            enabled=self.enabled,
            market=market,
            model_used=model_used,
            current_signal=current_signal,
            threshold=self.min_samples,
        )

        scope_candidates = [
            ("个股", lambda row: row["code"] == stock_code, max(8, self.min_samples // 2)),
            (
                "市场+模型",
                lambda row: row["market"] == market and model_used and row["model_used"] == model_used,
                self.min_samples,
            ),
            ("市场", lambda row: row["market"] == market, self.min_samples),
            ("全局", lambda row: True, self.min_samples),
        ]

        for scope_name, predicate, threshold in scope_candidates:
            scoped_rows = [row for row in rows if predicate(row)]
            decision = self._decide_adjustment(
                rows=scoped_rows,
                current_signal=current_signal,
                threshold=threshold,
                score=score,
            )
            if decision is None:
                continue

            profile.source_scope = scope_name
            profile.samples = decision["samples"]
            profile.current_signal_stats = decision["current_stats"]
            profile.best_alternative_stats = decision["best_alternative_stats"]
            profile.suggested_signal = decision["suggested_signal"]
            profile.score_adjustment = decision["score_adjustment"]
            profile.applied = bool(decision["applied"])
            profile.reason = decision["reason"]
            return profile

        return profile

    def _build_model_profile(
        self,
        *,
        stock_code: str,
        current_signal: str,
        model_used: Optional[str],
        score: int,
        trend_prediction: str,
        context_snapshot: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.model_enabled or self._small_model is None:
            return None

        if (
            self._small_model.validation_accuracy is not None
            and self._small_model.validation_count >= 12
            and self._small_model.baseline_accuracy is not None
            and self._small_model.validation_accuracy < max(0.35, self._small_model.baseline_accuracy - 0.02)
        ):
            return None

        features = self._extract_model_features(
            stock_code=stock_code,
            current_signal=current_signal,
            model_used=model_used,
            score=score,
            trend_prediction=trend_prediction,
            context_snapshot=context_snapshot,
        )
        prediction = self._small_model.predict(features)
        if prediction is None:
            return None

        suggested_signal = prediction.predicted_signal
        confidence = float(prediction.confidence or 0.0)
        applied = False
        score_adjustment = 0
        reason = ""

        if suggested_signal == current_signal:
            if confidence >= self.model_confidence_threshold:
                applied = True
                if suggested_signal == "buy":
                    score_adjustment = 4
                elif suggested_signal == "sell":
                    score_adjustment = -4
                reason = f"小型校准模型同向支持当前 {current_signal} 信号。"
        else:
            high_conf_threshold = max(0.72, self.model_confidence_threshold + 0.10)
            if confidence >= high_conf_threshold:
                applied = True
                if current_signal in {"buy", "sell"} and not 30 <= score <= 70:
                    suggested_signal = "hold"
                    reason = "小型校准模型与当前高强度方向冲突，先保守降级为观望。"
                else:
                    reason = f"小型校准模型认为当前场景更接近 {suggested_signal}。"
                if suggested_signal == "buy":
                    score_adjustment = 9
                elif suggested_signal == "sell":
                    score_adjustment = -9
            elif confidence >= self.model_confidence_threshold and current_signal in {"buy", "sell"}:
                applied = True
                suggested_signal = "hold"
                reason = "小型校准模型对当前方向把握不足，先降级为观望。"

        return {
            "applied": applied,
            "suggested_signal": suggested_signal,
            "score_adjustment": score_adjustment,
            "reason": reason,
            "confidence": confidence,
            "prediction_meta": prediction.to_dict(),
        }

    def _build_training_samples(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordered_rows = sorted(rows, key=lambda item: float(item.get("evaluated_at_ts") or 0.0))
        samples: List[Dict[str, Any]] = []
        for row in ordered_rows:
            label = self._derive_learning_label(row)
            if label not in {"buy", "hold", "sell"}:
                continue
            features = self._extract_model_features(
                stock_code=str(row.get("code") or ""),
                current_signal=str(row.get("signal") or "hold"),
                model_used=row.get("model_used"),
                score=int(row.get("sentiment_score") or 50),
                trend_prediction=str(row.get("trend_prediction") or ""),
                context_snapshot=row.get("context_payload"),
            )
            samples.append({"features": features, "label": label})
        return samples

    def _derive_learning_label(self, row: Dict[str, Any]) -> Optional[str]:
        stock_return_pct = row.get("stock_return_pct")
        if stock_return_pct is not None:
            stock_return = float(stock_return_pct)
            if stock_return >= self.learning_label_band_pct:
                return "buy"
            if stock_return <= -self.learning_label_band_pct:
                return "sell"
            return "hold"

        if row.get("direction_correct") is True:
            return str(row.get("signal") or "hold")
        return None

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clip(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def _unwrap_context_snapshot(context_snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(context_snapshot, dict):
            return {}
        enhanced = context_snapshot.get("enhanced_context")
        if isinstance(enhanced, dict):
            return enhanced
        return context_snapshot

    @staticmethod
    def _provider_bucket(model_used: Optional[str]) -> str:
        provider = str(model_used or "").split("/", 1)[0].strip().lower()
        if provider in {"gemini", "vertex_ai"}:
            return "gemini"
        if provider in {"anthropic", "claude"}:
            return "anthropic"
        if provider in {"openai", "deepseek"}:
            return "openai"
        return "other"

    @staticmethod
    def _trend_bucket(trend_prediction: str) -> str:
        text = str(trend_prediction or "").strip()
        if any(token in text for token in ("看多", "偏多", "bull")):
            return "bull"
        if any(token in text for token in ("看空", "偏空", "bear")):
            return "bear"
        return "neutral"

    @staticmethod
    def _china_exposure_score(level: Any) -> float:
        normalized = str(level or "").strip().lower()
        mapping = {
            "high": 1.0,
            "中": 0.66,
            "medium": 0.66,
            "中等": 0.66,
            "med": 0.66,
            "low": 0.33,
            "低": 0.33,
        }
        return mapping.get(normalized, 0.0)

    def _extract_model_features(
        self,
        *,
        stock_code: str,
        current_signal: str,
        model_used: Optional[str],
        score: int,
        trend_prediction: str,
        context_snapshot: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        enhanced = self._unwrap_context_snapshot(context_snapshot)
        market_context = enhanced.get("market_context") or {}
        realtime = enhanced.get("realtime") or enhanced.get("realtime_quote") or {}
        trend = enhanced.get("trend_analysis") or enhanced.get("trend_result") or {}
        chip = enhanced.get("chip") or enhanced.get("chip_distribution") or {}

        market = str(
            market_context.get("market")
            or enhanced.get("market")
            or get_market_for_stock(stock_code)
            or "cn"
        ).strip().lower()
        signal = current_signal if current_signal in {"buy", "hold", "sell"} else "hold"
        signal_bias = {"buy": 1.0, "hold": 0.0, "sell": -1.0}.get(signal, 0.0)
        provider_bucket = self._provider_bucket(model_used)
        trend_bucket = self._trend_bucket(trend_prediction)
        china_exposure = (
            market_context.get("china_exposure", {})
            if isinstance(market_context.get("china_exposure"), dict)
            else {}
        )

        change_pct = realtime.get("change_pct")
        if change_pct is None:
            change_pct = enhanced.get("price_change_ratio")
        volume_ratio = realtime.get("volume_ratio")
        if volume_ratio is None:
            volume_ratio = enhanced.get("volume_change_ratio", 1.0)

        features = {
            "score_norm": self._clip(self._to_float(score) / 100.0, 0.0, 1.0),
            "current_signal_bias": signal_bias,
            "trend_bull": 1.0 if trend_bucket == "bull" else 0.0,
            "trend_neutral": 1.0 if trend_bucket == "neutral" else 0.0,
            "trend_bear": 1.0 if trend_bucket == "bear" else 0.0,
            "market_cn": 1.0 if market == "cn" else 0.0,
            "market_hk": 1.0 if market == "hk" else 0.0,
            "market_us": 1.0 if market == "us" else 0.0,
            "provider_gemini": 1.0 if provider_bucket == "gemini" else 0.0,
            "provider_anthropic": 1.0 if provider_bucket == "anthropic" else 0.0,
            "provider_openai": 1.0 if provider_bucket == "openai" else 0.0,
            "provider_other": 1.0 if provider_bucket == "other" else 0.0,
            "change_pct_norm": self._clip(self._to_float(change_pct) / 10.0, -3.0, 3.0),
            "volume_ratio_norm": self._clip(self._to_float(volume_ratio, 1.0) / 3.0, 0.0, 3.0),
            "turnover_rate_norm": self._clip(self._to_float(realtime.get("turnover_rate")) / 30.0, 0.0, 3.0),
            "trend_strength_norm": self._clip(self._to_float(trend.get("trend_strength")) / 100.0, 0.0, 1.0),
            "technical_signal_score_norm": self._clip(self._to_float(trend.get("signal_score")) / 100.0, 0.0, 1.0),
            "bias_ma5_norm": self._clip(self._to_float(trend.get("bias_ma5")) / 10.0, -3.0, 3.0),
            "bias_ma10_norm": self._clip(self._to_float(trend.get("bias_ma10")) / 10.0, -3.0, 3.0),
            "chip_profit_ratio_norm": self._clip(self._to_float(chip.get("profit_ratio")) / 100.0, 0.0, 1.0),
            "chip_concentration90_norm": self._clip(self._to_float(chip.get("concentration_90")) / 100.0, 0.0, 1.0),
            "chip_concentration70_norm": self._clip(self._to_float(chip.get("concentration_70")) / 100.0, 0.0, 1.0),
            "china_exposure_norm": self._china_exposure_score(china_exposure.get("level")),
            "is_index_etf": 1.0 if enhanced.get("is_index_etf") else 0.0,
            "data_missing": 1.0 if enhanced.get("data_missing") else 0.0,
        }
        return features

    def _decide_adjustment(
        self,
        *,
        rows: List[Dict[str, Any]],
        current_signal: str,
        threshold: int,
        score: int,
    ) -> Optional[Dict[str, Any]]:
        if not rows:
            return None

        current_stats = self._compute_signal_stats(rows, current_signal)
        alt_candidates = [self._compute_signal_stats(rows, signal) for signal in ("buy", "hold", "sell") if signal != current_signal]
        alt_candidates = [stats for stats in alt_candidates if stats.samples > 0]
        best_alternative = max(alt_candidates, key=lambda item: (item.accuracy, item.samples), default=None)

        if current_signal == "hold":
            if best_alternative is None or best_alternative.samples < threshold:
                return None
            if best_alternative.accuracy >= 0.72 and 45 <= score <= 60:
                score_adjustment = 10 if best_alternative.signal == "buy" else -10
                return {
                    "samples": best_alternative.samples,
                    "current_stats": current_stats,
                    "best_alternative_stats": best_alternative,
                    "suggested_signal": best_alternative.signal,
                    "score_adjustment": score_adjustment,
                    "applied": True,
                    "reason": f"历史上同类场景更常演化为 {best_alternative.signal}。",
                }
            return None

        if current_stats.samples < threshold:
            return None

        suggested_signal = current_signal
        score_adjustment = 0
        applied = False
        reason = ""

        if current_stats.accuracy <= 0.30 and best_alternative and best_alternative.samples >= threshold and best_alternative.accuracy >= max(0.62, current_stats.accuracy + 0.20):
            if 35 <= score <= 70:
                suggested_signal = best_alternative.signal if best_alternative.accuracy >= 0.75 else "hold"
                applied = True
                score_adjustment = -12 if current_signal == "buy" else 12
                reason = f"{current_signal} 历史准确率偏低，而 {best_alternative.signal} 在相同范围内显著更优。"
        elif current_stats.accuracy <= 0.45:
            suggested_signal = "hold"
            applied = True
            score_adjustment = -8 if current_signal == "buy" else 8
            reason = f"{current_signal} 历史准确率偏低，先降级为更保守的观望。"
        elif current_stats.accuracy >= 0.65:
            applied = True
            if current_signal == "buy":
                score_adjustment = 6
            elif current_signal == "sell":
                score_adjustment = -6
            reason = f"{current_signal} 历史准确率较高，适度强化当前结论。"

        if not applied:
            return {
                "samples": current_stats.samples,
                "current_stats": current_stats,
                "best_alternative_stats": best_alternative,
                "suggested_signal": current_signal,
                "score_adjustment": 0,
                "applied": False,
                "reason": "",
            }

        return {
            "samples": current_stats.samples,
            "current_stats": current_stats,
            "best_alternative_stats": best_alternative,
            "suggested_signal": suggested_signal,
            "score_adjustment": score_adjustment,
            "applied": True,
            "reason": reason,
        }

    @staticmethod
    def _compute_signal_stats(rows: Iterable[Dict[str, Any]], signal: str) -> SignalStats:
        matched = [row for row in rows if row["signal"] == signal]
        if not matched:
            return SignalStats(signal=signal)

        accuracy = sum(1 for row in matched if row.get("direction_correct") is True) / len(matched)
        returns = [float(row.get("simulated_return_pct") or 0.0) for row in matched]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        return SignalStats(
            signal=signal,
            samples=len(matched),
            accuracy=accuracy,
            avg_return=avg_return,
        )

    def _load_learning_rows(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        history_limit = int(getattr(self.config, "analysis_learning_history_limit", _DEFAULT_HISTORY_LIMIT))
        with self.db.get_session() as session:
            pairs = session.execute(
                select(BacktestResult, AnalysisHistory)
                .join(AnalysisHistory, AnalysisHistory.id == BacktestResult.analysis_history_id)
                .where(BacktestResult.eval_status == "completed")
                .order_by(desc(BacktestResult.evaluated_at))
                .limit(history_limit)
            ).all()

        rows: List[Dict[str, Any]] = []
        for backtest_row, history_row in pairs:
            payload = self._parse_payload(getattr(history_row, "raw_result", None))
            context_payload = self._parse_payload(getattr(history_row, "context_snapshot", None))
            signal = self._extract_signal(payload, history_row)
            if signal not in {"buy", "hold", "sell"}:
                continue

            rows.append(
                {
                    "code": history_row.code,
                    "market": self._extract_market(history_row.code, payload, context_payload),
                    "model_used": self._extract_model_used(payload),
                    "signal": signal,
                    "sentiment_score": payload.get("sentiment_score", getattr(history_row, "sentiment_score", 50)),
                    "trend_prediction": payload.get("trend_prediction", getattr(history_row, "trend_prediction", "")),
                    "direction_correct": backtest_row.direction_correct,
                    "simulated_return_pct": backtest_row.simulated_return_pct,
                    "stock_return_pct": backtest_row.stock_return_pct,
                    "evaluated_at_ts": backtest_row.evaluated_at.timestamp() if backtest_row.evaluated_at else 0.0,
                    "context_payload": context_payload,
                }
            )
        return rows

    @staticmethod
    def _parse_payload(raw_result: Any) -> Dict[str, Any]:
        if isinstance(raw_result, dict):
            return dict(raw_result)
        if isinstance(raw_result, str) and raw_result.strip():
            try:
                payload = json.loads(raw_result)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return {}
        return {}

    @staticmethod
    def _extract_signal(payload: Dict[str, Any], history_row: AnalysisHistory) -> str:
        decision_type = str(payload.get("decision_type", "") or "").strip().lower()
        if decision_type in {"buy", "hold", "sell"}:
            return decision_type
        return _signal_from_operation_advice(
            payload.get("operation_advice") or getattr(history_row, "operation_advice", "")
        )

    @staticmethod
    def _extract_model_used(payload: Dict[str, Any]) -> Optional[str]:
        model_used = str(payload.get("model_used", "") or "").strip()
        return model_used or None

    @staticmethod
    def _extract_market(code: str, payload: Dict[str, Any], context_payload: Dict[str, Any]) -> str:
        enhanced_payload = context_payload.get("enhanced_context", {}) if isinstance(context_payload, dict) else {}
        market = str((context_payload.get("market_context", {}) or {}).get("market", "")).strip().lower()
        if market in {"cn", "hk", "us"}:
            return market

        enhanced_market = str(
            ((enhanced_payload.get("market_context", {}) or {}).get("market"))
            or enhanced_payload.get("market")
            or ""
        ).strip().lower()
        if enhanced_market in {"cn", "hk", "us"}:
            return enhanced_market

        payload_market = str((payload.get("market_snapshot") or {}).get("market", "")).strip().lower()
        if payload_market in {"cn", "hk", "us"}:
            return payload_market
        return get_market_for_stock(code) or "cn"
