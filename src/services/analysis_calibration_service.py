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

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.current_signal_stats is not None:
            payload["current_signal_stats"] = asdict(self.current_signal_stats)
        if self.best_alternative_stats is not None:
            payload["best_alternative_stats"] = asdict(self.best_alternative_stats)
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

    def calibrate_result(self, result: Any) -> Any:
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
        if not rows:
            return profile

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
            signal = self._extract_signal(payload, history_row)
            if signal not in {"buy", "hold", "sell"}:
                continue

            rows.append(
                {
                    "code": history_row.code,
                    "market": self._extract_market(history_row.code, payload, history_row),
                    "model_used": self._extract_model_used(payload),
                    "signal": signal,
                    "direction_correct": backtest_row.direction_correct,
                    "simulated_return_pct": backtest_row.simulated_return_pct,
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
    def _extract_market(code: str, payload: Dict[str, Any], history_row: AnalysisHistory) -> str:
        context_snapshot = getattr(history_row, "context_snapshot", None)
        if isinstance(context_snapshot, str) and context_snapshot.strip():
            try:
                context_payload = json.loads(context_snapshot)
                market = (
                    context_payload.get("market_context", {}) or {}
                ).get("market")
                if market in {"cn", "hk", "us"}:
                    return market
                enhanced_market = (
                    context_payload.get("enhanced_context", {}).get("market_context", {}) or {}
                ).get("market")
                if enhanced_market in {"cn", "hk", "us"}:
                    return enhanced_market
            except Exception:
                pass

        payload_market = str((payload.get("market_snapshot") or {}).get("market", "")).strip().lower()
        if payload_market in {"cn", "hk", "us"}:
            return payload_market
        return get_market_for_stock(code) or "cn"
