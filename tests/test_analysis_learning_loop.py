# -*- coding: utf-8 -*-
"""Tests for the backtest-driven evolution loop."""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for _mod in ("litellm", "json_repair", "markdown2", "newspaper"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analyzer import AnalysisResult
from src.core.pipeline import StockAnalysisPipeline
from src.enums import ReportType
from src.services.analysis_calibration_service import AnalysisCalibrationService
from src.storage import AnalysisHistory, BacktestResult, DatabaseManager


class AnalysisCalibrationServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = os.path.join(self._temp_dir.name, "learning_loop.db")
        os.environ["DATABASE_PATH"] = self._db_path

        from src.config import Config

        Config._instance = None
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()
        self.config = SimpleNamespace(
            backtest_enabled=True,
            analysis_learning_min_samples=3,
            analysis_learning_history_limit=100,
            analysis_learning_auto_backtest_limit=50,
            analysis_learning_refresh_interval_minutes=60,
            analysis_learning_model_enabled=True,
            analysis_learning_model_path=os.path.join(self._temp_dir.name, "analysis_calibration_model.json"),
            analysis_learning_model_retrain_interval_minutes=60,
            analysis_learning_model_train_min_samples=6,
            analysis_learning_model_confidence_threshold=0.55,
            analysis_learning_label_band_pct=2.0,
            backtest_neutral_band_pct=2.0,
        )

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        self._temp_dir.cleanup()

    def _insert_learning_record(
        self,
        *,
        query_id: str,
        code: str,
        signal: str,
        direction_correct: bool,
        simulated_return_pct: float,
        stock_return_pct: float,
        model_used: str = "gemini/gemini-2.5-flash",
        market: str = "us",
        sentiment_score: int | None = None,
        change_pct: float | None = None,
        trend_strength: float | None = None,
        signal_score: int | None = None,
        bias_ma5: float | None = None,
        china_exposure_level: str = "",
    ) -> None:
        advice_map = {
            "buy": "买入",
            "hold": "观望",
            "sell": "减仓/卖出",
        }
        trend_map = {
            "buy": "看多",
            "hold": "震荡",
            "sell": "看空",
        }
        created_at = datetime(2024, 1, 1, 0, 0, 0)
        score_value = sentiment_score if sentiment_score is not None else (
            72 if signal == "buy" else 28 if signal == "sell" else 50
        )
        context_snapshot = self._make_context_snapshot(
            market=market,
            signal=signal,
            change_pct=change_pct,
            trend_strength=trend_strength,
            signal_score=signal_score,
            bias_ma5=bias_ma5,
            china_exposure_level=china_exposure_level,
        )

        with self.db.get_session() as session:
            history = AnalysisHistory(
                query_id=query_id,
                code=code,
                name=code,
                report_type="simple",
                sentiment_score=score_value,
                operation_advice=advice_map[signal],
                trend_prediction=trend_map[signal],
                analysis_summary="test",
                raw_result=json.dumps({
                    "decision_type": signal,
                    "operation_advice": advice_map[signal],
                    "trend_prediction": trend_map[signal],
                    "model_used": model_used,
                    "sentiment_score": score_value,
                }, ensure_ascii=False),
                context_snapshot=json.dumps(context_snapshot, ensure_ascii=False),
                created_at=created_at,
            )
            session.add(history)
            session.flush()
            session.add(
                BacktestResult(
                    analysis_history_id=history.id,
                    code=code,
                    eval_window_days=10,
                    engine_version="v1",
                    eval_status="completed",
                    evaluated_at=created_at,
                    operation_advice=advice_map[signal],
                    direction_correct=direction_correct,
                    simulated_return_pct=simulated_return_pct,
                    stock_return_pct=stock_return_pct,
                )
            )
            session.commit()

    @staticmethod
    def _make_context_snapshot(
        *,
        market: str,
        signal: str,
        change_pct: float | None = None,
        trend_strength: float | None = None,
        signal_score: int | None = None,
        bias_ma5: float | None = None,
        china_exposure_level: str = "",
    ) -> dict:
        bullish = signal == "buy"
        bearish = signal == "sell"
        return {
            "enhanced_context": {
                "market": market,
                "market_context": {
                    "market": market,
                    "china_exposure": {"level": china_exposure_level} if china_exposure_level else {},
                },
                "realtime": {
                    "change_pct": change_pct if change_pct is not None else (3.6 if bullish else -3.8 if bearish else 0.4),
                    "volume_ratio": 1.4 if bullish else 0.7 if bearish else 1.0,
                    "turnover_rate": 6.0 if bullish else 2.0 if bearish else 3.0,
                },
                "trend_analysis": {
                    "trend_strength": trend_strength if trend_strength is not None else (86 if bullish else 24 if bearish else 50),
                    "signal_score": signal_score if signal_score is not None else (88 if bullish else 22 if bearish else 50),
                    "bias_ma5": bias_ma5 if bias_ma5 is not None else (0.8 if bullish else -0.9 if bearish else 0.0),
                    "bias_ma10": 0.5 if bullish else -0.6 if bearish else 0.0,
                },
                "chip": {
                    "profit_ratio": 78 if bullish else 28 if bearish else 50,
                    "concentration_90": 12 if bullish else 28 if bearish else 18,
                    "concentration_70": 8 if bullish else 20 if bearish else 12,
                },
                "is_index_etf": False,
            }
        }

    def test_calibrate_result_uses_historical_counter_evidence(self) -> None:
        # US market: sell signals have been wrong, buy signals have been right.
        for idx, code in enumerate(("AAPL", "MSFT", "NVDA"), start=1):
            self._insert_learning_record(
                query_id=f"sell-{idx}",
                code=code,
                signal="sell",
                direction_correct=False,
                simulated_return_pct=-6.0,
                stock_return_pct=6.0,
                china_exposure_level="high",
            )
        for idx, code in enumerate(("META", "GOOGL", "AMZN", "TSLA"), start=1):
            self._insert_learning_record(
                query_id=f"buy-{idx}",
                code=code,
                signal="buy",
                direction_correct=True,
                simulated_return_pct=8.0,
                stock_return_pct=8.0,
                china_exposure_level="high",
            )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        result = AnalysisResult(
            code="AAPL",
            name="Apple Inc.",
            sentiment_score=38,
            trend_prediction="看空",
            operation_advice="减仓/卖出",
            decision_type="sell",
            analysis_summary="原始分析认为短线应减仓。",
            success=True,
            model_used="gemini/gemini-2.5-flash",
        )

        calibrated = service.calibrate_result(result)

        self.assertEqual(calibrated.decision_type, "buy")
        self.assertIn("回测校准", calibrated.analysis_summary)
        self.assertIsInstance(calibrated.calibration_info, dict)
        self.assertTrue(calibrated.calibration_info.get("applied"))
        self.assertEqual(calibrated.calibration_info.get("source_scope"), "市场+模型")
        self.assertGreaterEqual(calibrated.sentiment_score, 60)

    def test_maybe_refresh_backtests_respects_cooldown(self) -> None:
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config, time_fn=lambda: 1000.0)
        with patch("src.services.analysis_calibration_service.BacktestService.run_backtest", return_value={"processed": 0, "saved": 0, "completed": 0, "insufficient": 0, "errors": 0}) as mock_run:
            service.maybe_refresh_backtests()
            service.maybe_refresh_backtests()
        self.assertEqual(mock_run.call_count, 1)

    def test_small_model_trains_and_calibrates_hold_to_buy(self) -> None:
        self.config.analysis_learning_min_samples = 10
        self.config.analysis_learning_model_train_min_samples = 6

        for idx, code in enumerate(("AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL"), start=1):
            self._insert_learning_record(
                query_id=f"bull-{idx}",
                code=code,
                signal="buy",
                direction_correct=True,
                simulated_return_pct=7.0,
                stock_return_pct=7.0,
                change_pct=4.2,
                trend_strength=90,
                signal_score=92,
                bias_ma5=1.1,
                china_exposure_level="high",
            )
        for idx, code in enumerate(("BABA", "NIO", "PDD", "JD", "BIDU", "LI"), start=1):
            self._insert_learning_record(
                query_id=f"bear-{idx}",
                code=code,
                signal="sell",
                direction_correct=True,
                simulated_return_pct=0.0,
                stock_return_pct=-7.0,
                change_pct=-4.1,
                trend_strength=20,
                signal_score=18,
                bias_ma5=-1.2,
                china_exposure_level="low",
            )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        train_stats = service.maybe_refresh_model()
        self.assertTrue(train_stats.get("success"))
        self.assertTrue(os.path.exists(self.config.analysis_learning_model_path))

        result = AnalysisResult(
            code="AAPL",
            name="Apple Inc.",
            sentiment_score=54,
            trend_prediction="震荡",
            operation_advice="观望",
            decision_type="hold",
            analysis_summary="原始分析偏中性。",
            success=True,
            model_used="gemini/gemini-2.5-flash",
        )
        calibrated = service.calibrate_result(
            result,
            context_snapshot=self._make_context_snapshot(
                market="us",
                signal="buy",
                change_pct=4.5,
                trend_strength=94,
                signal_score=95,
                bias_ma5=1.2,
                china_exposure_level="high",
            ),
        )

        self.assertEqual(calibrated.decision_type, "buy")
        self.assertEqual(calibrated.operation_advice, "买入")
        self.assertIsInstance(calibrated.calibration_info, dict)
        self.assertIn("model_prediction", calibrated.calibration_info)
        self.assertEqual(calibrated.calibration_info.get("source_scope"), "小型校准模型")


class PipelineLearningLoopTestCase(unittest.TestCase):
    def test_pipeline_calls_refresh_and_calibration_on_standard_path(self) -> None:
        with patch("src.core.pipeline.get_config") as mock_config, \
             patch("src.core.pipeline.get_db") as mock_db, \
             patch("src.core.pipeline.DataFetcherManager"), \
             patch("src.core.pipeline.GeminiAnalyzer"), \
             patch("src.core.pipeline.NotificationService"), \
             patch("src.core.pipeline.SearchService"):

            mock_cfg = MagicMock()
            mock_cfg.max_workers = 1
            mock_cfg.agent_mode = False
            mock_cfg.agent_skills = []
            mock_cfg.bocha_api_keys = []
            mock_cfg.tavily_api_keys = []
            mock_cfg.brave_api_keys = []
            mock_cfg.serpapi_keys = []
            mock_cfg.minimax_api_keys = []
            mock_cfg.news_max_age_days = 3
            mock_cfg.enable_realtime_quote = False
            mock_cfg.enable_chip_distribution = False
            mock_cfg.realtime_source_priority = []
            mock_cfg.save_context_snapshot = False
            mock_cfg.backtest_enabled = True
            mock_config.return_value = mock_cfg

            pipeline = StockAnalysisPipeline(config=mock_cfg)
            pipeline.calibration_service = MagicMock()
            pipeline.calibration_service.enabled = True
            pipeline.calibration_service.refresh_interval_seconds = 0
            pipeline.calibration_service._last_refresh_ts = 0.0
            pipeline.calibration_service.maybe_refresh_model = MagicMock()

            pipeline.fetcher_manager.get_stock_name.return_value = "Apple Inc."
            pipeline.fetcher_manager.get_realtime_quote.return_value = None
            pipeline.fetcher_manager.get_chip_distribution.return_value = None
            pipeline.fetcher_manager.get_fundamental_context.return_value = {"source_chain": [], "coverage": {}}
            pipeline.search_service.can_run_comprehensive_intel.return_value = False
            pipeline.search_service.build_market_intel_summary.return_value = {"market": "us", "market_label": "美股"}
            pipeline.db.get_data_range.return_value = []
            pipeline.db.get_analysis_context.return_value = {
                "code": "AAPL",
                "stock_name": "Apple Inc.",
                "date": "2026-03-24",
                "today": {},
                "yesterday": {},
            }

            raw_result = AnalysisResult(
                code="AAPL",
                name="Apple Inc.",
                sentiment_score=62,
                trend_prediction="看多",
                operation_advice="买入",
                decision_type="buy",
                analysis_summary="test",
            )
            pipeline.analyzer.analyze.return_value = raw_result
            pipeline.calibration_service.calibrate_result.side_effect = (
                lambda result, context_snapshot=None: result
            )

            result = pipeline.analyze_stock("AAPL", ReportType.SIMPLE, "q-learn")

            self.assertIsNotNone(result)
            pipeline.calibration_service.maybe_refresh_backtests.assert_called_once()
            pipeline.calibration_service.maybe_refresh_model.assert_called_once()
            pipeline.calibration_service.calibrate_result.assert_called_once()


if __name__ == "__main__":
    unittest.main()
