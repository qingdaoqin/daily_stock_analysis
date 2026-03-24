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
        model_used: str = "gemini/gemini-2.5-flash",
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

        with self.db.get_session() as session:
            history = AnalysisHistory(
                query_id=query_id,
                code=code,
                name=code,
                report_type="simple",
                sentiment_score=70 if signal == "buy" else 30 if signal == "sell" else 50,
                operation_advice=advice_map[signal],
                trend_prediction=trend_map[signal],
                analysis_summary="test",
                raw_result=json.dumps({
                    "decision_type": signal,
                    "operation_advice": advice_map[signal],
                    "trend_prediction": trend_map[signal],
                    "model_used": model_used,
                }, ensure_ascii=False),
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
                )
            )
            session.commit()

    def test_calibrate_result_uses_historical_counter_evidence(self) -> None:
        # US market: sell signals have been wrong, buy signals have been right.
        for idx, code in enumerate(("AAPL", "MSFT", "NVDA"), start=1):
            self._insert_learning_record(
                query_id=f"sell-{idx}",
                code=code,
                signal="sell",
                direction_correct=False,
                simulated_return_pct=-6.0,
            )
        for idx, code in enumerate(("META", "GOOGL", "AMZN", "TSLA"), start=1):
            self._insert_learning_record(
                query_id=f"buy-{idx}",
                code=code,
                signal="buy",
                direction_correct=True,
                simulated_return_pct=8.0,
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
            pipeline.calibration_service.calibrate_result.side_effect = lambda result: result

            result = pipeline.analyze_stock("AAPL", ReportType.SIMPLE, "q-learn")

            self.assertIsNotNone(result)
            pipeline.calibration_service.maybe_refresh_backtests.assert_called_once()
            pipeline.calibration_service.calibrate_result.assert_called_once()


if __name__ == "__main__":
    unittest.main()
