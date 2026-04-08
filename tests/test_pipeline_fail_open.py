# -*- coding: utf-8 -*-
"""Regression tests for pipeline fail-open and single-stock notification flow."""

import os
import sys
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.pipeline import StockAnalysisPipeline


class TestPipelineFailOpen(unittest.TestCase):
    def test_init_survives_search_service_failure(self) -> None:
        with patch("src.core.pipeline.get_db"), \
             patch("src.core.pipeline.DataFetcherManager"), \
             patch("src.core.pipeline.StockTrendAnalyzer"), \
             patch("src.core.pipeline.GeminiAnalyzer"), \
             patch("src.core.pipeline.NotificationService"), \
             patch("src.core.pipeline.AnalysisCalibrationService"), \
             patch("src.core.pipeline.SearchService", side_effect=RuntimeError("boom")):
            config = SimpleNamespace(
                max_workers=1,
                save_context_snapshot=False,
                bocha_api_keys=[],
                tavily_api_keys=[],
                brave_api_keys=[],
                serpapi_keys=[],
                minimax_api_keys=[],
                news_max_age_days=7,
                enable_realtime_quote=True,
                enable_chip_distribution=True,
                realtime_source_priority=[],
            )
            pipeline = StockAnalysisPipeline(config=config)

        self.assertIsNone(pipeline.search_service)

    def test_run_single_stock_notify_is_serialized_in_main_thread(self) -> None:
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.max_workers = 1
        pipeline.fetcher_manager = MagicMock()
        pipeline.fetcher_manager.prefetch_realtime_quotes.return_value = 0
        pipeline.fetcher_manager.prefetch_stock_names.return_value = None
        pipeline.notifier = MagicMock()
        pipeline.db = MagicMock()
        pipeline._save_local_report = MagicMock()
        pipeline._send_notifications = MagicMock()
        pipeline._send_single_stock_notification = MagicMock()
        pipeline.process_single_stock = MagicMock(
            return_value=SimpleNamespace(code="AAPL", success=True, operation_advice="持有", sentiment_score=60)
        )
        pipeline.config = SimpleNamespace(
            stock_list=["AAPL"],
            refresh_stock_list=lambda: None,
            single_stock_notify=True,
            report_type="simple",
            analysis_delay=0,
        )

        pipeline.run(stock_codes=["AAPL"], dry_run=False, send_notification=True)

        pipeline.process_single_stock.assert_called_once()
        self.assertEqual(pipeline.process_single_stock.call_args.kwargs["single_stock_notify"], False)
        pipeline._send_single_stock_notification.assert_called_once()

    @patch("src.core.pipeline.get_market_today", return_value=date(2026, 4, 7))
    @patch("src.core.pipeline.get_market_for_stock", return_value="us")
    def test_fetch_and_save_stock_data_uses_market_local_today(
        self, _mock_market, _mock_market_today
    ) -> None:
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.fetcher_manager = MagicMock()
        pipeline.fetcher_manager.get_stock_name.return_value = "Apple"
        pipeline.fetcher_manager.get_daily_data.return_value = (MagicMock(empty=False), "YfinanceFetcher")
        pipeline.db = MagicMock()
        pipeline.db.has_today_data.return_value = True

        success, error = pipeline.fetch_and_save_stock_data("AAPL")

        self.assertTrue(success)
        self.assertIsNone(error)
        pipeline.db.has_today_data.assert_called_once_with("AAPL", date(2026, 4, 7))
        pipeline.fetcher_manager.get_daily_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
