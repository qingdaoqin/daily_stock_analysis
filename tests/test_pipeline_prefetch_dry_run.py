# -*- coding: utf-8 -*-
"""
Regression tests for prefetch behavior in StockAnalysisPipeline.run().
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.core.pipeline import StockAnalysisPipeline


class TestPipelinePrefetchBehavior(unittest.TestCase):
    @staticmethod
    def _build_pipeline(process_result):
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.max_workers = 1
        pipeline.fetcher_manager = MagicMock()
        pipeline.db = MagicMock()
        pipeline.db.has_today_data.return_value = False
        pipeline.process_single_stock = MagicMock(return_value=process_result)
        pipeline.config = SimpleNamespace(
            stock_list=["000001"],
            refresh_stock_list=lambda: None,
            single_stock_notify=False,
            report_type="simple",
            analysis_delay=0,
        )
        return pipeline

    def test_run_dry_run_skips_stock_name_prefetch(self):
        pipeline = self._build_pipeline(process_result=None)

        pipeline.run(stock_codes=["000001"], dry_run=True, send_notification=False)

        pipeline.fetcher_manager.prefetch_stock_names.assert_not_called()

    def test_run_non_dry_run_prefetches_stock_names(self):
        pipeline = self._build_pipeline(process_result=SimpleNamespace(code="000001"))

        pipeline.run(stock_codes=["000001"], dry_run=False, send_notification=False)

        pipeline.fetcher_manager.prefetch_stock_names.assert_called_once_with(
            ["000001"], use_bulk=False
        )

    def test_run_dry_run_counts_process_results_instead_of_today_data(self):
        pipeline = self._build_pipeline(process_result=SimpleNamespace(code="AAPL", success=True))

        with self.assertLogs("src.core.pipeline", level="INFO") as logs:
            pipeline.run(stock_codes=["AAPL"], dry_run=True, send_notification=False)

        self.assertTrue(any("成功: 1, 失败: 0" in message for message in logs.output))

    def test_run_non_dry_run_counts_failed_analysis_results_as_failures(self):
        pipeline = self._build_pipeline(
            process_result=SimpleNamespace(code="PLTA", success=False, error_message="All LLM models failed")
        )

        with self.assertLogs("src.core.pipeline", level="INFO") as logs:
            pipeline.run(stock_codes=["PLTA"], dry_run=False, send_notification=False)

        self.assertTrue(any("成功: 0, 失败: 1" in message for message in logs.output))

    def test_run_skips_unknown_codes_even_when_called_directly(self):
        pipeline = self._build_pipeline(process_result=SimpleNamespace(code="AAPL", success=True))

        with patch("src.core.pipeline.get_market_for_stock", side_effect=[None, "us"]), \
             self.assertLogs("src.core.pipeline", level="WARNING") as logs:
            pipeline.run(stock_codes=["BADCODE", "AAPL"], dry_run=True, send_notification=False)

        pipeline.process_single_stock.assert_called_once()
        self.assertEqual(pipeline.process_single_stock.call_args.args[0], "AAPL")
        self.assertTrue(any("无法识别股票代码，已跳过: BADCODE" in message for message in logs.output))

    def test_process_single_stock_skips_unknown_codes_on_direct_entry(self):
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.max_workers = 1
        pipeline.fetcher_manager = MagicMock()
        pipeline.db = MagicMock()
        pipeline.notifier = MagicMock()
        pipeline.config = SimpleNamespace(
            stock_list=["BADCODE"],
            refresh_stock_list=lambda: None,
            single_stock_notify=False,
            report_type="simple",
            analysis_delay=0,
        )
        pipeline.fetch_and_save_stock_data = MagicMock()
        pipeline.analyze_stock = MagicMock()

        with patch("src.core.pipeline.get_market_for_stock", return_value=None), \
             self.assertLogs("src.core.pipeline", level="WARNING") as logs:
            result = pipeline.process_single_stock("BADCODE", skip_analysis=False)

        self.assertIsNone(result)
        pipeline.fetch_and_save_stock_data.assert_not_called()
        pipeline.analyze_stock.assert_not_called()
        self.assertTrue(any("无法识别股票代码，已跳过: BADCODE" in message for message in logs.output))


if __name__ == "__main__":
    unittest.main()
