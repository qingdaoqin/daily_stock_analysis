# -*- coding: utf-8 -*-
"""Regression tests for main entry orchestration."""

import argparse
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main


class TestRunFullAnalysis(unittest.TestCase):
    def test_cli_stock_list_suppresses_missing_stock_list_warning(self) -> None:
        config = SimpleNamespace(validate=lambda: ["未配置自选股列表 (STOCK_LIST)", "未配置通知渠道"])

        warnings = main._iter_effective_config_warnings(config, ["AAPL"])

        self.assertEqual(warnings, ["未配置通知渠道"])

    def test_run_full_analysis_errors_when_stock_list_empty(self) -> None:
        config = SimpleNamespace(
            stock_list=[],
            refresh_stock_list=lambda: None,
            market_review_enabled=False,
            single_stock_notify=False,
        )
        args = argparse.Namespace(
            force_run=False,
            no_market_review=False,
            single_notify=False,
            dry_run=False,
            no_notify=True,
            workers=1,
            no_context_snapshot=False,
        )

        with patch.object(main.logger, "error") as mock_error, \
             patch.object(main, "StockAnalysisPipeline") as mock_pipeline:
            main.run_full_analysis(config, args)

        mock_error.assert_called_once_with("未配置自选股列表，请在 .env 文件中设置 STOCK_LIST")
        mock_pipeline.assert_not_called()

    def test_run_full_analysis_rejects_all_unknown_codes_even_with_force_run(self) -> None:
        config = SimpleNamespace(
            stock_list=["BADCODE"],
            refresh_stock_list=lambda: None,
            market_review_enabled=False,
            single_stock_notify=False,
        )
        args = argparse.Namespace(
            force_run=True,
            no_market_review=True,
            single_notify=False,
            dry_run=True,
            no_notify=True,
            workers=1,
            no_context_snapshot=False,
        )

        with patch("src.core.trading_calendar.get_market_for_stock", return_value=None), \
             patch.object(main.logger, "warning") as mock_warning, \
             patch.object(main.logger, "error") as mock_error, \
             patch.object(main, "StockAnalysisPipeline") as mock_pipeline:
            main.run_full_analysis(config, args, stock_codes=["BADCODE"])

        mock_warning.assert_called_once_with("无法识别股票代码，已跳过: %s", "BADCODE")
        mock_error.assert_called_once_with("未识别到有效股票代码，请检查 STOCK_LIST 或命令行 --stocks 参数")
        mock_pipeline.assert_not_called()

    def test_compute_trading_day_filter_skips_unknown_codes(self) -> None:
        config = SimpleNamespace(
            trading_day_check_enabled=True,
            market_review_enabled=False,
            market_review_region="us",
        )
        args = argparse.Namespace(force_run=False, no_market_review=True)

        with patch("src.core.trading_calendar.get_open_markets_today", return_value={"us"}), \
             patch("src.core.trading_calendar.get_market_for_stock", side_effect=[None]), \
             patch.object(main.logger, "warning") as mock_warning:
            filtered_codes, effective_region, should_skip = main._compute_trading_day_filter(
                config, args, ["BADCODE"]
            )

        self.assertEqual(filtered_codes, [])
        self.assertIsNone(effective_region)
        self.assertTrue(should_skip)
        mock_warning.assert_called_once_with("无法识别股票代码，已跳过: %s", "BADCODE")


if __name__ == "__main__":
    unittest.main()
