# -*- coding: utf-8 -*-
"""
Tests for fundamental adapter helpers.
"""

import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.fundamental_adapter import (
    AkshareFundamentalAdapter,
    UsSecFundamentalAdapter,
    _extract_latest_row,
)


class TestFundamentalAdapter(unittest.TestCase):
    def test_extract_latest_row_returns_none_when_code_mismatch(self) -> None:
        df = pd.DataFrame(
            {
                "股票代码": ["600000", "000001"],
                "值": [1, 2],
            }
        )
        row = _extract_latest_row(df, "600519")
        self.assertIsNone(row)

    def test_extract_latest_row_fallback_when_no_code_column(self) -> None:
        df = pd.DataFrame({"值": [1, 2]})
        row = _extract_latest_row(df, "600519")
        self.assertIsNotNone(row)
        self.assertEqual(row["值"], 1)

    def test_dragon_tiger_no_match_with_code_column_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        df = pd.DataFrame(
            {
                "股票代码": ["600000"],
                "日期": ["2026-01-01"],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["is_on_list"])
        self.assertEqual(result["recent_count"], 0)

    def test_dragon_tiger_match_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "日期": [today],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["is_on_list"])
        self.assertGreaterEqual(result["recent_count"], 1)

    def test_northbound_flow_converts_to_yi(self) -> None:
        adapter = AkshareFundamentalAdapter()
        df = pd.DataFrame({"当日净流入": [1_230_000_000]})
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_hsgt_north_net_flow_in_em", [])):
            result = adapter.get_northbound_flow()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["net_inflow"], 12.3)

    def test_us_sec_adapter_extracts_growth_filings_and_institution(self) -> None:
        adapter = UsSecFundamentalAdapter()
        now = pd.Timestamp.now()
        recent_10q = (now - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        recent_form4 = (now - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        recent_13g = (now - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        older_quarter = (now - pd.Timedelta(days=110)).strftime("%Y-%m-%d")

        submissions = {
            "cik": "320193",
            "filings": {
                "recent": {
                    "form": ["10-Q", "4", "13G"],
                    "filingDate": [recent_10q, recent_form4, recent_13g],
                    "accessionNumber": [
                        "0000320193-26-000010",
                        "0000320193-26-000011",
                        "0000320193-26-000012",
                    ],
                    "primaryDocument": ["a10q.htm", "xslF345X05/form4.xml", "schedule13g.htm"],
                }
            },
        }
        companyfacts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-10-01",
                                    "end": "2025-12-31",
                                    "val": 120.0,
                                    "form": "10-Q",
                                    "filed": recent_10q,
                                },
                                {
                                    "start": "2025-07-01",
                                    "end": "2025-09-30",
                                    "val": 100.0,
                                    "form": "10-Q",
                                    "filed": older_quarter,
                                },
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-10-01",
                                    "end": "2025-12-31",
                                    "val": 24.0,
                                    "form": "10-Q",
                                    "filed": recent_10q,
                                },
                                {
                                    "start": "2025-07-01",
                                    "end": "2025-09-30",
                                    "val": 20.0,
                                    "form": "10-Q",
                                    "filed": older_quarter,
                                },
                            ]
                        }
                    },
                    "GrossProfit": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-10-01",
                                    "end": "2025-12-31",
                                    "val": 48.0,
                                    "form": "10-Q",
                                    "filed": recent_10q,
                                }
                            ]
                        }
                    },
                    "StockholdersEquity": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-12-31",
                                    "val": 400.0,
                                    "form": "10-Q",
                                    "filed": recent_10q,
                                }
                            ]
                        }
                    },
                }
            }
        }

        with patch.object(
            adapter,
            "_load_ticker_map",
            return_value={"AAPL": {"cik": "0000320193", "title": "Apple Inc."}},
        ), patch.object(adapter, "_get_json", side_effect=[submissions, companyfacts]):
            result = adapter.get_fundamental_bundle("AAPL")

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["growth"]["revenue_yoy"], 20.0)
        self.assertEqual(result["growth"]["net_profit_yoy"], 20.0)
        self.assertEqual(result["growth"]["roe"], 24.0)
        self.assertEqual(result["growth"]["gross_margin"], 40.0)
        self.assertEqual(result["earnings"]["latest_filing_form"], "10-Q")
        self.assertEqual(result["institution"]["insider_form4_count_90d"], 1)
        self.assertEqual(result["institution"]["ownership_disclosure_count_180d"], 1)


if __name__ == "__main__":
    unittest.main()
