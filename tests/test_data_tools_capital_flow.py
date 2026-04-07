# -*- coding: utf-8 -*-
"""Contract tests for get_capital_flow agent tool."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.tools.data_tools import _handle_get_capital_flow


class _DummyManager:
    def get_capital_flow_context(self, _stock_code: str):
        return {
            "status": "ok",
            "main_net_inflow": 1.23e8,
            "super_large_inflow": 5.0e7,
        }


class TestGetCapitalFlowTool(unittest.TestCase):
    def test_capital_flow_context_is_forwarded(self) -> None:
        with patch("src.agent.tools.data_tools._get_fetcher_manager", return_value=_DummyManager()):
            result = _handle_get_capital_flow("600519")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["main_net_inflow"], 1.23e8)


if __name__ == "__main__":
    unittest.main()
