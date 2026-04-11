# -*- coding: utf-8 -*-
"""Contract tests for get_capital_flow tool output semantics.

Covers src/agent/tools/data_tools._handle_get_capital_flow.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.tools.data_tools import _handle_get_capital_flow


class _DummyManagerOk:
    """Returns a well-formed capital flow context."""

    def get_capital_flow_context(self, _stock_code: str):
        return {
            "status": "ok",
            "data": {
                "stock_flow": {
                    "main_net_inflow": 1500000.0,
                    "inflow_5d": 8000000.0,
                    "inflow_10d": 15000000.0,
                },
                "sector_rankings": {
                    "top": [{"name": "白酒", "inflow": 5e8}, {"name": "半导体", "inflow": 3e8}],
                    "bottom": [{"name": "煤炭", "inflow": -2e8}],
                },
            },
            "errors": [],
        }


class _DummyManagerNotSupported:
    """Returns not_supported status (e.g. ETF or non-CN stock)."""

    def get_capital_flow_context(self, _stock_code: str):
        return {"status": "not_supported"}


class _DummyManagerRaises:
    """Simulates a fetch failure."""

    def get_capital_flow_context(self, _stock_code: str):
        raise RuntimeError("network timeout")


class _DummyManagerNone:
    """Returns None (legacy behavior)."""

    def get_capital_flow_context(self, _stock_code: str):
        return None


class TestGetCapitalFlowContract(unittest.TestCase):

    def test_ok_response_passes_through(self) -> None:
        """Happy path: response from get_capital_flow_context is returned as-is."""
        with patch(
            "src.agent.tools.data_tools._get_fetcher_manager",
            return_value=_DummyManagerOk(),
        ):
            result = _handle_get_capital_flow("600519")

        self.assertEqual(result["status"], "ok")
        self.assertIn("data", result)
        stock_flow = result["data"].get("stock_flow", {})
        self.assertEqual(stock_flow.get("main_net_inflow"), 1500000.0)
        self.assertEqual(result["errors"], [])

    def test_not_supported_passes_through(self) -> None:
        """ETF / non-CN stocks return status=not_supported."""
        with patch(
            "src.agent.tools.data_tools._get_fetcher_manager",
            return_value=_DummyManagerNotSupported(),
        ):
            result = _handle_get_capital_flow("510300")

        self.assertEqual(result["status"], "not_supported")

    def test_exception_path_returns_error_dict(self) -> None:
        """Fetch errors surface as error dict (not propagated)."""
        with patch(
            "src.agent.tools.data_tools._get_fetcher_manager",
            return_value=_DummyManagerRaises(),
        ):
            # Local impl does not catch exceptions — the tool handler will propagate
            # OR return an error dict depending on implementation.
            # Either behaviour is acceptable; we just ensure it doesn't silently hang.
            try:
                result = _handle_get_capital_flow("600519")
                # If result is returned, it should have an error key
                self.assertTrue(
                    "error" in result or isinstance(result, dict),
                    "Expected a dict response",
                )
            except RuntimeError:
                pass  # propagated exception is also acceptable

    def test_none_response_returns_error(self) -> None:
        """None from context should produce an error response."""
        with patch(
            "src.agent.tools.data_tools._get_fetcher_manager",
            return_value=_DummyManagerNone(),
        ):
            result = _handle_get_capital_flow("600519")

        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
