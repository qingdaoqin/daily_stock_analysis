# -*- coding: utf-8 -*-
"""
Tests for TickFlowFetcher.

Covers:
- _safe_float / _ratio_to_percent / _extract_name
- _is_universe_permission_error
- get_main_indices: 正常 / 无 client / 空结果 / 权限拒绝
- get_market_stats: 正常 / 负缓存 / 权限不足 / 无 client
- close() 生命周期管理
- _fetch_raw_data / _normalize_data raises DataFetchError
"""
import pytest
from time import monotonic
from unittest.mock import MagicMock, patch, PropertyMock

from data_provider.base import DataFetchError
from data_provider.tickflow_fetcher import (
    TickFlowFetcher,
    _UNIVERSE_PERMISSION_NEGATIVE_CACHE_TTL_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fetcher(api_key: str = "test-key") -> TickFlowFetcher:
    return TickFlowFetcher(api_key=api_key)


def _mock_client(quotes=None, universe=None, universe_exc=None):
    client = MagicMock()
    client.quotes.get.return_value = quotes or []
    if universe_exc:
        client.universe.list.side_effect = universe_exc
    else:
        client.universe.list.return_value = universe or []
    return client


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_safe_float_normal(self):
        f = TickFlowFetcher._safe_float
        assert f(1.5) == 1.5
        assert f("3.14") == pytest.approx(3.14)

    def test_safe_float_none_dash(self):
        f = TickFlowFetcher._safe_float
        assert f(None) is None
        assert f("") is None
        assert f("-") is None

    def test_ratio_to_percent(self):
        r = TickFlowFetcher._ratio_to_percent
        assert r(0.05) == pytest.approx(5.0)
        assert r(None) is None

    def test_extract_name_from_ext(self):
        quote = {"ext": {"name": "上证指数"}, "name": "fallback"}
        assert TickFlowFetcher._extract_name(quote) == "上证指数"

    def test_extract_name_fallback(self):
        quote = {"name": "沪深300"}
        assert TickFlowFetcher._extract_name(quote) == "沪深300"

    def test_extract_name_empty(self):
        assert TickFlowFetcher._extract_name({}) == ""


# ---------------------------------------------------------------------------
# _is_universe_permission_error
# ---------------------------------------------------------------------------

class TestPermissionError:
    def test_403_status_code(self):
        exc = Exception("forbidden")
        exc.status_code = 403
        assert TickFlowFetcher._is_universe_permission_error(exc)

    def test_permission_denied_code(self):
        exc = Exception("no access")
        exc.status_code = None
        exc.code = "PERMISSION_DENIED"
        assert TickFlowFetcher._is_universe_permission_error(exc)

    def test_permission_keyword_in_message(self):
        exc = Exception("permission denied for this universe query")
        assert TickFlowFetcher._is_universe_permission_error(exc)

    def test_non_permission_error(self):
        exc = Exception("network timeout")
        assert not TickFlowFetcher._is_universe_permission_error(exc)


# ---------------------------------------------------------------------------
# _fetch_raw_data / _normalize_data
# ---------------------------------------------------------------------------

class TestNotSupported:
    def test_fetch_raw_raises(self):
        f = _make_fetcher()
        with pytest.raises(DataFetchError, match="P0"):
            f._fetch_raw_data("000001", "2024-01-01", "2024-01-31")

    def test_normalize_raises(self):
        import pandas as pd
        f = _make_fetcher()
        with pytest.raises(DataFetchError, match="P0"):
            f._normalize_data(pd.DataFrame(), "000001")


# ---------------------------------------------------------------------------
# get_main_indices
# ---------------------------------------------------------------------------

class TestGetMainIndices:
    def test_no_api_key_returns_none(self):
        f = TickFlowFetcher(api_key="")
        assert f.get_main_indices() is None

    def test_non_cn_region_returns_none(self):
        f = _make_fetcher()
        assert f.get_main_indices(region="us") is None

    def test_empty_quotes_returns_none(self):
        f = _make_fetcher()
        client = _mock_client(quotes=[])
        with patch.object(f, "_get_client", return_value=client):
            result = f.get_main_indices()
        assert result is None

    def test_normal_result(self):
        f = _make_fetcher()
        quote = {
            "symbol": "000001.SH",
            "last": "3280.12",
            "prev_close": "3200.0",
            "volume": "1000000",
            "amount": "50000000",
            "ext": {"name": "上证指数"},
        }
        client = MagicMock()
        # 6 symbols → 2 batches (5 + 1); only first batch returns the quote
        client.quotes.get.side_effect = [[quote], []]
        with patch.object(f, "_get_client", return_value=client):
            result = f.get_main_indices()
        assert result is not None
        assert len(result) == 1
        idx = result[0]
        assert idx["code"] == "000001"
        assert idx["name"] == "上证指数"
        assert idx["current"] == pytest.approx(3280.12)
        assert idx["change"] is not None
        assert idx["change_pct"] is not None


    def test_unknown_symbol_skipped(self):
        f = _make_fetcher()
        quote = {"symbol": "UNKNOWN.XX", "last": "100", "prev_close": "100"}
        client = _mock_client(quotes=[quote])
        with patch.object(f, "_get_client", return_value=client):
            result = f.get_main_indices()
        assert result is None  # all skipped => None

    def test_batch_exception_continues(self):
        """If one batch fails, should continue with subsequent batches."""
        f = _make_fetcher()
        client = MagicMock()
        # First call raises, subsequent calls return empty
        client.quotes.get.side_effect = [RuntimeError("batch failed"), [], [], [], [], []]
        with patch.object(f, "_get_client", return_value=client):
            # Should not raise
            result = f.get_main_indices()
        assert result is None  # all empty after failure


# ---------------------------------------------------------------------------
# get_market_stats
# ---------------------------------------------------------------------------

STOCK_UP = {
    "symbol": "600519.SH",
    "last": "1800",
    "prev_close": "1700",
    "change_ratio": "0.06",
    "is_limit_up": False,
    "is_limit_down": False,
    "amount": 1_000_000,
}
STOCK_DOWN = {
    "symbol": "000001.SZ",
    "last": "9.5",
    "prev_close": "10",
    "change_ratio": "-0.05",
    "is_limit_down": True,
    "is_limit_up": False,
    "amount": 500_000,
}
STOCK_FLAT = {
    "symbol": "603259.SH",
    "last": "100",
    "prev_close": "100",
    "change_ratio": "0",
    "is_limit_up": False,
    "is_limit_down": False,
    "amount": 0,
}


class TestGetMarketStats:
    def test_no_api_key_returns_none(self):
        f = TickFlowFetcher(api_key="")
        assert f.get_market_stats() is None

    def test_non_cn_region_returns_none(self):
        f = _make_fetcher()
        assert f.get_market_stats(region="us") is None

    def test_normal_stats(self):
        f = _make_fetcher()
        client = _mock_client(universe=[STOCK_UP, STOCK_DOWN, STOCK_FLAT])
        with patch.object(f, "_get_client", return_value=client):
            stats = f.get_market_stats()
        assert stats is not None
        assert stats["up_count"] == 1
        assert stats["down_count"] == 1
        assert stats["flat_count"] == 1
        assert stats["limit_down_count"] == 1
        assert stats["limit_up_count"] == 0

    def test_permission_error_sets_negative_cache(self):
        f = _make_fetcher()
        perm_exc = Exception("permission denied")
        perm_exc.status_code = 403
        client = _mock_client(universe_exc=perm_exc)

        with patch.object(f, "_get_client", return_value=client):
            result = f.get_market_stats()
        assert result is None
        assert f._universe_query_supported is False
        assert f._universe_query_checked_at is not None

    def test_negative_cache_prevents_retry(self):
        f = _make_fetcher()
        f._universe_query_supported = False
        f._universe_query_checked_at = monotonic()  # just set, within TTL

        client = _mock_client(universe=[STOCK_UP])
        with patch.object(f, "_get_client", return_value=client):
            result = f.get_market_stats()
        assert result is None
        # universe.list should NOT have been called
        client.universe.list.assert_not_called()

    def test_negative_cache_expires_and_retries(self):
        f = _make_fetcher()
        # Set cache time far in the past (beyond TTL)
        f._universe_query_supported = False
        f._universe_query_checked_at = (
            monotonic() - _UNIVERSE_PERMISSION_NEGATIVE_CACHE_TTL_SECONDS - 1
        )

        client = _mock_client(universe=[STOCK_UP])
        with patch.object(f, "_get_client", return_value=client):
            result = f.get_market_stats()
        # Should have retried and succeeded
        assert result is not None
        assert f._universe_query_supported is True

    def test_non_equity_symbols_skipped(self):
        """Non-SH/SZ equity symbols (e.g., HK or US) should be skipped."""
        non_equity = {
            "symbol": "HK00700",  # not a 6-digit .SH/.SZ code
            "change_ratio": "0.1",
            "is_limit_up": False,
            "is_limit_down": False,
        }
        f = _make_fetcher()
        client = _mock_client(universe=[non_equity])
        with patch.object(f, "_get_client", return_value=client):
            stats = f.get_market_stats()
        # All skipped -> zero counts but still returned (not None)
        assert stats is not None
        assert stats["up_count"] == 0


# ---------------------------------------------------------------------------
# close() lifecycle
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_calls_client_close(self):
        f = _make_fetcher()
        mock_client = MagicMock()
        f._client = mock_client
        f.close()
        mock_client.close.assert_called_once()
        assert f._client is None

    def test_close_idempotent_without_client(self):
        f = _make_fetcher()
        # Should not raise even when client never created
        f.close()
        f.close()

    def test_close_suppresses_client_error(self):
        f = _make_fetcher()
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("already closed")
        f._client = mock_client
        # Should not propagate
        f.close()
        assert f._client is None
