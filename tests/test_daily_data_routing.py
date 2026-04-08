# -*- coding: utf-8 -*-
"""Regression tests for daily-data market routing."""

from __future__ import annotations

import pandas as pd

from data_provider.base import DataFetcherManager


class _DummyFetcher:
    def __init__(self, name: str, result: pd.DataFrame | None = None, error: Exception | None = None):
        self.name = name
        self._result = result
        self._error = error
        self.called = False

    def get_daily_data(self, stock_code: str, start_date=None, end_date=None, days: int = 30):
        self.called = True
        if self._error is not None:
            raise self._error
        return self._result


def test_cn_daily_data_skips_longbridge_fetcher():
    manager = DataFetcherManager.__new__(DataFetcherManager)
    first = _DummyFetcher("EfinanceFetcher", error=RuntimeError("efinance unavailable"))
    longbridge = _DummyFetcher("LongbridgeFetcher", error=AssertionError("should not be called"))
    fallback = _DummyFetcher(
        "BaostockFetcher",
        result=pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-07", "2026-04-08"]),
                "open": [1.0, 1.1],
                "high": [1.1, 1.2],
                "low": [0.9, 1.0],
                "close": [1.05, 1.15],
                "volume": [100, 110],
            }
        ),
    )
    manager._fetchers = [first, longbridge, fallback]

    df, source = manager.get_daily_data("600519")

    assert source == "BaostockFetcher"
    assert not df.empty
    assert first.called is True
    assert longbridge.called is False
    assert fallback.called is True
