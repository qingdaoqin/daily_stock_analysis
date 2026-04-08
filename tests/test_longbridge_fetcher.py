# -*- coding: utf-8 -*-
"""Tests for Longbridge symbol routing helpers."""

from data_provider.longbridge_fetcher import _to_longbridge_symbol


def test_longbridge_accepts_short_hk_numeric_code():
    assert _to_longbridge_symbol("0700") == "0700.HK"


def test_longbridge_accepts_prefixed_hk_code():
    assert _to_longbridge_symbol("HK00700") == "0700.HK"


def test_longbridge_accepts_us_stock_code():
    assert _to_longbridge_symbol("AAPL") == "AAPL.US"
