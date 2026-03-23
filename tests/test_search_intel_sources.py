# -*- coding: utf-8 -*-
"""Tests for market-specific intelligence source selection."""

import sys
import unittest
from unittest.mock import MagicMock, patch

if "newspaper" not in sys.modules:
    mock_np = MagicMock()
    mock_np.Article = MagicMock()
    mock_np.Config = MagicMock()
    sys.modules["newspaper"] = mock_np

from src.search_service import SearchResponse, SearchResult, SearchService


def _fake_response(query: str) -> SearchResponse:
    return SearchResponse(
        query=query,
        results=[
            SearchResult(
                title="Test",
                snippet="snippet",
                url="https://example.com",
                source="example.com",
                published_date=None,
            )
        ],
        provider="Mock",
        success=True,
    )


class TestSearchIntelSources(unittest.TestCase):
    def _create_service(self):
        service = SearchService(bocha_keys=["dummy"])
        mock_search = MagicMock(side_effect=lambda query, max_results=3, days=7: _fake_response(query))
        service._providers[0].search = mock_search
        return service, mock_search

    def test_us_intel_includes_sec_and_macro_dimensions(self) -> None:
        service, mock_search = self._create_service()

        with patch("src.search_service.time.sleep"):
            results = service.search_comprehensive_intel("AAPL", "Apple", max_searches=6)

        queries = [call.args[0] for call in mock_search.call_args_list]
        self.assertIn("official_filings", results)
        self.assertIn("macro_flows", results)
        self.assertTrue(any("site:sec.gov" in query for query in queries))
        self.assertTrue(any("treasury yield dxy vix" in query or "short interest options" in query for query in queries))

    def test_hk_intel_uses_hkex_official_queries(self) -> None:
        service, mock_search = self._create_service()

        with patch("src.search_service.time.sleep"):
            results = service.search_comprehensive_intel("HK00700", "腾讯控股", max_searches=3)

        queries = [call.args[0] for call in mock_search.call_args_list]
        self.assertIn("official_announcements", results)
        self.assertTrue(any("site:hkexnews.hk" in query or "site:hkex.com.hk" in query for query in queries))
        self.assertFalse(any("site:sec.gov" in query for query in queries))


if __name__ == "__main__":
    unittest.main()
