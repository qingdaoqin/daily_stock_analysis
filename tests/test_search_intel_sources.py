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

from src.search_service import SearchResponse, SearchResult, SearchService, XAIXSearchProvider


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

    def test_us_intel_includes_sec_and_china_exposure_dimensions(self) -> None:
        service, mock_search = self._create_service()

        with patch.object(service, "_direct_sec_filings", return_value=SearchResponse(query="sec", results=[], provider="SEC", success=False, error_message="fallback")), \
                patch.object(service, "_direct_us_china_exposure", return_value=SearchResponse(query="china", results=[], provider="SEC", success=False, error_message="fallback")), \
                patch("src.search_service.time.sleep"):
            results = service.search_comprehensive_intel("AAPL", "Apple", max_searches=6)

        queries = [call.args[0] for call in mock_search.call_args_list]
        self.assertIn("official_filings", results)
        self.assertIn("china_exposure", results)
        self.assertTrue(any("site:sec.gov" in query for query in queries))
        self.assertTrue(any("Greater China" in query or "China revenue" in query for query in queries))

    def test_hk_intel_uses_hkex_official_queries(self) -> None:
        service, mock_search = self._create_service()

        with patch.object(service, "_direct_hk_event_calendar", return_value=SearchResponse(query="hk", results=[], provider="HKEX", success=False, error_message="fallback")), \
                patch("src.search_service.time.sleep"):
            results = service.search_comprehensive_intel("HK00700", "腾讯控股", max_searches=3)

        queries = [call.args[0] for call in mock_search.call_args_list]
        self.assertIn("official_announcements", results)
        self.assertTrue(any("site:hkexnews.hk" in query or "site:hkex.com.hk" in query for query in queries))
        self.assertFalse(any("site:sec.gov" in query for query in queries))

    def test_hk_four_digit_code_routes_to_hk_logic(self) -> None:
        service, mock_search = self._create_service()

        with patch.object(service, "_direct_hk_event_calendar", return_value=SearchResponse(query="hk", results=[], provider="HKEX", success=False, error_message="fallback")), \
                patch("src.search_service.time.sleep"):
            results = service.search_comprehensive_intel("0700", "腾讯控股", max_searches=3)

        queries = [call.args[0] for call in mock_search.call_args_list]
        self.assertIn("official_announcements", results)
        self.assertTrue(any("site:hkexnews.hk" in query or "site:hkex.com.hk" in query for query in queries))
        self.assertFalse(any("site:cninfo.com.cn" in query for query in queries))

    def test_us_intel_runs_market_analysis_dimension_by_default(self) -> None:
        service, _ = self._create_service()

        with patch.object(service, "_direct_sec_filings", return_value=SearchResponse(query="sec", results=[], provider="SEC", success=False, error_message="fallback")), \
                patch.object(service, "_direct_us_china_exposure", return_value=SearchResponse(query="china", results=[], provider="SEC", success=False, error_message="fallback")), \
                patch("src.search_service.time.sleep"):
            results = service.search_comprehensive_intel("AAPL", "Apple")

        self.assertIn("market_analysis", results)
        self.assertIn("industry", results)

    def test_us_intel_adds_x_signal_dimension_when_xai_configured(self) -> None:
        service = SearchService(bocha_keys=["dummy"], xai_keys=["xai-test-key"])
        mock_search = MagicMock(side_effect=lambda query, max_results=3, days=7: _fake_response(query))
        service._providers[0].search = mock_search
        x_signal_resp = SearchResponse(
            query="x",
            results=[
                SearchResult(
                    title="X signal",
                    snippet="Social signal",
                    url="https://x.com/i/status/1",
                    source="x.com",
                )
            ],
            provider="xAI X Search",
            success=True,
        )

        with patch.object(service, "_direct_sec_filings", return_value=SearchResponse(query="sec", results=[], provider="SEC", success=False, error_message="fallback")), \
                patch.object(service, "_direct_us_china_exposure", return_value=SearchResponse(query="china", results=[], provider="SEC", success=False, error_message="fallback")), \
                patch.object(service, "_direct_x_social_signal", return_value=x_signal_resp) as mock_x_signal, \
                patch("src.search_service.time.sleep"):
            results = service.search_comprehensive_intel("AAPL", "Apple", max_searches=9)

        self.assertIn("x_signal", results)
        self.assertEqual(results["x_signal"].provider, "xAI X Search")
        mock_x_signal.assert_called_once()

    def test_format_intel_report_surfaces_failed_x_signal_reason(self) -> None:
        service, _ = self._create_service()
        intel_results = {
            "x_signal": SearchResponse(
                query="Apple AAPL X social signal",
                results=[],
                provider="xAI X Search",
                success=False,
                error_message="HTTP 503: upstream overloaded",
            )
        }

        report = service.format_intel_report(intel_results, "Apple")

        self.assertIn("X 社交信号", report)
        self.assertIn("搜索失败: HTTP 503: upstream overloaded", report)

    def test_us_intel_skips_x_signal_dimension_without_xai_configuration(self) -> None:
        service, _ = self._create_service()

        with patch.object(service, "_direct_sec_filings", return_value=SearchResponse(query="sec", results=[], provider="SEC", success=False, error_message="fallback")), \
                patch.object(service, "_direct_us_china_exposure", return_value=SearchResponse(query="china", results=[], provider="SEC", success=False, error_message="fallback")), \
                patch.object(service, "_direct_x_social_signal") as mock_x_signal, \
                patch("src.search_service.time.sleep"):
            results = service.search_comprehensive_intel("AAPL", "Apple", max_searches=9)

        self.assertNotIn("x_signal", results)
        mock_x_signal.assert_not_called()

    def test_cn_official_announcements_can_use_cninfo_direct_without_search_engine(self) -> None:
        service = SearchService()
        direct = SearchResponse(
            query="贵州茅台 600519 公告",
            results=[
                SearchResult(
                    title="贵州茅台关于高级管理人员被实施留置的公告",
                    snippet="官方公告",
                    url="https://static.cninfo.com.cn/test.pdf",
                    source="cninfo.com.cn",
                    published_date="2026-03-14",
                )
            ],
            provider="CNINFO",
            success=True,
        )

        with patch.object(service, "_direct_cninfo_announcements", return_value=direct):
            results = service.search_comprehensive_intel("600519", "贵州茅台", max_searches=2)

        self.assertEqual(results["official_announcements"].provider, "CNINFO")
        self.assertTrue(results["official_announcements"].success)

    def test_us_market_summary_raises_china_policy_weight_when_exposure_is_high(self) -> None:
        service, _ = self._create_service()
        intel_results = {
            "official_filings": SearchResponse(
                query="sec",
                results=[
                    SearchResult(
                        title="Apple Greater China revenue update",
                        snippet="Greater China revenue improved while supply chain remained concentrated in China.",
                        url="https://example.com/sec",
                        source="sec.gov",
                    )
                ],
                provider="Mock",
                success=True,
            ),
            "risk_check": SearchResponse(
                query="risk",
                results=[
                    SearchResult(
                        title="Tariff and export control risks remain",
                        snippet="Tariff and export control pressure still affects Apple manufacturing partners.",
                        url="https://example.com/risk",
                        source="news.example.com",
                    )
                ],
                provider="Mock",
                success=True,
            ),
        }

        summary = service.build_market_intel_summary("AAPL", "Apple", intel_results)

        self.assertEqual(summary["market"], "us")
        self.assertEqual(summary["china_exposure"]["level"], "high")
        self.assertEqual(summary["china_exposure"]["policy_weight"], "high")

    def test_us_market_summary_keeps_policy_weight_guarded_without_exposure_hits(self) -> None:
        service, _ = self._create_service()
        intel_results = {
            "official_filings": SearchResponse(
                query="sec",
                results=[
                    SearchResult(
                        title="US demand remains stable",
                        snippet="Management discussed cloud demand and domestic margins with no mention of China.",
                        url="https://example.com/sec",
                        source="sec.gov",
                    )
                ],
                provider="Mock",
                success=True,
            ),
        }

        summary = service.build_market_intel_summary("MSFT", "Microsoft", intel_results)

        self.assertEqual(summary["china_exposure"]["level"], "unknown")
        self.assertEqual(summary["china_exposure"]["policy_weight"], "guarded")

    def test_us_market_summary_prefers_sec_direct_china_exposure_metadata(self) -> None:
        service, _ = self._create_service()
        intel_results = {
            "china_exposure": SearchResponse(
                query="china exposure",
                results=[
                    SearchResult(
                        title="Apple China exposure (high)",
                        snippet="SEC evidence",
                        url="https://www.sec.gov/example",
                        source="sec.gov",
                        published_date="2026-02-01",
                    )
                ],
                provider="SEC",
                success=True,
                metadata={
                    "china_exposure": {
                        "status": "partial",
                        "level": "high",
                        "signals": ["revenue", "supply_chain"],
                        "evidence": ["中国收入/需求: Greater China net sales remained material."],
                        "filing_form": "10-K",
                        "filing_url": "https://www.sec.gov/example",
                    }
                },
            )
        }

        summary = service.build_market_intel_summary("AAPL", "Apple", intel_results)

        self.assertEqual(summary["china_exposure"]["level"], "high")
        self.assertEqual(summary["china_exposure"]["policy_weight"], "high")
        self.assertTrue(summary["china_exposure"]["evidence"])

    def test_us_market_summary_does_not_fallback_to_snippet_heuristics_when_sec_checked_without_hits(self) -> None:
        service, _ = self._create_service()
        intel_results = {
            "china_exposure": SearchResponse(
                query="china exposure",
                results=[
                    SearchResult(
                        title="Apple China exposure (unknown)",
                        snippet="SEC review found no strong evidence.",
                        url="https://www.sec.gov/example",
                        source="sec.gov",
                        published_date="2026-02-01",
                    )
                ],
                provider="SEC",
                success=True,
                metadata={
                    "china_exposure": {
                        "status": "partial",
                        "level": "unknown",
                        "signals": [],
                        "evidence": [],
                        "filing_form": "10-Q",
                        "filing_url": "https://www.sec.gov/example",
                    }
                },
            ),
            "risk_check": SearchResponse(
                query="risk",
                results=[
                    SearchResult(
                        title="Tariff headlines hit broader tech sector",
                        snippet="Generic tariff headlines mention China but not company-specific exposure.",
                        url="https://example.com/risk",
                        source="news.example.com",
                    )
                ],
                provider="Mock",
                success=True,
            ),
        }

        summary = service.build_market_intel_summary("AAPL", "Apple", intel_results)

        self.assertEqual(summary["china_exposure"]["level"], "unknown")
        self.assertEqual(summary["china_exposure"]["policy_weight"], "guarded")
        self.assertIn("未检索到明确", summary["china_exposure"]["reasoning"])


class TestXAIXSearchProvider(unittest.TestCase):
    def test_xai_provider_parses_inline_citations_into_search_results(self) -> None:
        text = (
            "1. CFO commentary points to stable ad demand — Meta finance discussion remained constructive."
            "[[1]](https://x.com/i/status/123)\n"
            "2. Product rollout drew attention — Threads ads rollout got fresh discussion on X."
            "[[2]](https://x.com/i/status/456)"
        )
        first_start = text.index("[[1]]")
        second_start = text.index("[[2]]")
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://x.com/i/status/123",
                                    "start_index": first_start,
                                    "end_index": first_start + len("[[1]](https://x.com/i/status/123)"),
                                    "title": "1",
                                },
                                {
                                    "type": "url_citation",
                                    "url": "https://x.com/i/status/456",
                                    "start_index": second_start,
                                    "end_index": second_start + len("[[2]](https://x.com/i/status/456)"),
                                    "title": "2",
                                },
                            ],
                        }
                    ],
                }
            ],
            "citations": [
                "https://x.com/i/status/123",
                "https://x.com/i/status/456",
            ],
        }

        mock_http = MagicMock()
        mock_http.status_code = 200
        mock_http.headers = {"content-type": "application/json"}
        mock_http.json.return_value = payload

        provider = XAIXSearchProvider(["xai-test-key"])
        with patch("src.search_service._post_with_retry", return_value=mock_http):
            response = provider.search("Meta META", max_results=2, days=3)

        self.assertTrue(response.success)
        self.assertEqual(response.provider, "xAI X Search")
        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.results[0].url, "https://x.com/i/status/123")
        self.assertIn("stable ad demand", response.results[0].title.lower() + " " + response.results[0].snippet.lower())
        self.assertEqual(response.results[1].url, "https://x.com/i/status/456")


if __name__ == "__main__":
    unittest.main()
