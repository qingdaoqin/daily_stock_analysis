# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Report renderer tests
===================================

Tests for Jinja2 report rendering and fallback behavior.
"""

import sys
import unittest
from unittest.mock import MagicMock

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from src.analyzer import AnalysisResult
from src.services.report_renderer import render, _get_signal_level


def _make_result(
    code: str = "600519",
    name: str = "贵州茅台",
    sentiment_score: int = 72,
    operation_advice: str = "持有",
    analysis_summary: str = "稳健",
    decision_type: str = "hold",
    dashboard: dict = None,
) -> AnalysisResult:
    if dashboard is None:
        dashboard = {
            "core_conclusion": {"one_sentence": "持有观望"},
            "intelligence": {"risk_alerts": []},
            "battle_plan": {"sniper_points": {"stop_loss": "110"}},
        }
    return AnalysisResult(
        code=code,
        name=name,
        trend_prediction="看多",
        sentiment_score=sentiment_score,
        operation_advice=operation_advice,
        analysis_summary=analysis_summary,
        decision_type=decision_type,
        dashboard=dashboard,
    )


class TestReportRenderer(unittest.TestCase):
    """Report renderer tests."""

    def test_get_signal_level_supports_compound_sell_advice(self) -> None:
        result = _make_result(
            code="TSLA",
            name="Tesla",
            sentiment_score=74,
            operation_advice="减仓/卖出",
            decision_type="sell",
        )

        self.assertEqual(_get_signal_level(result), ("卖出", "🔴", "卖出"))

    def test_render_markdown_summary_only(self) -> None:
        """Markdown platform renders with summary_only."""
        r = _make_result()
        out = render("markdown", [r], summary_only=True)
        self.assertIsNotNone(out)
        self.assertIn("决策仪表盘", out)
        self.assertIn("贵州茅台", out)
        self.assertIn("持有", out)

    def test_render_markdown_full(self) -> None:
        """Markdown platform renders full report."""
        r = _make_result()
        out = render("markdown", [r], summary_only=False)
        self.assertIsNotNone(out)
        self.assertIn("核心结论", out)
        self.assertIn("作战计划", out)

    def test_render_wechat(self) -> None:
        """Wechat platform renders."""
        r = _make_result()
        out = render("wechat", [r])
        self.assertIsNotNone(out)
        self.assertIn("贵州茅台", out)

    def test_render_brief(self) -> None:
        """Brief platform renders 3-5 sentence summary."""
        r = _make_result()
        out = render("brief", [r])
        self.assertIsNotNone(out)
        self.assertIn("决策简报", out)
        self.assertIn("贵州茅台", out)

    def test_render_markdown_counts_failed_results_separately(self) -> None:
        ok_result = _make_result(
            code="CRCL",
            name="Circle Internet Group, Inc.",
            sentiment_score=85,
            operation_advice="买入",
            decision_type="buy",
        )
        failed_result = AnalysisResult(
            code="PLTA",
            name="ProShares Ultra PLTR",
            trend_prediction="震荡",
            sentiment_score=50,
            operation_advice="持有",
            analysis_summary="分析过程出错: All LLM models failed",
            decision_type="hold",
            success=False,
            error_message="All LLM models failed (tried 2 model(s)). Last error: litellm.RateLimitError",
        )

        out = render("markdown", [ok_result, failed_result], summary_only=True)

        self.assertIsNotNone(out)
        self.assertIn("🟢买入:1", out)
        self.assertIn("🟡观望:0", out)
        self.assertIn("🔴卖出:0", out)
        self.assertIn("❌失败:1", out)
        self.assertIn("PLTA", out)
        self.assertIn("分析失败", out)

    def test_render_unknown_platform_returns_none(self) -> None:
        """Unknown platform returns None (caller fallback)."""
        r = _make_result()
        out = render("unknown_platform", [r])
        self.assertIsNone(out)

    def test_render_empty_results_returns_content(self) -> None:
        """Empty results still produces header."""
        out = render("markdown", [], summary_only=True)
        self.assertIsNotNone(out)
        self.assertIn("0", out)
