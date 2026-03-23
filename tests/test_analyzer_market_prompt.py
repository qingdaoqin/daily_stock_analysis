# -*- coding: utf-8 -*-
"""Tests for market-aware analyzer prompts."""

import sys
import unittest
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from src.analyzer import GeminiAnalyzer


class TestAnalyzerMarketPrompt(unittest.TestCase):
    def _make_analyzer(self) -> GeminiAnalyzer:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer()
        analyzer._litellm_available = False
        return analyzer

    def test_system_prompt_mentions_all_markets(self) -> None:
        analyzer = self._make_analyzer()

        self.assertIn("A 股、港股和美股", analyzer.SYSTEM_PROMPT)
        self.assertIn("China exposure", analyzer.SYSTEM_PROMPT)
        self.assertNotIn("A 股投资分析师", analyzer.SYSTEM_PROMPT.splitlines()[0])

    def test_us_prompt_includes_china_exposure_guidance(self) -> None:
        analyzer = self._make_analyzer()
        context = {
            "code": "AAPL",
            "stock_name": "Apple",
            "date": "2026-03-23",
            "today": {"close": 180.0, "ma5": 178.0, "ma10": 175.0, "ma20": 170.0},
            "market_context": {
                "market": "us",
                "market_label": "美股",
                "official_source_priority": "SEC 披露、财报电话会、公司指引",
                "analysis_focus": "先看 SEC/财报/指引，再判断政策变量。",
                "policy_scope": "中国政策只有在存在 China exposure 时才提高权重。",
                "china_exposure": {
                    "level_label": "高",
                    "policy_weight_label": "高权重",
                    "reasoning": "同时存在中国收入和供应链暴露。",
                },
            },
        }

        prompt = analyzer._format_prompt(context, "Apple", news_context="news")

        self.assertIn("当前市场：美股", prompt)
        self.assertIn("China exposure：高", prompt)
        self.assertIn("中国政策权重：高权重", prompt)
        self.assertIn("SEC/财报/公司指引", prompt)
        self.assertIn("SEC/财报/公司指引是否出现超预期或下修", prompt)

    def test_cn_prompt_keeps_a_share_policy_focus(self) -> None:
        analyzer = self._make_analyzer()
        context = {
            "code": "600519",
            "stock_name": "贵州茅台",
            "date": "2026-03-23",
            "today": {"close": 1500.0, "ma5": 1490.0, "ma10": 1480.0, "ma20": 1460.0},
            "market_context": {
                "market": "cn",
                "market_label": "A股",
                "official_source_priority": "巨潮/沪深交易所公告、业绩预告、监管函",
                "analysis_focus": "政策、监管、业绩预告与资金流本身就是 A 股核心变量。",
                "policy_scope": "中国政策与产业监管属于核心主因。",
            },
        }

        prompt = analyzer._format_prompt(context, "贵州茅台", news_context=None)

        self.assertIn("当前市场：A股", prompt)
        self.assertIn("中国政策与产业监管属于核心主因", prompt)
        self.assertIn("是否满足 MA5>MA10>MA20 多头排列", prompt)

    def test_prompt_keeps_markdown_tables_but_removes_prompt_emojis(self) -> None:
        analyzer = self._make_analyzer()
        context = {
            "code": "AAPL",
            "stock_name": "Apple",
            "date": "2026-03-23",
            "today": {"close": 180.0, "ma5": 178.0, "ma10": 175.0, "ma20": 170.0},
            "market_context": {
                "market": "us",
                "market_label": "美股",
                "official_source_priority": "SEC 披露、财报电话会、公司指引",
                "analysis_focus": "先看 SEC/财报/指引，再判断政策变量。",
                "policy_scope": "中国政策只有在存在 China exposure 时才提高权重。",
            },
        }

        prompt = analyzer._format_prompt(context, "Apple", news_context="news")

        self.assertIn("| 项目 | 数据 |", prompt)
        self.assertIn("## 股票基础信息", prompt)
        self.assertNotIn("## 📊 股票基础信息", prompt)
        self.assertNotIn("## 📰 舆情情报", prompt)
        self.assertNotIn("❓", prompt)
        self.assertNotIn("✅", prompt)
        self.assertNotIn("⚠️", prompt)
        self.assertNotIn("⚪", prompt)
        self.assertNotIn("❌", prompt)


if __name__ == "__main__":
    unittest.main()
