# -*- coding: utf-8 -*-
"""Tests for market strategy blueprints."""

import unittest

from src.core.market_strategy import get_market_strategy_blueprint
from src.core.trading_calendar import compute_effective_region, get_market_for_stock
from src.market_analyzer import MarketAnalyzer, MarketOverview


class TestMarketStrategyBlueprint(unittest.TestCase):
    """Validate CN/US strategy blueprint basics."""

    def test_cn_blueprint_contains_action_framework(self):
        blueprint = get_market_strategy_blueprint("cn")
        block = blueprint.to_prompt_block()

        self.assertIn("A股市场三段式复盘策略", block)
        self.assertIn("Action Framework", block)
        self.assertIn("进攻", block)

    def test_us_blueprint_contains_regime_strategy(self):
        blueprint = get_market_strategy_blueprint("us")
        block = blueprint.to_prompt_block()

        self.assertIn("US Market Regime Strategy", block)
        self.assertIn("Risk-on", block)
        self.assertIn("Macro & Flows", block)

    def test_hk_blueprint_contains_cross_market_strategy(self):
        blueprint = get_market_strategy_blueprint("hk")
        block = blueprint.to_prompt_block()

        self.assertIn("港股跨市场联动复盘策略", block)
        self.assertIn("南向资金", block)
        self.assertIn("Action Framework", block)


class TestMarketAnalyzerStrategyPrompt(unittest.TestCase):
    """Validate strategy section is injected into prompt/report."""

    def test_cn_prompt_contains_strategy_plan_section(self):
        analyzer = MarketAnalyzer(region="cn")
        prompt = analyzer._build_review_prompt(MarketOverview(date="2026-02-24"), [])

        self.assertIn("策略计划", prompt)
        self.assertIn("A股市场三段式复盘策略", prompt)

    def test_us_prompt_contains_strategy_plan_section(self):
        analyzer = MarketAnalyzer(region="us")
        prompt = analyzer._build_review_prompt(MarketOverview(date="2026-02-24"), [])

        self.assertIn("Strategy Plan", prompt)
        self.assertIn("US Market Regime Strategy", prompt)

    def test_cn_prompt_includes_northbound_flow_when_available(self):
        analyzer = MarketAnalyzer(region="cn")
        prompt = analyzer._build_review_prompt(
            MarketOverview(date="2026-02-24", northbound_flow=35.6),
            [],
        )

        self.assertIn("北向资金净流入: 35.60 亿元", prompt)

    def test_us_prompt_includes_macro_flows_block(self):
        analyzer = MarketAnalyzer(region="us")
        prompt = analyzer._build_review_prompt(
            MarketOverview(
                date="2026-02-24",
                macro_snapshot={
                    "treasury_10y": {"value": 4.32, "unit": "%", "change_pct": 1.25},
                    "dxy": {"value": 103.1, "unit": "", "change_pct": -0.45},
                },
            ),
            [],
        )

        self.assertIn("Macro & Flows", prompt)
        self.assertIn("10Y Treasury Yield: 4.32%", prompt)
        self.assertIn("US Dollar Index: 103.1", prompt)

    def test_hk_prompt_contains_hk_strategy_section(self):
        analyzer = MarketAnalyzer(region="hk")
        prompt = analyzer._build_review_prompt(MarketOverview(date="2026-02-24"), [])

        self.assertIn("港股跨市场联动复盘策略", prompt)
        self.assertIn("港股暂无统一涨跌家数统计", prompt)


class TestMarketReviewRegionResolution(unittest.TestCase):
    def test_compute_effective_region_supports_hk(self):
        self.assertEqual(compute_effective_region("hk", {"hk"}), "hk")

    def test_compute_effective_region_supports_all(self):
        self.assertEqual(compute_effective_region("all", {"cn", "hk", "us"}), "all")

    def test_get_market_for_stock_recognizes_four_digit_hk_code(self):
        self.assertEqual(get_market_for_stock("0700"), "hk")


if __name__ == "__main__":
    unittest.main()
