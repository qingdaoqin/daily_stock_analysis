# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Schema parsing and fallback tests
===================================

Tests for AnalysisReportSchema validation and analyzer fallback behavior.
"""

import json
import sys
import unittest
from unittest.mock import MagicMock

# Mock litellm before importing analyzer (optional runtime dep)
try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from src.schemas.report_schema import AnalysisReportSchema
from src.analyzer import GeminiAnalyzer, AnalysisResult


class TestAnalysisReportSchema(unittest.TestCase):
    """Schema parsing tests."""

    def test_valid_dashboard_parses(self) -> None:
        """Valid LLM-like JSON parses successfully."""
        data = {
            "stock_name": "贵州茅台",
            "sentiment_score": 75,
            "trend_prediction": "看多",
            "operation_advice": "持有",
            "decision_type": "hold",
            "confidence_level": "中",
            "dashboard": {
                "core_conclusion": {"one_sentence": "持有观望"},
                "intelligence": {"risk_alerts": []},
                "battle_plan": {"sniper_points": {"stop_loss": "110元"}},
            },
            "analysis_summary": "基本面稳健",
        }
        schema = AnalysisReportSchema.model_validate(data)
        self.assertEqual(schema.stock_name, "贵州茅台")
        self.assertEqual(schema.sentiment_score, 75)
        self.assertIsNotNone(schema.dashboard)

    def test_schema_allows_optional_fields_missing(self) -> None:
        """Schema accepts minimal valid structure."""
        data = {
            "stock_name": "测试",
            "sentiment_score": 50,
            "trend_prediction": "震荡",
            "operation_advice": "观望",
        }
        schema = AnalysisReportSchema.model_validate(data)
        self.assertIsNone(schema.dashboard)
        self.assertIsNone(schema.analysis_summary)

    def test_schema_allows_numeric_strings(self) -> None:
        """Schema accepts string values for numeric fields (LLM may return N/A)."""
        data = {
            "stock_name": "测试",
            "sentiment_score": 60,
            "trend_prediction": "看多",
            "operation_advice": "买入",
            "dashboard": {
                "data_perspective": {
                    "price_position": {
                        "current_price": "N/A",
                        "bias_ma5": "2.5",
                    }
                }
            },
        }
        schema = AnalysisReportSchema.model_validate(data)
        self.assertIsNotNone(schema.dashboard)
        pp = schema.dashboard and schema.dashboard.data_perspective and schema.dashboard.data_perspective.price_position
        self.assertIsNotNone(pp)
        if pp:
            self.assertEqual(pp.current_price, "N/A")
            self.assertEqual(pp.bias_ma5, "2.5")

    def test_schema_fails_on_invalid_sentiment_score(self) -> None:
        """Schema validation fails when sentiment_score out of range."""
        data = {
            "stock_name": "测试",
            "sentiment_score": 150,  # out of 0-100
            "trend_prediction": "看多",
            "operation_advice": "买入",
        }
        with self.assertRaises(Exception):
            AnalysisReportSchema.model_validate(data)


class TestAnalyzerSchemaFallback(unittest.TestCase):
    """Analyzer fallback when schema validation fails."""

    def test_parse_response_continues_when_schema_fails(self) -> None:
        """When schema validation fails, analyzer continues with raw dict."""
        analyzer = GeminiAnalyzer()
        response = json.dumps({
            "stock_name": "贵州茅台",
            "sentiment_score": 150,  # invalid for schema
            "trend_prediction": "看多",
            "operation_advice": "持有",
            "analysis_summary": "测试摘要",
        })
        result = analyzer._parse_response(response, "600519", "贵州茅台")
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.code, "600519")
        self.assertEqual(result.decision_type, "hold")
        self.assertTrue(40 <= result.sentiment_score <= 59)
        self.assertTrue(result.success)

    def test_parse_response_valid_json_succeeds(self) -> None:
        """Valid JSON produces correct AnalysisResult."""
        analyzer = GeminiAnalyzer()
        response = json.dumps({
            "stock_name": "贵州茅台",
            "sentiment_score": 72,
            "trend_prediction": "看多",
            "operation_advice": "持有",
            "decision_type": "hold",
            "confidence_level": "高",
            "analysis_summary": "技术面向好",
        })
        result = analyzer._parse_response(response, "600519", "股票600519")
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.name, "贵州茅台")
        self.assertEqual(result.decision_type, "hold")
        self.assertEqual(result.operation_advice, "持有")
        self.assertTrue(52 <= result.sentiment_score <= 59)
        self.assertEqual(result.analysis_summary, "技术面向好")

    def test_parse_response_reconciles_conflicting_signal_fields(self) -> None:
        """Conflicting top-level fields should be normalized into a directional hold state."""
        analyzer = GeminiAnalyzer()
        response = json.dumps({
            "stock_name": "Apple",
            "sentiment_score": 78,
            "trend_prediction": "看空",
            "operation_advice": "买入",
            "decision_type": "buy",
            "dashboard": {
                "core_conclusion": {
                    "one_sentence": "建议直接买入",
                    "signal_type": "买入信号",
                }
            },
            "analysis_summary": "原始输出存在分歧",
        })

        result = analyzer._parse_response(response, "AAPL", "Apple")

        self.assertEqual(result.decision_type, "hold")
        self.assertEqual(result.operation_advice, "持有")
        self.assertEqual(result.trend_prediction, "震荡偏多")
        self.assertTrue(52 <= result.sentiment_score <= 59)
        self.assertIsInstance(result.dashboard, dict)
        self.assertEqual(result.dashboard["decision_type"], "hold")
        self.assertEqual(result.dashboard["core_conclusion"]["signal_type"], "持有待机")
        self.assertEqual(result.dashboard["core_conclusion"]["one_sentence"], "多空有分歧，当前先持有等待更优买点")

    def test_parse_response_can_surface_bearish_hold_bias(self) -> None:
        """Sell-vs-bullish conflicts should become a defensive hold with bearish bias when negatives dominate."""
        analyzer = GeminiAnalyzer()
        response = json.dumps({
            "stock_name": "Apple",
            "sentiment_score": 28,
            "trend_prediction": "看多",
            "operation_advice": "卖出",
            "decision_type": "sell",
            "analysis_summary": "原始输出存在分歧",
        })

        result = analyzer._parse_response(response, "AAPL", "Apple")

        self.assertEqual(result.decision_type, "hold")
        self.assertEqual(result.operation_advice, "观望")
        self.assertEqual(result.trend_prediction, "震荡偏空")
        self.assertTrue(40 <= result.sentiment_score <= 47)

    def test_parse_response_downgrades_weak_buy_without_support_to_hold(self) -> None:
        """Directional action without score/trend support should downgrade to a bullish hold."""
        analyzer = GeminiAnalyzer()
        response = json.dumps({
            "stock_name": "Apple",
            "sentiment_score": 50,
            "trend_prediction": "震荡",
            "operation_advice": "买入",
            "decision_type": "buy",
            "analysis_summary": "方向字段过于激进",
        })

        result = analyzer._parse_response(response, "AAPL", "Apple")

        self.assertEqual(result.decision_type, "hold")
        self.assertEqual(result.operation_advice, "持有")
        self.assertEqual(result.trend_prediction, "震荡偏多")
        self.assertTrue(52 <= result.sentiment_score <= 59)

    def test_parse_response_normalizes_strong_decision_label_but_keeps_buy_signal(self) -> None:
        """Model-facing strong_buy labels should be accepted and mapped to the stable enum."""
        analyzer = GeminiAnalyzer()
        response = json.dumps({
            "stock_name": "Apple",
            "sentiment_score": 88,
            "trend_prediction": "强烈看多",
            "operation_advice": "买入",
            "decision_type": "strong_buy",
            "analysis_summary": "趋势延续",
        })

        result = analyzer._parse_response(response, "AAPL", "Apple")

        self.assertEqual(result.decision_type, "buy")
        self.assertEqual(result.operation_advice, "买入")
        self.assertEqual(result.trend_prediction, "强烈看多")
        self.assertEqual(result.sentiment_score, 88)

    def test_parse_text_response_downgrades_conflicting_plain_text_to_hold(self) -> None:
        """Plain-text fallback should degrade contradictory signals to a defensive hold."""
        analyzer = GeminiAnalyzer()

        result = analyzer._parse_text_response(
            "短线看多但同时提示减仓，建议先等待确认，避免直接追价。",
            "AAPL",
            "Apple",
        )

        self.assertEqual(result.decision_type, "hold")
        self.assertIn(result.operation_advice, {"持有", "观望"})
        self.assertIn(result.trend_prediction, {"震荡偏多", "震荡", "震荡偏空"})
        self.assertTrue(40 <= result.sentiment_score <= 59)

    def test_parse_text_response_keeps_bearish_plain_text_conservative(self) -> None:
        """Bearish plain-text fallback should prefer the canonical sell advice."""
        analyzer = GeminiAnalyzer()

        result = analyzer._parse_text_response(
            "走势偏弱，跌破支撑，建议减仓或卖出，短线看空。",
            "TSLA",
            "Tesla",
        )

        self.assertEqual(result.decision_type, "sell")
        self.assertEqual(result.operation_advice, "减仓/卖出")
        self.assertEqual(result.trend_prediction, "看空")
        self.assertTrue(0 <= result.sentiment_score <= 39)

    def test_analysis_result_get_emoji_uses_normalized_signal_level(self) -> None:
        """Compound sell advice should render a sell emoji when decision_type is sell."""
        result = AnalysisResult(
            code="TSLA",
            name="Tesla",
            sentiment_score=78,
            trend_prediction="看空",
            operation_advice="减仓/卖出",
            decision_type="sell",
            analysis_summary="应优先防守",
        )

        self.assertEqual(result.get_emoji(), "🔴")
