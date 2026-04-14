# -*- coding: utf-8 -*-
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from bot.commands.market import MarketCommand
from bot.models import BotMessage, ChatType


def _make_message() -> BotMessage:
    return BotMessage(
        platform="feishu",
        message_id="m1",
        user_id="u1",
        user_name="tester",
        chat_id="c1",
        chat_type=ChatType.GROUP,
        content="/market",
    )


def test_run_market_review_creates_search_service_when_only_anspire_configured():
    command = MarketCommand()
    notifier = MagicMock()
    market_analyzer = MagicMock()
    market_analyzer.run_daily_review.return_value = "done"
    search_service_cls = MagicMock(name="SearchService")
    config = SimpleNamespace(
        anspire_api_keys=["anspire-key"],
        bocha_api_keys=[],
        tavily_api_keys=[],
        brave_api_keys=[],
        serpapi_keys=[],
        minimax_api_keys=[],
        xai_api_keys=[],
        xai_search_model="grok-4-1-fast-reasoning",
        searxng_base_urls=[],
        searxng_public_instances_enabled=False,
        news_max_age_days=3,
        gemini_api_key=None,
        openai_api_key=None,
        market_review_region="cn",
        has_search_capability_enabled=lambda: True,
    )

    fake_config = types.ModuleType("src.config")
    fake_config.get_config = MagicMock(return_value=config)
    fake_notification = types.ModuleType("src.notification")
    fake_notification.NotificationService = MagicMock(return_value=notifier)
    fake_market_analyzer = types.ModuleType("src.market_analyzer")
    fake_market_analyzer.MarketAnalyzer = MagicMock(return_value=market_analyzer)
    fake_search_service = types.ModuleType("src.search_service")
    fake_search_service.SearchService = search_service_cls
    fake_analyzer = types.ModuleType("src.analyzer")
    fake_analyzer.GeminiAnalyzer = MagicMock()

    original_modules = {name: sys.modules.get(name) for name in (
        "src.config",
        "src.notification",
        "src.market_analyzer",
        "src.search_service",
        "src.analyzer",
    )}
    sys.modules.update(
        {
            "src.config": fake_config,
            "src.notification": fake_notification,
            "src.market_analyzer": fake_market_analyzer,
            "src.search_service": fake_search_service,
            "src.analyzer": fake_analyzer,
        }
    )
    try:
        command._run_market_review(_make_message())
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    search_service_cls.assert_called_once()
    fake_market_analyzer.MarketAnalyzer.assert_called_once_with(
        search_service=search_service_cls.return_value,
        analyzer=None,
        region="cn",
    )
    notifier.send.assert_called_once()