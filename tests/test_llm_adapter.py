# -*- coding: utf-8 -*-
"""Regression tests for LLMToolAdapter fallback behavior."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.llm_adapter import LLMResponse, LLMToolAdapter


class TestLLMToolAdapterFallback(unittest.TestCase):
    def test_rate_limit_error_falls_back_to_next_model(self) -> None:
        adapter = LLMToolAdapter.__new__(LLMToolAdapter)
        adapter._config = SimpleNamespace(
            litellm_model="gemini/primary",
            litellm_fallback_models=["openai/fallback"],
        )

        responses = [
            RuntimeError("Rate limit exceeded"),
            LLMResponse(content="ok", provider="openai", model="openai/fallback"),
        ]

        def _side_effect(*_args, **_kwargs):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        adapter._call_litellm_model = _side_effect

        with patch("src.agent.llm_adapter.time.sleep") as sleep_mock:
            response = adapter.call_completion([{"role": "user", "content": "hi"}], tools=None)

        self.assertEqual(response.content, "ok")
        sleep_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
