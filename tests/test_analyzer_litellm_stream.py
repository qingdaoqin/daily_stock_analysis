# -*- coding: utf-8 -*-
"""
Tests for LiteLLM streaming support in GeminiAnalyzer.

Covers:
- _LiteLLMStreamError construction
- _extract_stream_text  (dict / object / content_blocks)
- _normalize_usage      (dict / object / empty)
- _consume_litellm_stream (normal / empty / interrupted)
- _call_litellm stream -> fallback path
"""
import types
import pytest
from unittest.mock import MagicMock, patch

# Patch litellm import before importing analyzer
import sys

litellm_stub = types.ModuleType("litellm")
litellm_stub.Router = MagicMock()
litellm_stub.completion = MagicMock()
sys.modules.setdefault("litellm", litellm_stub)

from src.analyzer import GeminiAnalyzer, _LiteLLMStreamError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer() -> GeminiAnalyzer:
    """Return a GeminiAnalyzer with LiteLLM stubbed out."""
    with (
        patch("src.analyzer.litellm", litellm_stub),
        patch("src.analyzer.Router", MagicMock()),
        patch("src.analyzer.get_config") as mock_cfg,
    ):
        cfg = MagicMock()
        cfg.litellm_model = "gemini/gemini-2.5-flash"
        cfg.litellm_fallback_models = []
        cfg.llm_model_list = []
        mock_cfg.return_value = cfg
        a = GeminiAnalyzer.__new__(GeminiAnalyzer)
        a._router = None
        a._litellm_available = True
        return a


# ---------------------------------------------------------------------------
# _LiteLLMStreamError
# ---------------------------------------------------------------------------

class TestLiteLLMStreamError:
    def test_default_partial_received_false(self):
        err = _LiteLLMStreamError("oops")
        assert err.partial_received is False
        assert "oops" in str(err)

    def test_partial_received_true(self):
        err = _LiteLLMStreamError("mid-stream", partial_received=True)
        assert err.partial_received is True


# ---------------------------------------------------------------------------
# _extract_stream_text
# ---------------------------------------------------------------------------

class TestExtractStreamText:
    def setup_method(self):
        self.a = _make_analyzer()

    def _chunk_dict(self, content):
        return {"choices": [{"delta": {"content": content}}]}

    def test_dict_chunk_text(self):
        chunk = self._chunk_dict("hello")
        assert self.a._extract_stream_text(chunk) == "hello"

    def test_dict_chunk_none_content(self):
        chunk = {"choices": [{"delta": {"content": None}}]}
        assert self.a._extract_stream_text(chunk) == ""

    def test_empty_choices(self):
        assert self.a._extract_stream_text({"choices": []}) == ""

    def test_object_chunk(self):
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = "world"
        chunk.choices = [MagicMock(delta=delta, message=None)]
        assert self.a._extract_stream_text(chunk) == "world"

    def test_content_blocks_list(self):
        chunk = {
            "choices": [
                {"delta": {"content": [{"text": "foo"}, {"text": "bar"}]}}
            ]
        }
        assert self.a._extract_stream_text(chunk) == "foobar"

    def test_content_blocks_mixed(self):
        chunk = {
            "choices": [
                {"delta": {"content": ["raw", {"text": "ok"}, 123]}}
            ]
        }
        assert self.a._extract_stream_text(chunk) == "rawok"

    def test_fallback_message_content(self):
        """When delta is missing, fall back to message.content."""
        chunk = {"choices": [{"delta": None, "message": {"content": "msg"}}]}
        assert self.a._extract_stream_text(chunk) == "msg"

    def test_no_choices_key(self):
        assert self.a._extract_stream_text({}) == ""


# ---------------------------------------------------------------------------
# _normalize_usage
# ---------------------------------------------------------------------------

class TestNormalizeUsage:
    def setup_method(self):
        self.a = _make_analyzer()

    def test_dict_usage(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert self.a._normalize_usage(usage) == usage

    def test_object_usage(self):
        obj = MagicMock()
        obj.prompt_tokens = 5
        obj.completion_tokens = 15
        obj.total_tokens = 20
        result = self.a._normalize_usage(obj)
        assert result == {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}

    def test_none_returns_empty(self):
        assert self.a._normalize_usage(None) == {}

    def test_empty_dict_returns_empty(self):
        assert self.a._normalize_usage({}) == {}

    def test_partial_fields(self):
        obj = MagicMock()
        obj.prompt_tokens = None
        obj.completion_tokens = 8
        obj.total_tokens = 8
        result = self.a._normalize_usage(obj)
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 8


# ---------------------------------------------------------------------------
# _consume_litellm_stream
# ---------------------------------------------------------------------------

def _make_chunks(*texts):
    """Create a list of fake streaming chunks."""
    return [{"choices": [{"delta": {"content": t}}]} for t in texts]


class TestConsumeLiteLLMStream:
    def setup_method(self):
        self.a = _make_analyzer()

    def test_normal_consumption(self):
        chunks = _make_chunks("hello", " ", "world")
        text, usage = self.a._consume_litellm_stream(
            iter(chunks), model="test-model"
        )
        assert text == "hello world"
        assert usage == {}

    def test_empty_stream_raises(self):
        with pytest.raises(_LiteLLMStreamError) as exc_info:
            self.a._consume_litellm_stream(iter([]), model="m")
        assert not exc_info.value.partial_received

    def test_interrupted_stream_raises_with_partial(self):
        def bad_iter():
            yield {"choices": [{"delta": {"content": "start"}}]}
            raise RuntimeError("connection dropped")

        with pytest.raises(_LiteLLMStreamError) as exc_info:
            self.a._consume_litellm_stream(bad_iter(), model="m")
        assert exc_info.value.partial_received

    def test_progress_callback_called(self):
        calls = []
        self.a._consume_litellm_stream(
            iter(_make_chunks("a" * 200)),
            model="m",
            progress_callback=lambda n: calls.append(n),
        )
        assert len(calls) >= 1
        assert calls[-1] == 200

    def test_skip_empty_delta(self):
        chunks = _make_chunks("real", "", "  ")
        text, _ = self.a._consume_litellm_stream(iter(chunks), model="m")
        assert "real" in text


# ---------------------------------------------------------------------------
# _call_litellm stream fallback
# ---------------------------------------------------------------------------

class TestCallLiteLLMStream:
    def setup_method(self):
        self.a = _make_analyzer()

    @patch("src.analyzer.get_config")
    def test_stream_success_returns_immediately(self, mock_cfg):
        cfg = MagicMock()
        cfg.litellm_model = "gemini/test"
        cfg.litellm_fallback_models = []
        cfg.llm_model_list = []
        mock_cfg.return_value = cfg

        chunks = _make_chunks("streamed", " response")
        with (
            patch.object(self.a, "_dispatch_litellm_completion", return_value=iter(chunks)),
            patch.object(self.a, "_get_model_retry_policy", return_value=(0, 0.0)),
        ):
            text, model, usage = self.a._call_litellm(
                "prompt",
                {"max_tokens": 512},
                stream=True,
            )
        assert text == "streamed response"
        assert model == "gemini/test"

    @patch("src.analyzer.get_config")
    def test_stream_failure_falls_back_to_nonstream(self, mock_cfg):
        cfg = MagicMock()
        cfg.litellm_model = "gemini/test"
        cfg.litellm_fallback_models = []
        cfg.llm_model_list = []
        mock_cfg.return_value = cfg

        # stream response raises immediately (no chunks)
        def _dispatch(model, kw, **kwargs):
            if kw.get("stream"):
                return iter([])  # empty → will raise _LiteLLMStreamError
            # non-stream fallback
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "fallback text"
            resp.usage = None
            return resp

        with (
            patch.object(self.a, "_dispatch_litellm_completion", side_effect=_dispatch),
            patch.object(self.a, "_get_model_retry_policy", return_value=(0, 0.0)),
        ):
            text, model, usage = self.a._call_litellm(
                "prompt",
                {"max_tokens": 512},
                stream=True,
            )
        assert text == "fallback text"
