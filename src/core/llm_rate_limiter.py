# -*- coding: utf-8 -*-
"""LLM 请求节流器。"""

from __future__ import annotations

import random
import threading
import time
from typing import Dict


def get_rate_limit_bucket(model: str, default: str = "openai") -> str:
    """Return a stable provider bucket name for rate limiting."""
    normalized_default = (default or "openai").strip().lower() or "openai"
    normalized_model = (model or "").strip().lower()
    if not normalized_model:
        return normalized_default
    if "/" in normalized_model:
        return normalized_model.split("/", 1)[0] or normalized_default
    return normalized_default


class LLMRateLimiter:
    """线程安全的按桶最小间隔限流器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_allowed_at: Dict[str, float] = {}

    def acquire(self, bucket: str, min_interval_seconds: float, jitter: float = 0.0) -> None:
        interval = max(0.0, float(min_interval_seconds or 0.0))
        jitter_value = max(0.0, float(jitter or 0.0))
        if interval <= 0 and jitter_value <= 0:
            return

        bucket_key = (bucket or "default").strip().lower() or "default"

        while True:
            with self._lock:
                now = time.monotonic()
                next_allowed_at = self._next_allowed_at.get(bucket_key, 0.0)
                wait_seconds = next_allowed_at - now
                if wait_seconds <= 0:
                    scheduled_interval = interval
                    if jitter_value > 0:
                        scheduled_interval += random.uniform(0.0, jitter_value)
                    self._next_allowed_at[bucket_key] = now + scheduled_interval
                    return
            time.sleep(wait_seconds)


_global_rate_limiter = LLMRateLimiter()


def get_llm_rate_limiter() -> LLMRateLimiter:
    return _global_rate_limiter

