# -*- coding: utf-8 -*-
"""
Default skill definitions and helper functions.

This module centralises constants and helpers used by the skill router,
aggregator, and prompt builder so that callers only need to import from
one place.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# Naming convention
# ---------------------------------------------------------------------------
_SKILL_PREFIX = "skill:"

SKILL_CONSENSUS_AGENT_NAME = "skill_consensus"
"""Agent name used by the SkillAggregator for the consensus opinion."""


def is_skill_agent_name(agent_name: str) -> bool:
    """Return True when *agent_name* looks like a skill-agent identifier."""
    return agent_name.startswith(_SKILL_PREFIX) if agent_name else False


def extract_skill_id(agent_name: str) -> Optional[str]:
    """Strip the ``skill:`` prefix and return the bare skill id, or *None*."""
    if agent_name and agent_name.startswith(_SKILL_PREFIX):
        return agent_name[len(_SKILL_PREFIX):]
    return None


# ---------------------------------------------------------------------------
# Default skill routing
# ---------------------------------------------------------------------------
_DEFAULT_SKILL_IDS = ("trend_follow", "mean_reversion", "breakout")

_REGIME_SKILL_MAP = {
    "trending_up": ("trend_follow", "breakout"),
    "trending_down": ("trend_follow", "mean_reversion"),
    "sideways": ("mean_reversion",),
    "volatile": ("mean_reversion", "breakout"),
    "sector_hot": ("trend_follow", "breakout"),
}


def get_default_router_skill_ids(
    skill_catalog=None,
    *,
    max_count: int = 3,
    available_skill_ids: Optional[Set[str]] = None,
) -> List[str]:
    """Return a list of default skill ids, optionally filtered by availability."""
    candidates = list(_DEFAULT_SKILL_IDS)
    if available_skill_ids is not None:
        candidates = [sid for sid in candidates if sid in available_skill_ids]
    return candidates[:max_count]


def get_regime_skill_ids(
    regime: str,
    skill_catalog=None,
    *,
    max_count: int = 3,
    available_skill_ids: Optional[Set[str]] = None,
) -> List[str]:
    """Return skill ids recommended for a detected market *regime*."""
    candidates = list(_REGIME_SKILL_MAP.get(regime, ()))
    if available_skill_ids is not None:
        candidates = [sid for sid in candidates if sid in available_skill_ids]
    return candidates[:max_count]


# ---------------------------------------------------------------------------
# Prompt policy — embedded into LLM system prompt by analyzer.py
# ---------------------------------------------------------------------------
CORE_TRADING_SKILL_POLICY_ZH = """## 核心交易纪律

1. **趋势为王**：顺势而为，不与大趋势对抗。
2. **量价配合**：上涨放量、下跌缩量是健康信号，反之需要警惕。
3. **严格止损**：预设止损位，跌破关键支撑果断执行。
4. **仓位管理**：单只个股仓位不超过总资金的30%，分批建仓减仓。
5. **情绪纪律**：不追高杀跌、不频繁交易、不因单次亏损改变策略。
6. **综合研判**：技术面、基本面、消息面、资金面四维共振。
7. **风险收益比**：只参与预期收益 / 潜在风险 ≥ 2:1 的交易机会。"""
