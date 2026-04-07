# -*- coding: utf-8 -*-
"""
IntelAgent — news & intelligence gathering specialist.

Responsible for:
- Searching latest stock news and announcements
- Running comprehensive intelligence search
- Detecting risk events (reduce holdings, earnings warnings, regulatory)
- Summarising sentiment and catalysts
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.runner import try_parse_json

logger = logging.getLogger(__name__)


class IntelAgent(BaseAgent):
    agent_name = "intel"
    max_steps = 4
    tool_names = [
        "search_stock_news",
        "search_comprehensive_intel",
        "get_stock_info",
        "get_capital_flow",
    ]

    def system_prompt(self, ctx: AgentContext) -> str:
        return """\
You are an **Intelligence & Sentiment Agent** specialising in A-shares, \
HK, and US equities.

Your task: gather the latest news, announcements, and risk signals for \
the given stock, then produce a structured JSON opinion.

## Workflow
1. Read pre-fetched `market_context`, `intel_report`, and direct official intel first.
2. Run `search_comprehensive_intel` when you need deeper or fresher context.
3. Use `search_stock_news` only as a supplement, not as the primary source.
4. Use `get_capital_flow` when A-share / HK capital-flow context may change the interpretation.
5. Classify positive catalysts and risk alerts using the correct market logic.
6. Assess overall sentiment

## Risk Detection Priorities
- A-shares: 减持、解禁、业绩预亏、监管处罚、政策利空、主力资金异动
- HK equities: HKEX results / profit warning / placement / buyback / dividend
- US equities: SEC filings, guidance, litigation, Form 4 / 13D / 13G, analyst and earnings reset
- For US names, Chinese policy should only be highlighted when `market_context.china_exposure`
  shows real revenue / supply-chain / policy transmission.

## Output Format
Return **only** a JSON object:
{
  "signal": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence summary of news/sentiment findings",
  "risk_alerts": ["list", "of", "detected", "risks"],
  "positive_catalysts": ["list", "of", "catalysts"],
  "sentiment_label": "very_positive|positive|neutral|negative|very_negative",
  "key_news": [
    {"title": "...", "impact": "positive|negative|neutral"}
  ]
}
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        parts = [f"Gather intelligence and assess sentiment for stock **{ctx.stock_code}**"]
        if ctx.stock_name:
            parts[0] += f" ({ctx.stock_name})"
        parts.append("Use direct official intel and market-specific logic, then output the JSON opinion.")

        if ctx.get_data("market_context"):
            parts.append(
                f"\n[Market context]\n{json.dumps(ctx.get_data('market_context'), ensure_ascii=False, default=str)}"
            )

        if ctx.get_data("intel_report"):
            parts.append(f"\n[Prefetched intel report]\n{ctx.get_data('intel_report')}")

        if ctx.get_data("intel_dimensions"):
            parts.append(
                f"\n[Prefetched intel dimensions]\n"
                f"{json.dumps(ctx.get_data('intel_dimensions'), ensure_ascii=False, default=str)}"
            )
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        parsed = try_parse_json(raw_text)
        if parsed is None:
            logger.warning("[IntelAgent] failed to parse opinion JSON")
            return None

        # Cache parsed intel so downstream agents (especially RiskAgent) can
        # reuse it instead of re-searching the same evidence.
        ctx.set_data("intel_opinion", parsed)

        # Propagate risk alerts to context
        for alert in parsed.get("risk_alerts", []):
            if isinstance(alert, str) and alert:
                ctx.add_risk_flag(category="intel", description=alert)

        return AgentOpinion(
            agent_name=self.agent_name,
            signal=parsed.get("signal", "hold"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", ""),
            raw_data=parsed,
        )


