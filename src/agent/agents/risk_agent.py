# -*- coding: utf-8 -*-
"""
RiskAgent — dedicated risk screening specialist.

Responsible for:
- Scanning for insider sell-downs, earnings warnings, regulatory actions
- Checking valuation anomalies (PE/PB extremes)
- Evaluating lock-up expiration risks
- Producing risk flags that can override or downgrade signals from other agents

Risk flags use a two-level severity system:
- **soft**: downgrades the signal and adds a visible warning
- **hard**: vetoes buy signals entirely when risk override is enabled
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.runner import try_parse_json

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    agent_name = "risk"
    max_steps = 4
    tool_names = [
        "search_comprehensive_intel",
        "search_stock_news",
        "get_realtime_quote",
        "get_stock_info",
    ]

    def system_prompt(self, ctx: AgentContext) -> str:
        return """\
You are a **Risk Screening Agent** focused exclusively on identifying \
risks and red flags for the given stock across A-shares, HK equities, \
and US equities.

Your task: search for and evaluate ALL material risk factors, then \
output a structured JSON risk assessment.

## Mandatory Risk Checks
1. **Official Source First** — use pre-fetched `market_context`, `intel_report`, \
   and `search_comprehensive_intel` before generic news search. Prefer CNINFO / \
   HKEX / SEC evidence over media summaries.
2. **A-shares** — insider sell-downs (减持), earnings warnings, regulatory \
   penalties, lock-up expirations (解禁), policy crackdowns.
3. **HK equities** — profit warning, placement/rights issue, buyback/dividend \
   changes, litigation, HKEX corporate actions, mainland-policy transmission \
   only when the issuer actually has mainland exposure.
4. **US equities** — SEC filings, litigation, guidance cuts, Form 4 / 13D / \
   13G activity, valuation extremes, short-interest / options stress. \
   Chinese policy matters only if `market_context.china_exposure` shows \
   genuine China revenue / supply-chain / export-control exposure.
5. **Technical Warning Signs** — death crosses, breaks of key supports, \
   sharp gap-downs.

## Severity Levels
- "high": existential or material risk (lawsuits, fraud, massive insider selling)
- "medium": significant concern (earnings miss, lock-up, sector headwind)
- "low": minor or informational (analyst downgrade, minor insider sale)

## Output Format
Return **only** a JSON object:
{
  "risk_level": "high|medium|low|none",
  "risk_score": 0-100,
  "flags": [
    {
      "category": "insider|earnings|regulatory|industry|lockup|valuation|technical",
      "severity": "high|medium|low",
      "description": "Clear description of the risk",
      "source": "Where this information came from"
    }
  ],
  "veto_buy": true|false,
  "reasoning": "2-3 sentence overall risk assessment",
  "signal_adjustment": "none|downgrade_one|downgrade_two|veto"
}

Important: be thorough but factual. Only flag risks backed by evidence \
from your search results. Do NOT invent risks.
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        parts = [f"Screen stock **{ctx.stock_code}**"]
        if ctx.stock_name:
            parts[0] += f" ({ctx.stock_name})"
        parts.append("for ALL risk factors listed in your instructions.")
        parts.append("Use pre-fetched market context and comprehensive intel first; search latest news only to fill gaps.")

        if ctx.get_data("market_context"):
            parts.append(
                f"\n[Market context]\n{json.dumps(ctx.get_data('market_context'), ensure_ascii=False, default=str)}"
            )

        if ctx.get_data("intel_report"):
            parts.append(f"\n[Intel report]\n{ctx.get_data('intel_report')}")

        if ctx.get_data("intel_dimensions"):
            parts.append(
                f"\n[Intel dimensions]\n{json.dumps(ctx.get_data('intel_dimensions'), ensure_ascii=False, default=str)}"
            )

        # Feed any existing intel data so the risk agent doesn't redo searches
        if ctx.get_data("intel_opinion"):
            parts.append(f"\n[Existing intel data]\n{json.dumps(ctx.get_data('intel_opinion'), ensure_ascii=False, default=str)}")

        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        parsed = try_parse_json(raw_text)
        if parsed is None:
            logger.warning("[RiskAgent] failed to parse risk JSON")
            return None

        # Propagate structured risk flags to context
        for flag in parsed.get("flags", []):
            if isinstance(flag, dict):
                ctx.add_risk_flag(
                    category=flag.get("category", "unknown"),
                    description=flag.get("description", ""),
                    severity=flag.get("severity", "medium"),
                )

        return AgentOpinion(
            agent_name=self.agent_name,
            signal=_risk_to_signal(parsed.get("risk_level", "none")),
            confidence=float(parsed.get("risk_score", 50)) / 100.0,
            reasoning=parsed.get("reasoning", ""),
            raw_data=parsed,
        )


def _risk_to_signal(risk_level: str) -> str:
    """Map risk level to a trading signal (inverted)."""
    mapping = {
        "none": "buy",
        "low": "hold",
        "medium": "sell",
        "high": "strong_sell",
    }
    return mapping.get(risk_level, "hold")

