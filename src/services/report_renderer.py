# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Jinja2 Report Renderer
===================================

Renders reports from Jinja2 templates. Falls back to caller's logic on template
missing or render error. Template path is relative to project root.
Any expensive data preparation should be injected by the caller via extra_context.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analyzer import AnalysisResult, get_result_signal_level
from src.config import get_config

logger = logging.getLogger(__name__)


def _get_signal_level(result: AnalysisResult) -> tuple:
    """Return (signal_text, emoji, color_tag) for a result."""
    return get_result_signal_level(result)


def _is_failed_result(result: AnalysisResult) -> bool:
    """Return True when the analysis failed and no valid conclusion was produced."""
    return (not getattr(result, "success", True)) and bool(getattr(result, "error_message", None))


def _truncate_display_text(value: Any, max_length: int) -> str:
    """Truncate long display text with a visible ellipsis."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3].rstrip() + "..."


def _escape_md(text: str) -> str:
    """Escape markdown special chars (*ST etc)."""
    if not text:
        return ""
    return text.replace("*", "\\*").replace("_", "\\_")


def _clean_sniper_value(val: Any) -> str:
    """Format sniper point value for display (strip label prefixes)."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).strip() if val else ""
    if not s or s == "N/A":
        return s or "N/A"
    prefixes = [
        "理想买入点：", "次优买入点：", "止损位：", "目标位：",
        "理想买入点:", "次优买入点:", "止损位:", "目标位:",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _resolve_templates_dir() -> Path:
    """Resolve template directory relative to project root."""
    config = get_config()
    base = Path(__file__).resolve().parent.parent.parent
    templates_dir = Path(config.report_templates_dir)
    if not templates_dir.is_absolute():
        return base / templates_dir
    return templates_dir


def render(
    platform: str,
    results: List[AnalysisResult],
    report_date: Optional[str] = None,
    summary_only: bool = False,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Render report using Jinja2 template.

    Args:
        platform: One of: markdown, wechat, brief
        results: List of AnalysisResult
        report_date: Report date string (default: today)
        summary_only: Whether to output summary only
        extra_context: Additional template context

    Returns:
        Rendered string, or None on error (caller should fallback).
    """
    from datetime import datetime

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        logger.warning("jinja2 not installed, report renderer disabled")
        return None

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    templates_dir = _resolve_templates_dir()
    template_name = f"report_{platform}.j2"
    template_path = templates_dir / template_name
    if not template_path.exists():
        logger.debug("Report template not found: %s", template_path)
        return None

    sorted_results = sorted(
        results,
        key=lambda result: (
            _is_failed_result(result),
            -(getattr(result, "sentiment_score", 0) or 0),
        ),
    )
    sorted_enriched = []
    for r in sorted_results:
        st, se, _ = _get_signal_level(r)
        rn = r.name if r.name and not r.name.startswith("股票") else f"股票{r.code}"
        sorted_enriched.append({
            "result": r,
            "signal_text": st,
            "signal_emoji": se,
            "stock_name": _escape_md(rn),
            "is_failed": _is_failed_result(r),
            "failure_text": _truncate_display_text(
                getattr(r, "error_message", None) or getattr(r, "analysis_summary", None) or "分析失败",
                140,
            ),
        })

    successful_results = [r for r in results if not _is_failed_result(r)]
    failed_count = len(results) - len(successful_results)
    buy_count = sum(1 for r in successful_results if getattr(r, "decision_type", "") == "buy")
    sell_count = sum(1 for r in successful_results if getattr(r, "decision_type", "") == "sell")
    hold_count = sum(1 for r in successful_results if getattr(r, "decision_type", "") in ("hold", ""))

    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def failed_checks(checklist: List[str]) -> List[str]:
        return [c for c in (checklist or []) if c.startswith("❌") or c.startswith("⚠️")]

    context: Dict[str, Any] = {
        "report_date": report_date,
        "report_timestamp": report_timestamp,
        "results": sorted_results,
        "enriched": sorted_enriched,  # Sorted by sentiment_score desc
        "summary_only": summary_only,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "failed_count": failed_count,
        "escape_md": _escape_md,
        "clean_sniper": _clean_sniper_value,
        "failed_checks": failed_checks,
        "history_by_code": {},
    }
    if extra_context:
        context.update(extra_context)

    try:
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(default=False),
        )
        template = env.get_template(template_name)
        return template.render(**context)
    except Exception as e:
        logger.warning("Report render failed for %s: %s", template_name, e)
        return None
