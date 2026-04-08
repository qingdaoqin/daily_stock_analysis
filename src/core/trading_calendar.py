# -*- coding: utf-8 -*-
"""
===================================
交易日历模块 (Issue #373)
===================================

职责：
1. 按市场（A股/港股/美股）判断当日是否为交易日
2. 按市场时区取“今日”日期，避免服务器 UTC 导致日期错误
3. 支持 per-stock 过滤：只分析当日开市市场的股票

依赖：exchange-calendars（可选，不可用时 fail-open）
"""

import logging
from datetime import date, datetime
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Exchange-calendars availability
_XCALS_AVAILABLE = False
try:
    import exchange_calendars as xcals
    _XCALS_AVAILABLE = True
except ImportError:
    logger.warning(
        "exchange-calendars not installed; trading day check disabled. "
        "Run: pip install exchange-calendars"
    )

# Market -> exchange code (exchange-calendars)
MARKET_EXCHANGE = {"cn": "XSHG", "hk": "XHKG", "us": "XNYS"}

# Market -> IANA timezone for "today"
MARKET_TIMEZONE = {
    "cn": "Asia/Shanghai",
    "hk": "Asia/Hong_Kong",
    "us": "America/New_York",
}


def _normalize_market_code(code: str) -> str:
    """Normalize exchange-prefixed/suffixed stock codes for market detection."""
    normalized = (code or "").strip().upper()
    if not normalized:
        return ""

    for suffix in (".SH", ".SZ", ".SS", ".BJ"):
        if normalized.endswith(suffix):
            digits = normalized[: -len(suffix)]
            if digits.isdigit() and len(digits) in (5, 6):
                return digits

    for prefix in ("SH", "SZ", "SS", "BJ"):
        if normalized.startswith(prefix):
            digits = normalized[len(prefix):]
            if digits.isdigit() and len(digits) in (5, 6):
                return digits

    return normalized


def _is_hk_stock_code(code: str) -> bool:
    """Lightweight HK code detector without importing heavy fetcher modules."""
    normalized = (code or "").strip().upper()
    if normalized.endswith(".HK"):
        digits = normalized[:-3]
        return digits.isdigit() and 1 <= len(digits) <= 5
    if normalized.startswith("HK"):
        digits = normalized[2:]
        return digits.isdigit() and 1 <= len(digits) <= 5
    return normalized.isdigit() and 1 <= len(normalized) <= 5


def get_market_for_stock(code: str) -> Optional[str]:
    """
    Infer market region for a stock code.

    Returns:
        'cn' | 'hk' | 'us' | None (None = unrecognized, fail-open: treat as open)
    """
    if not code or not isinstance(code, str):
        return None
    raw_code = (code or "").strip().upper()
    code = _normalize_market_code(raw_code)

    from data_provider.us_index_mapping import is_us_stock_code, is_us_index_code

    if is_us_stock_code(raw_code) or is_us_index_code(raw_code) or is_us_stock_code(code) or is_us_index_code(code):
        return "us"
    if _is_hk_stock_code(raw_code) or _is_hk_stock_code(code):
        return "hk"
    # A-share: 6-digit numeric
    if code.isdigit() and len(code) == 6:
        return "cn"
    return None


def is_market_open(market: str, check_date: date) -> bool:
    """
    Check if the given market is open on the given date.

    Fail-open: returns True if exchange-calendars unavailable or date out of range.

    Args:
        market: 'cn' | 'hk' | 'us'
        check_date: Date to check

    Returns:
        True if trading day (or fail-open), False otherwise
    """
    if not _XCALS_AVAILABLE:
        return True
    ex = MARKET_EXCHANGE.get(market)
    if not ex:
        return True
    try:
        cal = xcals.get_calendar(ex)
        session = datetime(check_date.year, check_date.month, check_date.day)
        return cal.is_session(session)
    except Exception as e:
        logger.warning("trading_calendar.is_market_open fail-open: %s", e)
        return True


def get_open_markets_today() -> Set[str]:
    """
    Get markets that are open today (by each market's local timezone).

    Returns:
        Set of market keys ('cn', 'hk', 'us') that are trading today
    """
    if not _XCALS_AVAILABLE:
        return {"cn", "hk", "us"}
    result: Set[str] = set()
    from zoneinfo import ZoneInfo
    for mkt, tz_name in MARKET_TIMEZONE.items():
        try:
            tz = ZoneInfo(tz_name)
            today = datetime.now(tz).date()
            if is_market_open(mkt, today):
                result.add(mkt)
        except Exception as e:
            logger.warning("get_open_markets_today fail-open for %s: %s", mkt, e)
            result.add(mkt)
    return result


def get_market_today(market: Optional[str]) -> date:
    """
    Return today's date in the market's local timezone.

    Args:
        market: 'cn' | 'hk' | 'us' | None

    Returns:
        Local calendar date for the given market; falls back to server-local date
        when market is unknown or timezone resolution fails.
    """
    if market not in MARKET_TIMEZONE:
        return date.today()

    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(MARKET_TIMEZONE[market])
        return datetime.now(tz).date()
    except Exception as e:
        logger.warning("get_market_today fail-open for %s: %s", market, e)
        return date.today()


def compute_effective_region(
    config_region: str, open_markets: Set[str]
) -> Optional[str]:
    """
    Compute effective market review region given config and open markets.

    Args:
        config_region: From MARKET_REVIEW_REGION ('cn' | 'hk' | 'us' | 'both' | 'all')
        open_markets: Markets open today

    Returns:
        None: caller uses config default (check disabled)
        '': all relevant markets closed, skip market review
        'cn' | 'hk' | 'us' | 'both' | 'all': effective subset for today
    """
    if config_region not in ("cn", "hk", "us", "both", "all"):
        config_region = "cn"
    if config_region == "cn":
        return "cn" if "cn" in open_markets else ""
    if config_region == "hk":
        return "hk" if "hk" in open_markets else ""
    if config_region == "us":
        return "us" if "us" in open_markets else ""
    if config_region == "both":
        parts = []
        if "cn" in open_markets:
            parts.append("cn")
        if "us" in open_markets:
            parts.append("us")
        if not parts:
            return ""
        return "both" if len(parts) == 2 else parts[0]

    # all
    parts = []
    if "cn" in open_markets:
        parts.append("cn")
    if "hk" in open_markets:
        parts.append("hk")
    if "us" in open_markets:
        parts.append("us")
    if not parts:
        return ""
    if len(parts) == 3:
        return "all"
    if len(parts) == 2:
        combo = "+".join(parts)
        return combo
    return parts[0]
