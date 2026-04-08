# -*- coding: utf-8 -*-
"""
===================================
LongbridgeFetcher - 长桥兜底数据源 (Priority 5)
===================================

数据来源：长桥 OpenAPI (https://open.longbridge.com)
特点：覆盖美股 + 港股，可计算量比/换手率/PE 等 yfinance 缺失字段
定位：美股/港股最后兜底数据源

关键策略：
1. 组合 quote + static_info 接口计算 turnover_rate / pe_ratio / total_mv
2. 通过 history_candlesticks 计算 volume_ratio（近5日均量比）
3. 懒加载 QuoteContext，首次调用时才建立连接
4. static_info 进程内短缓存，减少重复请求（默认 24h，可调；见 LONGBRIDGE_STATIC_INFO_TTL_SECONDS）

凭证：`LONGBRIDGE_APP_KEY` / `LONGBRIDGE_APP_SECRET` / `LONGBRIDGE_ACCESS_TOKEN`。
可选：`LONGBRIDGE_STATIC_INFO_TTL_SECONDS`；SDK `language` 取自 `REPORT_LANGUAGE`，`log_path` 为 `{LOG_DIR}/longbridge_sdk.log`；
`LONGBRIDGE_HTTP_URL` / `LONGBRIDGE_QUOTE_WS_URL` / `LONGBRIDGE_TRADE_WS_URL` / `LONGBRIDGE_REGION` （见官方文档默认值）。
"""

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseFetcher, STANDARD_COLUMNS
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote, safe_float
from .us_index_mapping import is_us_index_code, is_us_stock_code

logger = logging.getLogger(__name__)

_DEFAULT_STATIC_INFO_TTL = 86400


def _static_info_ttl_seconds() -> int:
    """TTL for static_info cache; 0 disables caching."""
    raw = os.getenv("LONGBRIDGE_STATIC_INFO_TTL_SECONDS", "").strip()
    if raw == "":
        return _DEFAULT_STATIC_INFO_TTL
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_STATIC_INFO_TTL


_REGION_URL_MAP: Dict[str, Dict[str, str]] = {
    "cn": {
        "http_url": "https://openapi.longbridge.cn",
        "quote_ws_url": "wss://openapi-quote.longbridge.cn/v2",
        "trade_ws_url": "wss://openapi-trade.longbridge.cn/v2",
    },
    "hk": {
        "http_url": "https://openapi.longbridge.com",
        "quote_ws_url": "wss://openapi-quote.longbridge.com/v2",
        "trade_ws_url": "wss://openapi-trade.longbridge.com/v2",
    },
}


def _sanitize_longbridge_env() -> None:
    """Remove empty-string Longbridge env vars and apply region defaults."""
    for key in (
        "LONGBRIDGE_HTTP_URL",
        "LONGBRIDGE_QUOTE_WS_URL",
        "LONGBRIDGE_TRADE_WS_URL",
        "LONGBRIDGE_ENABLE_OVERNIGHT",
        "LONGBRIDGE_PUSH_CANDLESTICK_MODE",
        "LONGBRIDGE_PRINT_QUOTE_PACKAGES",
        "LONGBRIDGE_REGION",
        "LONGBRIDGE_STATIC_INFO_TTL_SECONDS",
        "LONGBRIDGE_LOG_PATH",
    ):
        val = os.environ.get(key)
        if val is not None and val.strip() == "":
            del os.environ[key]
            logger.debug("[Longbridge] 删除空环境变量 %s", key)

    if "LONGBRIDGE_PRINT_QUOTE_PACKAGES" not in os.environ:
        os.environ["LONGBRIDGE_PRINT_QUOTE_PACKAGES"] = "false"

    if not os.environ.get("LONGBRIDGE_LOG_PATH"):
        try:
            log_dir = (os.getenv("LOG_DIR") or "./logs").strip() or "./logs"
            path = Path(log_dir).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            os.environ["LONGBRIDGE_LOG_PATH"] = str(path / "longbridge_sdk.log")
        except Exception:
            pass

    region = (os.getenv("LONGBRIDGE_REGION") or "").strip().lower()
    if region:
        if not os.environ.get("LONGPORT_REGION"):
            os.environ["LONGPORT_REGION"] = region
        urls = _REGION_URL_MAP.get(region, {})
        for env_name, default_url in (
            ("LONGBRIDGE_HTTP_URL", urls.get("http_url")),
            ("LONGBRIDGE_QUOTE_WS_URL", urls.get("quote_ws_url")),
            ("LONGBRIDGE_TRADE_WS_URL", urls.get("trade_ws_url")),
        ):
            if default_url and not os.environ.get(env_name):
                os.environ[env_name] = default_url


def _longbridge_config_kwargs() -> Dict[str, Any]:
    """Optional kwargs for Config.from_apikey."""
    try:
        import inspect
        from longbridge.openapi import Config, Language, PushCandlestickMode
    except Exception:
        return {}

    try:
        params = inspect.signature(Config.from_apikey).parameters
    except Exception:
        return {}

    kw: Dict[str, Any] = {}

    if "enable_print_quote_packages" in params:
        raw = os.getenv("LONGBRIDGE_PRINT_QUOTE_PACKAGES")
        if raw is None or not str(raw).strip():
            kw["enable_print_quote_packages"] = False
        else:
            raw_norm = str(raw).strip().lower()
            kw["enable_print_quote_packages"] = raw_norm not in ("0", "false", "no")

    for pname, envname in (
        ("http_url", "LONGBRIDGE_HTTP_URL"),
        ("quote_ws_url", "LONGBRIDGE_QUOTE_WS_URL"),
        ("trade_ws_url", "LONGBRIDGE_TRADE_WS_URL"),
    ):
        if pname in params:
            value = os.getenv(envname, "").strip()
            if value:
                kw[pname] = value

    if "language" in params:
        try:
            from src.report_language import normalize_report_language

            report_language = normalize_report_language(os.getenv("REPORT_LANGUAGE"), default="zh")
            if report_language == "zh":
                kw["language"] = Language.ZH_CN
            elif report_language == "en":
                kw["language"] = Language.EN
        except Exception as e:
            logger.debug("Longbridge language from REPORT_LANGUAGE skipped: %s", e)

    if "enable_overnight" in params:
        raw = os.getenv("LONGBRIDGE_ENABLE_OVERNIGHT", "").strip().lower()
        if raw:
            kw["enable_overnight"] = raw in ("1", "true", "yes")

    if "push_candlestick_mode" in params:
        raw = os.getenv("LONGBRIDGE_PUSH_CANDLESTICK_MODE", "").strip().lower()
        if raw == "realtime":
            kw["push_candlestick_mode"] = PushCandlestickMode.Realtime
        elif raw == "confirmed":
            kw["push_candlestick_mode"] = PushCandlestickMode.Confirmed

    if "log_path" in params:
        try:
            log_dir = (os.getenv("LOG_DIR") or "./logs").strip() or "./logs"
            path = Path(log_dir).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            kw["log_path"] = str(path / "longbridge_sdk.log")
        except Exception as e:
            logger.debug("Longbridge log_path from LOG_DIR skipped: %s", e)

    return kw


def _is_us_code(stock_code: str) -> bool:
    normalized = stock_code.strip().upper()
    return is_us_stock_code(normalized) or is_us_index_code(normalized)


def _is_hk_code(stock_code: str) -> bool:
    normalized = (stock_code or "").strip().upper()
    if normalized.startswith("HK"):
        digits = normalized[2:]
        return digits.isdigit() and 1 <= len(digits) <= 5
    if normalized.endswith(".HK"):
        return True
    if normalized.isdigit() and 1 <= len(normalized) <= 5:
        return True
    return False


def _to_longbridge_symbol(stock_code: str) -> Optional[str]:
    """Convert internal stock code to Longbridge symbol format."""
    code = stock_code.strip()
    upper = code.upper()

    if upper.endswith(".US") or upper.endswith(".HK"):
        return upper

    if _is_us_code(code):
        return f"{upper}.US"

    if _is_hk_code(code):
        if upper.startswith("HK"):
            digits = upper[2:]
        else:
            digits = upper
        digits = digits.lstrip("0") or "0"
        return f"{digits.zfill(4)}.HK"

    return None


class LongbridgeFetcher(BaseFetcher):
    """长桥 OpenAPI 美股/港股兜底数据源。"""

    name = "LongbridgeFetcher"
    priority = int(os.getenv("LONGBRIDGE_PRIORITY", "5"))

    _CONNECTION_ERRORS = ("client is closed", "context closed", "connection closed")

    def __init__(self):
        self._ctx = None
        self._config = None
        self._ctx_lock = threading.Lock()
        self._available = None
        self._static_cache: Dict[str, Any] = {}
        self._static_cache_lock = threading.Lock()

    def _is_connection_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in self._CONNECTION_ERRORS)

    def _invalidate_ctx(self) -> None:
        with self._ctx_lock:
            self._ctx = None
            self._config = None

    def _is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from src.config import get_config

            config = get_config()
            has_creds = bool(
                config.longbridge_app_key
                and config.longbridge_app_secret
                and config.longbridge_access_token
            )
        except Exception:
            has_creds = bool(
                os.getenv("LONGBRIDGE_APP_KEY")
                and os.getenv("LONGBRIDGE_APP_SECRET")
                and os.getenv("LONGBRIDGE_ACCESS_TOKEN")
            )
        self._available = has_creds
        return has_creds

    def _get_ctx(self):
        if self._ctx is not None:
            return self._ctx
        with self._ctx_lock:
            if self._ctx is not None:
                return self._ctx
            if not self._is_available():
                return None
            try:
                from longbridge.openapi import Config, QuoteContext

                _sanitize_longbridge_env()

                try:
                    from src.config import get_config

                    app_config = get_config()
                    app_key = app_config.longbridge_app_key
                    app_secret = app_config.longbridge_app_secret
                    access_token = app_config.longbridge_access_token
                except Exception:
                    app_key = os.getenv("LONGBRIDGE_APP_KEY")
                    app_secret = os.getenv("LONGBRIDGE_APP_SECRET")
                    access_token = os.getenv("LONGBRIDGE_ACCESS_TOKEN")

                for key, value in {
                    "LONGBRIDGE_APP_KEY": app_key,
                    "LONGBRIDGE_APP_SECRET": app_secret,
                    "LONGBRIDGE_ACCESS_TOKEN": access_token,
                }.items():
                    if value and not os.environ.get(key):
                        os.environ[key] = value

                extra_kw = _longbridge_config_kwargs()
                lb_config = None
                for factory_name in ("from_apikey_env", "from_env"):
                    factory = getattr(Config, factory_name, None)
                    if factory is None:
                        continue
                    try:
                        lb_config = factory()
                        break
                    except Exception as e:
                        logger.debug("[Longbridge] Config.%s() 失败: %s", factory_name, e)

                if lb_config is None:
                    lb_config = Config.from_apikey(app_key, app_secret, access_token, **extra_kw)

                self._config = lb_config
                self._ctx = QuoteContext(lb_config)
                logger.info("[Longbridge] QuoteContext 初始化成功")
                return self._ctx
            except Exception as e:
                logger.warning("[Longbridge] QuoteContext 初始化失败: %s", e)
                self._available = False
                return None

    def _get_static_info(self, symbol: str) -> Optional[Any]:
        """Fetch static info with optional in-process TTL cache."""
        ttl = _static_info_ttl_seconds()
        now = time.time()
        if ttl > 0:
            with self._static_cache_lock:
                cached = self._static_cache.get(symbol)
                if cached and (now - cached[1]) < ttl:
                    return cached[0]

        ctx = self._get_ctx()
        if ctx is None:
            return None
        try:
            infos = ctx.static_info([symbol])
            if infos:
                info = infos[0]
                if ttl > 0:
                    with self._static_cache_lock:
                        self._static_cache[symbol] = (info, now)
                return info
        except Exception as e:
            logger.debug("[Longbridge] static_info(%s) 失败: %s", symbol, e)
            if self._is_connection_error(e):
                self._invalidate_ctx()
        return None

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """Return stock name from Longbridge static_info."""
        symbol = _to_longbridge_symbol(stock_code)
        if symbol is None:
            return None
        info = self._get_static_info(symbol)
        if info is None:
            return None
        name = getattr(info, "name_cn", "") or getattr(info, "name_en", "") or ""
        return name.strip() or None

    def _ts_sort_key(self, candle: Any) -> float:
        """Monotonic sort key for a candle timestamp."""
        ts = getattr(candle, "timestamp", None)
        if ts is None:
            return 0.0
        if hasattr(ts, "timestamp"):
            return float(ts.timestamp())
        return float(int(ts))

    def _compute_volume_ratio(self, symbol: str, today_volume: int) -> Optional[float]:
        """Compute volume_ratio using recent completed daily volumes."""
        if not today_volume or today_volume <= 0:
            return None
        ctx = self._get_ctx()
        if ctx is None:
            return None
        try:
            from longbridge.openapi import AdjustType, Period

            candles = ctx.history_candlesticks_by_offset(
                symbol,
                Period.Day,
                AdjustType.NoAdjust,
                False,
                6,
                datetime.now(),
            )
            if not candles or len(candles) < 2:
                return None
            ordered = sorted(candles, key=self._ts_sort_key, reverse=True)
            past_vols = []
            for candle in ordered[1:6]:
                volume = int(getattr(candle, "volume", 0) or 0)
                if volume > 0:
                    past_vols.append(volume)
            if not past_vols:
                return None
            avg_vol = sum(past_vols) / len(past_vols)
            if avg_vol <= 0:
                return None
            return round(today_volume / avg_vol, 2)
        except Exception as e:
            logger.debug("[Longbridge] 计算量比失败(%s): %s", symbol, e)
            return None

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """Fetch realtime quote from Longbridge."""
        if not self._is_available():
            return None

        symbol = _to_longbridge_symbol(stock_code)
        if symbol is None:
            return None

        ctx = self._get_ctx()
        if ctx is None:
            return None

        try:
            quotes = ctx.quote([symbol])
            if not quotes:
                return None
            quote_data = quotes[0]
        except Exception as e:
            logger.info("[Longbridge] quote(%s) 失败: %s", symbol, e)
            if self._is_connection_error(e):
                self._invalidate_ctx()
            return None

        price = safe_float(getattr(quote_data, "last_done", None))
        if price is None or price <= 0:
            return None

        prev_close = safe_float(getattr(quote_data, "prev_close", None))
        open_price = safe_float(getattr(quote_data, "open", None))
        high = safe_float(getattr(quote_data, "high", None))
        low = safe_float(getattr(quote_data, "low", None))
        volume = int(getattr(quote_data, "volume", 0) or 0)
        turnover = safe_float(getattr(quote_data, "turnover", None))

        change_amount = None
        change_pct = None
        amplitude = None
        if prev_close and prev_close > 0:
            change_amount = round(price - prev_close, 4)
            change_pct = round((price - prev_close) / prev_close * 100, 2)
            if high is not None and low is not None:
                amplitude = round((high - low) / prev_close * 100, 2)

        static = self._get_static_info(symbol)
        turnover_rate = None
        pe_ratio = None
        pb_ratio = None
        total_mv = None
        circ_mv = None
        name = ""

        if static is not None:
            name = getattr(static, "name_cn", "") or getattr(static, "name_en", "") or ""
            circulating = int(getattr(static, "circulating_shares", 0) or 0)
            total_shares = int(getattr(static, "total_shares", 0) or 0)
            eps_ttm = safe_float(getattr(static, "eps_ttm", None))
            eps_plain = safe_float(getattr(static, "eps", None))
            bps = safe_float(getattr(static, "bps", None))

            shares_for_turnover = circulating if circulating > 0 else total_shares
            if shares_for_turnover > 0 and volume > 0:
                turnover_rate = round(volume / shares_for_turnover * 100, 4)

            eps_for_pe = None
            if eps_ttm is not None and eps_ttm > 0:
                eps_for_pe = eps_ttm
            elif eps_plain is not None and eps_plain > 0:
                eps_for_pe = eps_plain
            if eps_for_pe:
                pe_ratio = round(price / eps_for_pe, 2)

            if bps is not None and bps > 0:
                pb_ratio = round(price / bps, 2)
            if total_shares > 0:
                total_mv = round(price * total_shares, 2)
            if circulating > 0:
                circ_mv = round(price * circulating, 2)

        volume_ratio = self._compute_volume_ratio(symbol, volume)

        quote = UnifiedRealtimeQuote(
            code=stock_code,
            name=name,
            source=RealtimeSource.LONGBRIDGE,
            price=price,
            change_pct=change_pct,
            change_amount=change_amount,
            volume=volume if volume > 0 else None,
            amount=turnover,
            volume_ratio=volume_ratio,
            turnover_rate=turnover_rate,
            amplitude=amplitude,
            open_price=open_price,
            high=high,
            low=low,
            pre_close=prev_close,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            total_mv=total_mv,
            circ_mv=circ_mv,
        )
        logger.info(
            "[Longbridge] %s 行情获取成功: 价格=%s, 量比=%s, 换手率=%s",
            symbol,
            price,
            volume_ratio,
            turnover_rate,
        )
        return quote

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical candlesticks from Longbridge."""
        symbol = _to_longbridge_symbol(stock_code)
        if symbol is None:
            raise ValueError(f"Cannot convert {stock_code} to Longbridge symbol")

        ctx = self._get_ctx()
        if ctx is None:
            raise RuntimeError("Longbridge QuoteContext not available")

        from longbridge.openapi import AdjustType, Period

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        try:
            candles = ctx.history_candlesticks_by_date(
                symbol,
                Period.Day,
                AdjustType.ForwardAdjust,
                start_dt,
                end_dt,
            )
        except Exception as e:
            if self._is_connection_error(e):
                self._invalidate_ctx()
            raise

        if not candles:
            return pd.DataFrame()

        rows = []
        for candle in candles:
            ts = getattr(candle, "timestamp", None)
            if ts is None:
                continue
            if hasattr(ts, "date"):
                dt = ts.date()
            else:
                dt = datetime.fromtimestamp(int(ts)).date()
            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "open": safe_float(getattr(candle, "open", None)),
                    "high": safe_float(getattr(candle, "high", None)),
                    "low": safe_float(getattr(candle, "low", None)),
                    "close": safe_float(getattr(candle, "close", None)),
                    "volume": int(getattr(candle, "volume", 0) or 0),
                    "turnover": safe_float(getattr(candle, "turnover", None)),
                }
            )
        return pd.DataFrame(rows)

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """Normalize column names to standard format."""
        if df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        df = df.rename(columns={"turnover": "amount"})
        if "pct_chg" not in df.columns and "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[STANDARD_COLUMNS]
