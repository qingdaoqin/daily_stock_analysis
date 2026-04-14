# -*- coding: utf-8 -*-
"""
===================================
TickFlowFetcher - 大盘复盘专用（market review only）
===================================

Issue #632 仅要求 TickFlow 用于 A 股大盘复盘稳定性增强。
此 fetcher 实现了一个窄 P0 接口：

1. A 股主要指数实时报价
2. A 股市场涨跌统计

不参与常规 daily / per-stock realtime 管道，
只在 DataFetcherManager 明确调用时生效。
使用前必须配置环境变量 TICKFLOW_API_KEY。
"""

import logging
from threading import RLock
from time import monotonic
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import (
    BaseFetcher,
    DataFetchError,
    normalize_stock_code,
)


logger = logging.getLogger(__name__)

_CN_MAIN_INDEX_QUOTES = (
    ("000001.SH", "000001", "上证指数"),
    ("399001.SZ", "399001", "深证成指"),
    ("399006.SZ", "399006", "创业板指"),
    ("000688.SH", "000688", "科创50"),
    ("000016.SH", "000016", "上证50"),
    ("000300.SH", "000300", "沪深300"),
)
_MAX_SYMBOLS_PER_QUOTE_REQUEST = 5
_UNIVERSE_PERMISSION_NEGATIVE_CACHE_TTL_SECONDS = 900


class TickFlowFetcher(BaseFetcher):
    """TickFlow 数据源 —— 专用于大盘复盘。

    只实现 get_main_indices / get_market_stats；
    _fetch_raw_data / _normalize_data 不支持，直接抛 DataFetchError。
    """

    name = "TickFlowFetcher"
    priority = 99  # 最低优先级，仅在 market review 明确调用时生效

    def __init__(self, api_key: Optional[str], timeout: float = 30.0):
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self._client = None
        self._client_lock = RLock()
        self._universe_query_supported: Optional[bool] = None
        self._universe_query_checked_at: Optional[float] = None

    def close(self) -> None:
        """关闭底层 TickFlow client（若已创建）。"""
        with self._client_lock:
            client = self._client
            self._client = None
            self._universe_query_supported = None
            self._universe_query_checked_at = None
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                logger.debug("[TickFlowFetcher] 关闭客户端失败: %s", exc)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # 解释器关闭阶段尽力清理
            pass

    def _build_client(self):
        from tickflow import TickFlow  # type: ignore[import]

        return TickFlow(api_key=self.api_key, timeout=self.timeout)

    def _get_client(self):
        if not self.api_key:
            return None
        if self._client is not None:
            return self._client

        with self._client_lock:
            if self._client is None:
                try:
                    self._client = self._build_client()
                except ImportError:
                    logger.warning(
                        "[TickFlowFetcher] tickflow 包未安装，请 pip install tickflow；已降级跳过"
                    )
                    return None
                except Exception as exc:
                    logger.warning("[TickFlowFetcher] 初始化客户端失败: %s", exc)
                    return None
            return self._client

    def _fetch_raw_data(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        raise DataFetchError(
            "TickFlowFetcher P0 仅支持大盘复盘端点（get_main_indices / get_market_stats）"
        )

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        raise DataFetchError(
            "TickFlowFetcher P0 仅支持大盘复盘端点（get_main_indices / get_market_stats）"
        )

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value in (None, "", "-"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _ratio_to_percent(cls, value: Any) -> Optional[float]:
        ratio = cls._safe_float(value)
        if ratio is None:
            return None
        return ratio * 100.0

    @staticmethod
    def _extract_name(quote: Dict[str, Any]) -> str:
        ext = quote.get("ext") or {}
        name = ext.get("name") or quote.get("name") or ""
        return str(name).strip()

    @staticmethod
    def _is_universe_permission_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        code = str(getattr(exc, "code", "") or "").upper()
        message = f"{getattr(exc, 'message', '')} {exc}".strip().lower()

        if status_code == 403:
            return True
        if code in {"PERMISSION_DENIED", "FORBIDDEN"}:
            return True
        return any(
            keyword in message
            for keyword in (
                "universe",
                "permission",
                "forbidden",
                "权限",
            )
        )

    @staticmethod
    def _is_cn_equity_symbol(symbol: str) -> bool:
        normalized = normalize_stock_code(symbol)
        if not (normalized.isdigit() and len(normalized) == 6):
            return False

        upper_symbol = (symbol or "").strip().upper()
        return not (upper_symbol.startswith("HK") or upper_symbol.endswith(".HK"))



    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        """通过 TickFlow quotes 接口获取 A 股主要指数行情。

        Args:
            region: 市场区域，当前仅支持 "cn"。

        Returns:
            标准化指数行情列表，或 None（client 不可用 / 查询失败）。
        """
        if region != "cn":
            return None

        client = self._get_client()
        if client is None:
            return None

        symbols = [symbol for symbol, _, _ in _CN_MAIN_INDEX_QUOTES]
        quotes: List[Dict[str, Any]] = []
        for offset in range(0, len(symbols), _MAX_SYMBOLS_PER_QUOTE_REQUEST):
            batch_symbols = symbols[offset : offset + _MAX_SYMBOLS_PER_QUOTE_REQUEST]
            try:
                batch_quotes = client.quotes.get(symbols=batch_symbols)
                if batch_quotes:
                    quotes.extend(batch_quotes)
            except Exception as exc:
                logger.warning("[TickFlowFetcher] 指数报价批次失败 %s: %s", batch_symbols, exc)
                continue

        if not quotes:
            logger.warning("[TickFlowFetcher] 指数行情为空")
            return None

        # 构建 symbol → (code, name) 的查找表
        symbol_meta: Dict[str, tuple] = {
            symbol: (code, name) for symbol, code, name in _CN_MAIN_INDEX_QUOTES
        }

        result: List[Dict[str, Any]] = []
        for quote in quotes:
            symbol = quote.get("symbol") or ""
            meta = symbol_meta.get(symbol)
            if meta is None:
                continue
            code, fallback_name = meta
            name = self._extract_name(quote) or fallback_name
            current = self._safe_float(quote.get("last") or quote.get("close"))
            prev_close = self._safe_float(quote.get("prev_close"))
            change = None
            change_pct = None
            if current is not None and prev_close is not None and prev_close != 0:
                change = round(current - prev_close, 2)
                change_pct = round((current - prev_close) / prev_close * 100, 2)
            result.append(
                {
                    "code": code,
                    "name": name,
                    "current": current,
                    "change": change,
                    "change_pct": change_pct,
                    "volume": self._safe_float(quote.get("volume")),
                    "amount": self._safe_float(quote.get("amount") or quote.get("turnover")),
                }
            )

        return result if result else None

    def get_market_stats(self, region: str = "cn") -> Optional[Dict[str, Any]]:
        """通过 TickFlow universe 接口获取 A 股全市场涨跌统计。

        需要 TickFlow 套餐包含 universe 查询权限；
        若无权限，会在首次失败后缓存（TTL 900 秒）并跳过后续请求。

        Args:
            region: 市场区域，当前仅支持 "cn"。

        Returns:
            涨跌统计字典，或 None（权限不足 / 查询失败）。
        """
        if region != "cn":
            return None

        client = self._get_client()
        if client is None:
            return None

        # 负缓存：权限不足时停止重复尝试
        now = monotonic()
        if self._universe_query_supported is False:
            if (
                self._universe_query_checked_at is not None
                and now - self._universe_query_checked_at
                < _UNIVERSE_PERMISSION_NEGATIVE_CACHE_TTL_SECONDS
            ):
                logger.debug("[TickFlowFetcher] universe 查询负缓存中，跳过")
                return None
            # TTL 到期，重置并重试
            self._universe_query_supported = None
            self._universe_query_checked_at = None

        try:
            # TickFlow universe.list() 返回全市场股票快照
            all_stocks = client.universe.list(
                filter={"exchange": ["SH", "SZ"]},
                fields=[
                    "symbol",
                    "last",
                    "prev_close",
                    "change",
                    "change_ratio",
                    "is_limit_up",
                    "is_limit_down",
                    "amount",
                    "turnover",
                ],
            )
        except Exception as exc:
            if self._is_universe_permission_error(exc):
                logger.warning(
                    "[TickFlowFetcher] universe 查询无权限，后续 %s 秒内不再重试: %s",
                    _UNIVERSE_PERMISSION_NEGATIVE_CACHE_TTL_SECONDS,
                    exc,
                )
                self._universe_query_supported = False
                self._universe_query_checked_at = now
            else:
                logger.warning("[TickFlowFetcher] universe 查询失败: %s", exc)
            return None

        self._universe_query_supported = True

        if not all_stocks:
            return None

        up_count = 0
        down_count = 0
        flat_count = 0
        limit_up_count = 0
        limit_down_count = 0
        total_amount = 0.0

        for stock in all_stocks:
            if not self._is_cn_equity_symbol(stock.get("symbol", "")):
                continue

            chg_ratio = self._safe_float(stock.get("change_ratio"))
            if chg_ratio is None:
                # 兜底：通过 last / prev_close 计算
                last = self._safe_float(stock.get("last"))
                prev = self._safe_float(stock.get("prev_close"))
                if last is not None and prev and prev != 0:
                    chg_ratio = (last - prev) / prev * 100

            if chg_ratio is not None:
                if chg_ratio > 0.01:
                    up_count += 1
                elif chg_ratio < -0.01:
                    down_count += 1
                else:
                    flat_count += 1

            if stock.get("is_limit_up"):
                limit_up_count += 1
            if stock.get("is_limit_down"):
                limit_down_count += 1

            amt = self._safe_float(stock.get("amount") or stock.get("turnover"))
            if amt:
                total_amount += amt

        return {
            "up_count": up_count,
            "down_count": down_count,
            "flat_count": flat_count,
            "limit_up_count": limit_up_count,
            "limit_down_count": limit_down_count,
            "total_amount": round(total_amount, 2),
        }
