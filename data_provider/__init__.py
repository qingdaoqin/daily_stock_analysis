# -*- coding: utf-8 -*-
"""
===================================
数据源策略层 - 包初始化
===================================

为避免仅导入单个子模块时触发所有数据源依赖，这里改为惰性导出。
例如 `src.search_service` 只需要 `fundamental_adapter`，不应因为
`data_provider.__init__` 预加载 `efinance_fetcher` 而要求安装
`fake_useragent` 等可选依赖。
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BaseFetcher": ("data_provider.base", "BaseFetcher"),
    "DataFetcherManager": ("data_provider.base", "DataFetcherManager"),
    "EfinanceFetcher": ("data_provider.efinance_fetcher", "EfinanceFetcher"),
    "AkshareFetcher": ("data_provider.akshare_fetcher", "AkshareFetcher"),
    "is_hk_stock_code": ("data_provider.akshare_fetcher", "is_hk_stock_code"),
    "TushareFetcher": ("data_provider.tushare_fetcher", "TushareFetcher"),
    "PytdxFetcher": ("data_provider.pytdx_fetcher", "PytdxFetcher"),
    "BaostockFetcher": ("data_provider.baostock_fetcher", "BaostockFetcher"),
    "YfinanceFetcher": ("data_provider.yfinance_fetcher", "YfinanceFetcher"),
    "is_us_index_code": ("data_provider.us_index_mapping", "is_us_index_code"),
    "is_us_stock_code": ("data_provider.us_index_mapping", "is_us_stock_code"),
    "get_us_index_yf_symbol": ("data_provider.us_index_mapping", "get_us_index_yf_symbol"),
    "US_INDEX_MAPPING": ("data_provider.us_index_mapping", "US_INDEX_MAPPING"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if not target:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = list(_EXPORTS)
