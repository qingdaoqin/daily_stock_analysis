#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare 股票列表获取脚本

从 Tushare Pro 获取 A股、港股、美股列表信息，保存为 CSV 文件

使用方法：
    python3 scripts/fetch_tushare_stock_list.py

环境要求：
    - 需要在 .env 中配置 TUSHARE_TOKEN
    - 需要安装 tushare: pip install tushare
    - 账号积分要求：
        * A股/港股：2000积分
        * 美股：120积分试用，5000积分正式权限

输出文件：
    - data/stock_list_a.csv      A股列表
    - data/stock_list_hk.csv     港股列表
    - data/stock_list_us.csv     美股列表
    - data/README_stock_list.md  数据说明文档
"""

import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tushare as ts
except ImportError:
    print("[错误] 未安装 tushare 库")
    print("请执行: pip install tushare")
    sys.exit(1)


# 配置
load_dotenv()

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
OUTPUT_DIR = Path(__file__).parent.parent / "data"
PAGE_SIZE = 5000  # 美股每页读取数量（API 最大6000，设置5000留余量）
SLEEP_MIN = 5     # 最小睡眠时间（秒）
SLEEP_MAX = 10    # 最大睡眠时间（秒）


def get_tushare_api() -> Optional[ts.pro_api]:
    """
    获取 Tushare API 实例

    Returns:
        Tushare API 实例，失败返回 None
    """
    if not TUSHARE_TOKEN:
        print("[错误] 未找到 TUSHARE_TOKEN")
        print("请在 .env 文件中配置: TUSHARE_TOKEN=你的token")
        return None

    try:
        api = ts.pro_api(TUSHARE_TOKEN)
        # 测试连接
        api.trade_cal(exchange='SSE', start_date='20240101', end_date='20240101')
        print("✓ Tushare API 连接成功")
        return api
    except Exception as e:
        print(f"[错误] Tushare API 连接失败: {e}")
        print("请检查：")
        print("  1. TUSHARE_TOKEN 是否正确")
        print("  2. 账号积分是否足够（A股/港股需要2000积分）")
        return None


def random_sleep(min_seconds: int = SLEEP_MIN, max_seconds: int = SLEEP_MAX):
    """
    随机睡眠，避免频繁请求

    Args:
        min_seconds: 最小睡眠时间
        max_seconds: 最大睡眠时间
    """
    sleep_time = random.uniform(min_seconds, max_seconds)
    print(f"  ⏱  休息 {sleep_time:.1f} 秒...")
    time.sleep(sleep_time)


def fetch_a_stock_list(api: ts.pro_api) -> Optional[pd.DataFrame]:
    """
    获取 A股列表

    接口：stock_basic
    限量：单次最多6000行（覆盖全市场A股）

    Args:
        api: Tushare API 实例

    Returns:
        A股数据 DataFrame，失败返回 None
    """
    print("\n[1/3] 正在获取 A股列表...")

    try:
        # 获取所有正常上市的股票
        df = api.stock_basic(
            exchange='',        # 空：全部交易所
            list_status='L',    # L: 上市, D: 退市, P: 暂停上市
            fields='ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type'
        )

        if df is not None and len(df) > 0:
            print(f"✓ A股列表获取成功，共 {len(df)} 只股票")
            print("  - 交易所分布：")
            for exchange, count in df['exchange'].value_counts().items():
                print(f"    {exchange}: {count} 只")
            return df
        else:
            print("[错误] A股数据为空")
            return None

    except Exception as e:
        print(f"[错误] 获取 A股列表失败: {e}")
        return None


def fetch_hk_stock_list(api: ts.pro_api) -> Optional[pd.DataFrame]:
    """
    获取港股列表

    接口：hk_basic
    限量：单次可提取全部在交易的港股

    Args:
        api: Tushare API 实例

    Returns:
        港股数据 DataFrame，失败返回 None
    """
    print("\n[2/3] 正在获取港股列表...")

    try:
        # 获取所有正常上市的港股
        df = api.hk_basic(
            list_status='L'    # L: 上市, D: 退市
        )

        if df is not None and len(df) > 0:
            print(f"✓ 港股列表获取成功，共 {len(df)} 只股票")
            return df
        else:
            print("[错误] 港股数据为空")
            return None

    except Exception as e:
        print(f"[错误] 获取港股列表失败: {e}")
        return None


def fetch_us_stock_list(api: ts.pro_api) -> Optional[pd.DataFrame]:
    """
    获取美股列表（分页读取）

    接口：us_basic
    限量：单次最大6000，需要分页提取

    Args:
        api: Tushare API 实例

    Returns:
        美股数据 DataFrame，失败返回 None
    """
    print("\n[3/3] 正在获取美股列表（分页读取）...")

    all_data = []
    offset = 0
    page = 1

    try:
        while True:
            print(f"  第 {page} 页（offset={offset}）...")

            df = api.us_basic(
                offset=offset,
                limit=PAGE_SIZE
            )

            if df is None or len(df) == 0:
                print(f"  ✓ 第 {page} 页无数据，读取完成")
                break

            all_data.append(df)
            print(f"  ✓ 第 {page} 页获取 {len(df)} 只股票")

            # 如果返回数据少于页大小，说明已经到最后一页
            if len(df) < PAGE_SIZE:
                break

            offset += PAGE_SIZE
            page += 1

            # 随机休息（最后一页不需要休息）
            random_sleep()

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            print(f"✓ 美股列表获取成功，共 {len(result_df)} 只股票（{page} 页）")

            # 按分类统计
            if 'classify' in result_df.columns:
                print("  - 分类分布：")
                for classify, count in result_df['classify'].value_counts().items():
                    print(f"    {classify}: {count} 只")

            return result_df
        else:
            print("[错误] 美股数据为空")
            return None

    except Exception as e:
        print(f"[错误] 获取美股列表失败: {e}")
        return None


def save_to_csv(df: pd.DataFrame, filename: str, market_name: str) -> bool:
    """
    保存数据到 CSV 文件

    Args:
        df: 数据 DataFrame
        filename: 文件名
        market_name: 市场名称（用于日志）

    Returns:
        是否保存成功
    """
    if df is None or len(df) == 0:
        print(f"[跳过] {market_name} 数据为空，不保存文件")
        return False

    try:
        output_path = OUTPUT_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        file_size = output_path.stat().st_size / 1024  # KB
        print(f"✓ {market_name} 数据已保存：{output_path} ({file_size:.2f} KB)")
        return True

    except Exception as e:
        print(f"[错误] 保存 {market_name} 数据失败: {e}")
        return False


def generate_data_documentation(
    a_df: Optional[pd.DataFrame],
    hk_df: Optional[pd.DataFrame],
    us_df: Optional[pd.DataFrame]
):
    """
    生成数据说明文档

    Args:
        a_df: A股数据
        hk_df: 港股数据
        us_df: 美股数据
    """
    doc_path = OUTPUT_DIR / "README_stock_list.md"

    content = f"""# Tushare 股票列表数据说明

> 数据来源：[Tushare Pro](https://tushare.pro)
> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 文件说明

| 文件 | 说明 | 记录数 |
|------|------|--------|
| `stock_list_a.csv` | A股列表 | {len(a_df) if a_df is not None else 0} |
| `stock_list_hk.csv` | 港股列表 | {len(hk_df) if hk_df is not None else 0} |
| `stock_list_us.csv` | 美股列表 | {len(us_df) if us_df is not None else 0} |

## 使用说明

```python
import pandas as pd

# 读取数据
a_stocks = pd.read_csv('data/stock_list_a.csv')
hk_stocks = pd.read_csv('data/stock_list_hk.csv')
us_stocks = pd.read_csv('data/stock_list_us.csv')
```

## 生成自动补全索引

```bash
python3 scripts/generate_index_from_csv.py
```
"""

    try:
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 数据说明文档已生成：{doc_path}")
    except Exception as e:
        print(f"[错误] 生成说明文档失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("Tushare 股票列表获取工具")
    print("=" * 60)

    # 1. 获取 API 实例
    api = get_tushare_api()
    if not api:
        return 1

    # 2. 获取 A股数据
    a_df = fetch_a_stock_list(api)
    if a_df is not None:
        save_to_csv(a_df, 'stock_list_a.csv', 'A股')

    # 3. 获取港股数据
    random_sleep()  # 休息后再获取港股
    hk_df = fetch_hk_stock_list(api)
    if hk_df is not None:
        save_to_csv(hk_df, 'stock_list_hk.csv', '港股')

    # 4. 获取美股数据（分页）
    random_sleep()  # 休息后再获取美股
    us_df = fetch_us_stock_list(api)
    if us_df is not None:
        save_to_csv(us_df, 'stock_list_us.csv', '美股')

    # 5. 生成数据说明文档
    print("\n正在生成数据说明文档...")
    generate_data_documentation(a_df, hk_df, us_df)

    # 6. 总结
    print("\n" + "=" * 60)
    print("任务完成！")
    print("=" * 60)

    total_count = 0
    if a_df is not None:
        total_count += len(a_df)
        print(f"  ✓ A股：{len(a_df)} 只")
    if hk_df is not None:
        total_count += len(hk_df)
        print(f"  ✓ 港股：{len(hk_df)} 只")
    if us_df is not None:
        total_count += len(us_df)
        print(f"  ✓ 美股：{len(us_df)} 只")

    print(f"\n总计：{total_count} 只股票")
    print(f"输出目录：{OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[中断] 用户取消操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 未预期的异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
