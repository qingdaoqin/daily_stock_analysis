# -*- coding: utf-8 -*-
"""
Market context detection for LLM prompts.

Detects the market (A-shares, HK, US) from a stock code and returns
market-specific role descriptions so prompts are not hardcoded to a
single market.

Fixes: https://github.com/ZhuLinsen/daily_stock_analysis/issues/644
"""

import re
from typing import Optional


def detect_market(stock_code: Optional[str]) -> str:
    """Detect market from stock code.

    Returns:
        One of 'cn', 'hk', 'us', or 'cn' as fallback.
    """
    if not stock_code:
        return "cn"

    code = stock_code.strip().upper()

    # HK stocks: HK00700, 00700.HK, or 5-digit pure numbers
    if code.startswith("HK") or code.endswith(".HK"):
        return "hk"
    lower = code.lower()
    if lower.endswith(".hk"):
        return "hk"
    # 5-digit pure numbers are HK (A-shares are 6-digit)
    if code.isdigit() and len(code) == 5:
        return "hk"

    # US stocks: 1-5 uppercase letters (AAPL, TSLA, GOOGL)
    # Also handles suffixed forms like BRK.B
    if re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', code):
        return "us"

    # Default: A-shares (6-digit numbers like 600519, 000001)
    return "cn"


# -- Market-specific role descriptions --

_MARKET_ROLES = {
    "cn": {
        "zh": " A 股",
        "en": "China A-shares",
    },
    "hk": {
        "zh": "港股",
        "en": "Hong Kong stock",
    },
    "us": {
        "zh": "美股",
        "en": "US stock",
    },
}

_MARKET_GUIDELINES = {
    "cn": {
        "zh": (
            "- 本次分析对象为 **A 股**（中国沪深交易所上市股票）。\n"
            "- 请关注 A 股特有的涨跌停机制（±10%/±20%/±30%）、T+1 交易制度及相关政策因素。"
        ),
        "en": (
            "- This analysis covers a **China A-share** (listed on Shanghai/Shenzhen exchanges).\n"
            "- Consider A-share-specific rules: daily price limits (±10%/±20%/±30%), T+1 settlement, and PRC policy factors."
        ),
    },
    "hk": {
        "zh": (
            "- 本次分析对象为 **港股**（香港交易所上市股票）。\n"
            "- 港股无涨跌停限制，支持 T+0 交易，需关注港币汇率、南北向资金流及联交所特有规则。"
        ),
        "en": (
            "- This analysis covers a **Hong Kong stock** (listed on HKEX).\n"
            "- HK stocks have no daily price limits, allow T+0 trading. Consider HKD FX, Southbound/Northbound flows, and HKEX-specific rules."
        ),
    },
    "us": {
        "zh": (
            "- 本次分析对象为 **美股**（美国交易所上市股票）。\n"
            "- 美股无涨跌停限制（但有熔断机制），支持 T+0 交易和盘前盘后交易，需关注美元汇率、美联储政策及 SEC 监管动态。"
        ),
        "en": (
            "- This analysis covers a **US stock** (listed on NYSE/NASDAQ).\n"
            "- US stocks have no daily price limits (but have circuit breakers), allow T+0 and pre/after-market trading. Consider USD FX, Fed policy, and SEC regulations."
        ),
    },
}

# -- Market-specific scoring criteria --

_MARKET_SCORING = {
    "cn": {
        "zh": (
            "评分采用「技术基准 × 技能共振 × 风险否决」三层模型。\n\n"
            "**第一层 · 技术面基准**（决定初始分区）\n\n"
            "### 强烈买入（80-100分）：\n"
            "- ✅ 多头排列：MA5 > MA10 > MA20\n"
            "- ✅ 低乖离率：<2%，最佳买点\n"
            "- ✅ 缩量回调或放量突破\n"
            "- ✅ 筹码集中健康\n"
            "- ✅ 消息面有利好催化\n\n"
            "### 买入（60-79分）：\n"
            "- ✅ 多头排列或弱势多头\n"
            "- ✅ 乖离率 <5%\n"
            "- ✅ 量能正常\n"
            "- ⚪ 允许一项次要条件不满足\n\n"
            "### 观望（40-59分）：\n"
            "- ⚠️ 乖离率 >5%（追高风险）\n"
            "- ⚠️ 均线缠绕趋势不明\n"
            "- ⚠️ 有风险事件（监管函、减持、业绩预警）\n\n"
            "### 卖出/减仓（0-39分）：\n"
            "- ❌ 空头排列\n"
            "- ❌ 跌破MA20\n"
            "- ❌ 放量下跌\n"
            "- ❌ 重大利空（立案调查、退市风险、大额减持）\n\n"
            "**第二层 · 技能共振修正**（在基准分区内上下调整）\n"
            "- 📈 多个激活技能同时给出积极信号 → 在当前分区内向上调整 5-10 分\n"
            "- ⚖️ 技能信号相互矛盾（如趋势看多但量能看空）→ 降低一档，归入观望区\n"
            "- 📊 技能结论与技术指标方向一致 → 提升 confidence_level\n"
            "- 🔄 技能结论与技术指标方向矛盾 → 降低 confidence_level\n\n"
            "**第三层 · 风险否决**（命中关键风险则强制压分）\n"
            "- 🚫 监管函/问询函/立案调查 → 无论技术面多好，评分上限 55 分\n"
            "- 🚫 退市风险警示（*ST）→ 评分上限 30 分\n"
            "- 🚫 大股东大额减持（占总股本 >1%）→ 评分减 15 分\n"
            "- 🚫 业绩预告亏损/大幅下滑 → 评分减 10 分"
        ),
        "en": (
            "Scoring uses a three-layer model: Technical Baseline × Skill Resonance × Risk Veto.\n\n"
            "**Layer 1 · Technical Baseline** (determines initial score range)\n\n"
            "### Strong Buy (80-100):\n"
            "- ✅ Bullish alignment: MA5 > MA10 > MA20\n"
            "- ✅ Low bias rate: <2%, optimal entry\n"
            "- ✅ Low-volume pullback or high-volume breakout\n"
            "- ✅ Healthy chip concentration\n"
            "- ✅ Positive news catalyst\n\n"
            "### Buy (60-79):\n"
            "- ✅ Bullish or weak-bullish alignment\n"
            "- ✅ Bias rate <5%\n"
            "- ✅ Normal volume\n"
            "- ⚪ One minor condition may be unmet\n\n"
            "### Hold (40-59):\n"
            "- ⚠️ Bias rate >5% (chasing risk)\n"
            "- ⚠️ MA entangled, unclear trend\n"
            "- ⚠️ Risk events (regulatory, insider selling)\n\n"
            "### Sell/Reduce (0-39):\n"
            "- ❌ Bearish alignment\n"
            "- ❌ Price below MA20\n"
            "- ❌ Heavy-volume decline\n"
            "- ❌ Major negative (investigation, delisting risk)\n\n"
            "**Layer 2 · Skill Resonance Adjustment** (adjust within baseline range)\n"
            "- 📈 Multiple active skills give bullish signals → adjust up 5-10 points within range\n"
            "- ⚖️ Skill signals conflict (e.g. trend bullish but volume bearish) → downgrade one tier to Hold\n"
            "- 📊 Skill conclusions align with technicals → raise confidence_level\n"
            "- 🔄 Skill conclusions contradict technicals → lower confidence_level\n\n"
            "**Layer 3 · Risk Veto** (critical risks override score)\n"
            "- 🚫 Regulatory letter / inquiry / investigation → cap score at 55\n"
            "- 🚫 Delisting risk warning (*ST) → cap score at 30\n"
            "- 🚫 Major insider selling (>1% of total shares) → deduct 15 points\n"
            "- 🚫 Earnings warning (loss or sharp decline) → deduct 10 points"
        ),
    },
    "hk": {
        "zh": (
            "评分采用「技术基准 × 技能共振 × 风险否决」三层模型。\n\n"
            "**第一层 · 技术面基准**（决定初始分区）\n\n"
            "### 强烈买入（80-100分）：\n"
            "- ✅ 多头排列：MA5 > MA10 > MA20，且 MA50 向上\n"
            "- ✅ 低乖离率：<3%（港股波动大于A股，阈值适当放宽）\n"
            "- ✅ 成交量较近期均量放大（港股常年低换手，需对比自身均量）\n"
            "- ✅ 南向资金持续净流入\n"
            "- ✅ 消息面有利好催化\n\n"
            "### 买入（60-79分）：\n"
            "- ✅ 多头排列或弱势多头\n"
            "- ✅ 乖离率 <6%\n"
            "- ✅ 量能较自身均值正常\n"
            "- ⚪ 允许一项次要条件不满足\n\n"
            "### 观望（40-59分）：\n"
            "- ⚠️ 乖离率 >6%（追高风险）\n"
            "- ⚠️ 均线缠绕趋势不明\n"
            "- ⚠️ 有风险事件（沽空比率骤升、盈利预警、大股东减持）\n\n"
            "### 卖出/减仓（0-39分）：\n"
            "- ❌ 空头排列\n"
            "- ❌ 跌破MA50\n"
            "- ❌ 放量下跌且沽空比率上升\n"
            "- ❌ 重大利空（SFC 调查、做空报告、停牌风险）\n\n"
            "**第二层 · 技能共振修正**（在基准分区内上下调整）\n"
            "- 📈 多个激活技能同时给出积极信号 → 在当前分区内向上调整 5-10 分\n"
            "- ⚖️ 技能信号相互矛盾 → 降低一档，归入观望区\n"
            "- 📊 技能结论与技术指标方向一致 → 提升 confidence_level\n"
            "- 🔄 技能结论与技术指标方向矛盾 → 降低 confidence_level\n\n"
            "**第三层 · 风险否决**（命中关键风险则强制压分）\n"
            "- 🚫 SFC 调查/警告 → 评分上限 55 分\n"
            "- 🚫 做空机构报告 → 评分上限 45 分\n"
            "- 🚫 沽空比率突破 20% → 评分减 15 分\n"
            "- 🚫 盈利预警公告 → 评分减 10 分\n"
            "- 🚫 停牌风险 → 评分上限 30 分"
        ),
        "en": (
            "Scoring uses a three-layer model: Technical Baseline × Skill Resonance × Risk Veto.\n\n"
            "**Layer 1 · Technical Baseline** (determines initial score range)\n\n"
            "### Strong Buy (80-100):\n"
            "- ✅ Bullish alignment: MA5 > MA10 > MA20, MA50 trending up\n"
            "- ✅ Low bias rate: <3% (HK stocks are more volatile, wider threshold)\n"
            "- ✅ Volume above recent average (HK stocks trade thin; compare to own average)\n"
            "- ✅ Sustained Southbound net inflow\n"
            "- ✅ Positive news catalyst\n\n"
            "### Buy (60-79):\n"
            "- ✅ Bullish or weak-bullish alignment\n"
            "- ✅ Bias rate <6%\n"
            "- ✅ Normal volume vs own average\n"
            "- ⚪ One minor condition may be unmet\n\n"
            "### Hold (40-59):\n"
            "- ⚠️ Bias rate >6% (chasing risk)\n"
            "- ⚠️ MA entangled, unclear trend\n"
            "- ⚠️ Risk events (short-selling spike, profit warning, major shareholder sale)\n\n"
            "### Sell/Reduce (0-39):\n"
            "- ❌ Bearish alignment\n"
            "- ❌ Price below MA50\n"
            "- ❌ Heavy-volume decline with rising short interest\n"
            "- ❌ Major negative (SFC investigation, short-seller report, suspension risk)\n\n"
            "**Layer 2 · Skill Resonance Adjustment** (adjust within baseline range)\n"
            "- 📈 Multiple active skills give bullish signals → adjust up 5-10 points within range\n"
            "- ⚖️ Skill signals conflict → downgrade one tier to Hold\n"
            "- 📊 Skill conclusions align with technicals → raise confidence_level\n"
            "- 🔄 Skill conclusions contradict technicals → lower confidence_level\n\n"
            "**Layer 3 · Risk Veto** (critical risks override score)\n"
            "- 🚫 SFC investigation / warning → cap score at 55\n"
            "- 🚫 Short-seller report → cap score at 45\n"
            "- 🚫 Short-selling ratio exceeds 20% → deduct 15 points\n"
            "- 🚫 Profit warning announcement → deduct 10 points\n"
            "- 🚫 Suspension risk → cap score at 30"
        ),
    },
    "us": {
        "zh": (
            "评分采用「技术基准 × 技能共振 × 风险否决」三层模型。\n\n"
            "**第一层 · 技术面基准**（决定初始分区）\n\n"
            "### 强烈买入（80-100分）：\n"
            "- ✅ 多头排列：MA10 > MA50 > MA200（美股机构重视长周期均线）\n"
            "- ✅ 低乖离率：<5%（美股日内波动大，阈值宽于A股）\n"
            "- ✅ 放量突破关键阻力位或缩量回调至支撑位\n"
            "- ✅ 财报超预期或 guidance 上调\n"
            "- ✅ 期权市场看涨情绪偏多（如有数据）\n\n"
            "### 买入（60-79分）：\n"
            "- ✅ 多头排列或弱势多头\n"
            "- ✅ 乖离率 <8%\n"
            "- ✅ 量能正常\n"
            "- ⚪ 允许一项次要条件不满足\n\n"
            "### 观望（40-59分）：\n"
            "- ⚠️ 乖离率 >8%（追高风险）\n"
            "- ⚠️ 均线缠绕趋势不明\n"
            "- ⚠️ 有风险事件（SEC 调查、做空报告、集体诉讼）\n\n"
            "### 卖出/减仓（0-39分）：\n"
            "- ❌ 空头排列，跌破 MA200\n"
            "- ❌ 放量下跌\n"
            "- ❌ 财报不及预期或 guidance 下修\n"
            "- ❌ 重大利空（做空报告、SEC 调查、关税/制裁冲击）\n\n"
            "**第二层 · 技能共振修正**（在基准分区内上下调整）\n"
            "- 📈 多个激活技能同时给出积极信号 → 在当前分区内向上调整 5-10 分\n"
            "- ⚖️ 技能信号相互矛盾 → 降低一档，归入观望区\n"
            "- 📊 技能结论与技术指标方向一致 → 提升 confidence_level\n"
            "- 🔄 技能结论与技术指标方向矛盾 → 降低 confidence_level\n\n"
            "**第三层 · 风险否决**（命中关键风险则强制压分）\n"
            "- 🚫 SEC 调查/集体诉讼 → 评分上限 55 分\n"
            "- 🚫 做空机构报告（Hindenburg/Muddy Waters 等）→ 评分上限 45 分\n"
            "- 🚫 财报不及预期 + guidance 下修 → 评分减 15 分\n"
            "- 🚫 关税/出口管制/制裁冲击 → 评分减 10 分\n"
            "- 🚫 13F 机构持仓大幅减少 → 评分减 10 分"
        ),
        "en": (
            "Scoring uses a three-layer model: Technical Baseline × Skill Resonance × Risk Veto.\n\n"
            "**Layer 1 · Technical Baseline** (determines initial score range)\n\n"
            "### Strong Buy (80-100):\n"
            "- ✅ Bullish alignment: MA10 > MA50 > MA200 (institutional focus on longer MAs)\n"
            "- ✅ Low bias rate: <5% (US stocks have wider intraday swings)\n"
            "- ✅ Volume breakout above resistance or low-volume pullback to support\n"
            "- ✅ Earnings beat or guidance raise\n"
            "- ✅ Bullish options sentiment (if data available)\n\n"
            "### Buy (60-79):\n"
            "- ✅ Bullish or weak-bullish alignment\n"
            "- ✅ Bias rate <8%\n"
            "- ✅ Normal volume\n"
            "- ⚪ One minor condition may be unmet\n\n"
            "### Hold (40-59):\n"
            "- ⚠️ Bias rate >8% (chasing risk)\n"
            "- ⚠️ MA entangled, unclear trend\n"
            "- ⚠️ Risk events (SEC investigation, short-seller report, class action)\n\n"
            "### Sell/Reduce (0-39):\n"
            "- ❌ Bearish alignment, price below MA200\n"
            "- ❌ Heavy-volume decline\n"
            "- ❌ Earnings miss or guidance cut\n"
            "- ❌ Major negative (short-seller report, SEC probe, tariff/sanctions impact)\n\n"
            "**Layer 2 · Skill Resonance Adjustment** (adjust within baseline range)\n"
            "- 📈 Multiple active skills give bullish signals → adjust up 5-10 points within range\n"
            "- ⚖️ Skill signals conflict → downgrade one tier to Hold\n"
            "- 📊 Skill conclusions align with technicals → raise confidence_level\n"
            "- 🔄 Skill conclusions contradict technicals → lower confidence_level\n\n"
            "**Layer 3 · Risk Veto** (critical risks override score)\n"
            "- 🚫 SEC investigation / class-action lawsuit → cap score at 55\n"
            "- 🚫 Short-seller report (Hindenburg / Muddy Waters etc.) → cap score at 45\n"
            "- 🚫 Earnings miss + guidance cut → deduct 15 points\n"
            "- 🚫 Tariff / export control / sanctions impact → deduct 10 points\n"
            "- 🚫 Significant 13F institutional position reduction → deduct 10 points"
        ),
    },
}

# -- Market-specific risk checklist --

_MARKET_RISK_CHECKLIST = {
    "cn": {
        "zh": (
            "- 🔍 大股东/高管减持公告\n"
            "- 🔍 业绩预告/快报（是否低于预期）\n"
            "- 🔍 监管函、问询函、立案调查\n"
            "- 🔍 退市风险警示（*ST / ST）\n"
            "- 🔍 大宗交易异常（折价率过高）\n"
            "- 🔍 行业政策变动（如集采、反垄断）\n"
            "- 🔍 北向资金大幅流出"
        ),
        "en": (
            "- 🔍 Major shareholder / insider selling announcements\n"
            "- 🔍 Earnings guidance / flash report (below expectations?)\n"
            "- 🔍 Regulatory letters, inquiry notices, investigations\n"
            "- 🔍 Delisting risk warnings (*ST / ST)\n"
            "- 🔍 Abnormal block trades (deep discount)\n"
            "- 🔍 Industry policy shifts (centralized procurement, antitrust)\n"
            "- 🔍 Significant Northbound capital outflow"
        ),
    },
    "hk": {
        "zh": (
            "- 🔍 董事/大股东权益变动（HKEX 披露）\n"
            "- 🔍 盈利预警公告\n"
            "- 🔍 沽空比率骤升\n"
            "- 🔍 做空机构报告\n"
            "- 🔍 SFC（证监会）警告或调查\n"
            "- 🔍 停牌/复牌风险\n"
            "- 🔍 南向资金大幅流出\n"
            "- 🔍 港币汇率异常波动 / 联系汇率压力"
        ),
        "en": (
            "- 🔍 Director / major shareholder interest changes (HKEX disclosure)\n"
            "- 🔍 Profit warning announcements\n"
            "- 🔍 Short-selling ratio spike\n"
            "- 🔍 Short-seller research reports\n"
            "- 🔍 SFC warnings or investigations\n"
            "- 🔍 Suspension / resumption risk\n"
            "- 🔍 Significant Southbound capital outflow\n"
            "- 🔍 HKD FX volatility / peg pressure"
        ),
    },
    "us": {
        "zh": (
            "- 🔍 SEC Form 4（内部人交易）/ 13D 变动\n"
            "- 🔍 财报不及预期 / guidance 下修\n"
            "- 🔍 做空报告（Hindenburg、Muddy Waters 等）\n"
            "- 🔍 SEC 调查 / 集体诉讼\n"
            "- 🔍 13F 机构持仓大幅减少\n"
            "- 🔍 关税/出口管制/制裁（若有中国业务敞口）\n"
            "- 🔍 美联储政策转向 / 美债收益率飙升"
        ),
        "en": (
            "- 🔍 SEC Form 4 (insider trading) / 13D changes\n"
            "- 🔍 Earnings miss / guidance cut\n"
            "- 🔍 Short-seller reports (Hindenburg, Muddy Waters, etc.)\n"
            "- 🔍 SEC investigation / class-action lawsuits\n"
            "- 🔍 Significant institutional 13F position reduction\n"
            "- 🔍 Tariffs / export controls / sanctions (if China exposure)\n"
            "- 🔍 Fed policy shift / US Treasury yield spike"
        ),
    },
}

# -- Market-specific tool guidance for the agent workflow --

_MARKET_TOOL_GUIDANCE = {
    "cn": {
        "zh": (
            "**第一阶段 · 行情与K线**（首先执行）\n"
            "- `get_realtime_quote` 获取实时行情\n"
            "- `get_daily_history` 获取历史K线\n\n"
            "**第二阶段 · 技术与基本面**（等第一阶段结果返回后执行）\n"
            "- `analyze_trend` 获取技术指标（MA/MACD/RSI/均线排列）\n"
            "- `get_stock_info` 获取基本面信息（估值、业绩、板块）\n"
            "- `get_chip_distribution` 获取筹码分布（可选，若失败则跳过）\n"
            "- `get_capital_flow` 获取资金流向（北向资金、主力流向）\n\n"
            "**第三阶段 · 情报搜索**（等前两阶段完成后执行）\n"
            "- `search_stock_news` 搜索最新资讯、减持公告、业绩预告、监管函等风险信号\n"
            "- `search_comprehensive_intel` 搜索多维度综合情报（可选，用于深度分析）\n\n"
            "**第四阶段 · 生成报告**（所有数据就绪后输出）"
        ),
        "en": (
            "**Phase 1 · Quote & K-lines** (execute first)\n"
            "- `get_realtime_quote` fetch real-time quote\n"
            "- `get_daily_history` fetch historical K-lines\n\n"
            "**Phase 2 · Technicals & Fundamentals** (after Phase 1 returns)\n"
            "- `analyze_trend` technical indicators (MA/MACD/RSI/alignment)\n"
            "- `get_stock_info` fundamentals (valuation, earnings, sector)\n"
            "- `get_chip_distribution` chip distribution (optional, skip on failure)\n"
            "- `get_capital_flow` capital flow (Northbound, main force)\n\n"
            "**Phase 3 · Intelligence** (after Phase 1-2 complete)\n"
            "- `search_stock_news` latest news, insider selling, earnings alerts, regulatory risks\n"
            "- `search_comprehensive_intel` multi-dimension intel (optional, for deep analysis)\n\n"
            "**Phase 4 · Generate Report** (after all data ready)"
        ),
    },
    "hk": {
        "zh": (
            "**第一阶段 · 行情与K线**（首先执行）\n"
            "- `get_realtime_quote` 获取实时行情\n"
            "- `get_daily_history` 获取历史K线\n\n"
            "**第二阶段 · 技术与基本面**（等第一阶段结果返回后执行）\n"
            "- `analyze_trend` 获取技术指标（重点关注 MA50/MA200 长周期均线）\n"
            "- `get_stock_info` 获取基本面信息（估值、业绩、板块）\n"
            "- `get_capital_flow` 获取资金流向（南向资金流向）\n\n"
            "**第三阶段 · 情报搜索**（等前两阶段完成后执行）\n"
            "- `search_stock_news` 搜索最新资讯、盈利预警、沽空信号等\n"
            "- `search_comprehensive_intel` 搜索多维度综合情报（可选，用于深度分析）\n\n"
            "**第四阶段 · 生成报告**（所有数据就绪后输出）"
        ),
        "en": (
            "**Phase 1 · Quote & K-lines** (execute first)\n"
            "- `get_realtime_quote` fetch real-time quote\n"
            "- `get_daily_history` fetch historical K-lines\n\n"
            "**Phase 2 · Technicals & Fundamentals** (after Phase 1 returns)\n"
            "- `analyze_trend` technical indicators (focus on MA50/MA200 long-cycle MAs)\n"
            "- `get_stock_info` fundamentals (valuation, earnings, sector)\n"
            "- `get_capital_flow` capital flow (Southbound flow)\n\n"
            "**Phase 3 · Intelligence** (after Phase 1-2 complete)\n"
            "- `search_stock_news` latest news, profit warnings, short-selling signals\n"
            "- `search_comprehensive_intel` multi-dimension intel (optional, for deep analysis)\n\n"
            "**Phase 4 · Generate Report** (after all data ready)"
        ),
    },
    "us": {
        "zh": (
            "**第一阶段 · 行情与K线**（首先执行）\n"
            "- `get_realtime_quote` 获取实时行情\n"
            "- `get_daily_history` 获取历史K线\n\n"
            "**第二阶段 · 技术与基本面**（等第一阶段结果返回后执行）\n"
            "- `analyze_trend` 获取技术指标（重点关注 MA50/MA200 机构标准均线）\n"
            "- `get_stock_info` 获取基本面信息（估值、业绩增长、板块）\n\n"
            "**第三阶段 · 情报搜索**（等前两阶段完成后执行）\n"
            "- `search_stock_news` 搜索最新资讯、做空报告、SEC 动态等风险信号\n"
            "- `search_comprehensive_intel` 搜索多维度综合情报（含 SEC 披露、X 社交信号等）\n\n"
            "**第四阶段 · 生成报告**（所有数据就绪后输出）"
        ),
        "en": (
            "**Phase 1 · Quote & K-lines** (execute first)\n"
            "- `get_realtime_quote` fetch real-time quote\n"
            "- `get_daily_history` fetch historical K-lines\n\n"
            "**Phase 2 · Technicals & Fundamentals** (after Phase 1 returns)\n"
            "- `analyze_trend` technical indicators (focus on MA50/MA200 institutional MAs)\n"
            "- `get_stock_info` fundamentals (valuation, earnings growth, sector)\n\n"
            "**Phase 3 · Intelligence** (after Phase 1-2 complete)\n"
            "- `search_stock_news` latest news, short-seller reports, SEC activity\n"
            "- `search_comprehensive_intel` multi-dimension intel (SEC filings, X social signals, etc.)\n\n"
            "**Phase 4 · Generate Report** (after all data ready)"
        ),
    },
}


def get_market_role(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific role description for LLM prompt.

    Args:
        stock_code: The stock code being analyzed.
        lang: 'zh' or 'en'.

    Returns:
        Role string like 'A 股投资分析' or 'US stock investment analysis'.
    """
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    return _MARKET_ROLES.get(market, _MARKET_ROLES["cn"])[lang_key]


def get_market_guidelines(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific analysis guidelines for LLM prompt.

    Args:
        stock_code: The stock code being analyzed.
        lang: 'zh' or 'en'.

    Returns:
        Multi-line string with market-specific guidelines.
    """
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    return _MARKET_GUIDELINES.get(market, _MARKET_GUIDELINES["cn"])[lang_key]


def get_market_scoring(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific scoring criteria for LLM prompt."""
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    return _MARKET_SCORING.get(market, _MARKET_SCORING["cn"])[lang_key]


def get_market_risk_checklist(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific risk checklist for LLM prompt."""
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    return _MARKET_RISK_CHECKLIST.get(market, _MARKET_RISK_CHECKLIST["cn"])[lang_key]


def get_market_tool_guidance(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific tool workflow guidance for LLM prompt."""
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    return _MARKET_TOOL_GUIDANCE.get(market, _MARKET_TOOL_GUIDANCE["cn"])[lang_key]
