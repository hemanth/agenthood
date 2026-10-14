"""Pure-Python, zero-dependency quantitative analytics and risk sizing.

Computes deterministic technical indicators and position-sizing metrics directly
from Robinhood historical OHLCV bars so the LLM analyst and TUI dot-commands never
have to guess or approximate arithmetic:
  - Wilder's 14-period RSI (Wilder, 1978)
  - 20-day annualized realized volatility (log/simple return std * sqrt(252))
  - Maximum drawdown (%) over the lookback window
  - 14-day Average True Range (ATR) for volatility-adjusted stop placement
  - Unicode 8-level sparkline (▁▂▃▄▅▆▇█) for terminal visual perception
  - Fixed-fractional position sizing (1% / 2% account risk rule + R-multiples)
"""
from __future__ import annotations

import math
from typing import Any

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def compute_wilder_rsi(closes: list[float], period: int = 14) -> float | None:
    """Compute Wilder's smoothed Relative Strength Index (RSI) over *closes*."""
    if len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def compute_realized_volatility(closes: list[float], trading_days: int = 252) -> float | None:
    """Return annualized realized volatility (%) from daily close-to-close returns."""
    if len(closes) < 3:
        return None
    returns: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev > 0 and cur > 0:
            returns.append(math.log(cur / prev))
    if len(returns) < 2:
        return None
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    ann_vol = math.sqrt(variance) * math.sqrt(trading_days) * 100.0
    return round(ann_vol, 2)


def compute_max_drawdown(closes: list[float]) -> float | None:
    """Return maximum peak-to-trough drawdown as a negative percentage (e.g. -25.0)."""
    if len(closes) < 2:
        return None
    peak = closes[0]
    max_dd = 0.0
    for price in closes:
        if price > peak:
            peak = price
        if peak > 0:
            dd = ((price - peak) / peak) * 100.0
            if dd < max_dd:
                max_dd = dd
    return round(max_dd, 2)


def compute_atr(bars: list[dict[str, Any]], period: int = 14) -> float | None:
    """Compute Wilder's Average True Range (ATR) from OHLC bar dicts."""
    trs: list[float] = []
    prev_close: float | None = None
    for b in bars:
        try:
            high = float(b.get("high_price") or b.get("close_price"))
            low = float(b.get("low_price") or b.get("close_price"))
            close = float(b.get("close_price"))
        except (TypeError, ValueError):
            continue
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close

    if not trs:
        return None
    if len(trs) <= period:
        return round(sum(trs) / len(trs), 2)
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 2)


def compute_sparkline(values: list[float], width: int = 20) -> str:
    """Render a compact Unicode block sparkline (▁▂▃▄▅▆▇█) of up to *width* points."""
    if not values:
        return ""
    series = values[-width:] if len(values) > width else list(values)
    lo = min(series)
    hi = max(series)
    span = hi - lo
    if span <= 0:
        return _SPARK_CHARS[3] * len(series)
    out: list[str] = []
    for v in series:
        idx = int(round(((v - lo) / span) * (len(_SPARK_CHARS) - 1)))
        idx = max(0, min(len(_SPARK_CHARS) - 1, idx))
        out.append(_SPARK_CHARS[idx])
    return "".join(out)


def extract_bars(historicals_section: Any) -> list[dict[str, Any]]:
    """Extract the list of OHLCV bar dicts from a Robinhood historicals payload."""
    if isinstance(historicals_section, dict):
        inner = historicals_section.get("data", historicals_section)
        if isinstance(inner, dict) and "results" in inner:
            inner = inner["results"]
        if isinstance(inner, dict):
            for key in ("bars", "historicals"):
                if isinstance(inner.get(key), list):
                    return inner[key]
        historicals_section = inner
    if isinstance(historicals_section, list) and historicals_section:
        first = historicals_section[0]
        if isinstance(first, dict):
            for key in ("bars", "historicals"):
                if isinstance(first.get(key), list):
                    return first[key]
            if "close_price" in first:
                return historicals_section
    return []


def summarize_historicals_quant(historicals_section: Any) -> dict[str, Any]:
    """Compute a compact dictionary of deterministic quant metrics from historical bars."""
    bars = extract_bars(historicals_section)
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        try:
            closes.append(float(b["close_price"]))
            if "high_price" in b:
                highs.append(float(b["high_price"]))
            if "low_price" in b:
                lows.append(float(b["low_price"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not closes:
        return {}

    rsi = compute_wilder_rsi(closes, period=14)
    vol = compute_realized_volatility(closes)
    mdd = compute_max_drawdown(closes)
    atr = compute_atr(bars, period=14)
    spark = compute_sparkline(closes, width=20)
    ret_30d = round(((closes[-1] - closes[0]) / closes[0]) * 100.0, 2) if closes[0] > 0 else 0.0

    summary: dict[str, Any] = {
        "bars_count": len(closes),
        "latest_close": round(closes[-1], 2),
        "high_30d": round(max(highs) if highs else max(closes), 2),
        "low_30d": round(min(lows) if lows else min(closes), 2),
        "return_30d_pct": ret_30d,
        "sparkline_30d": spark,
    }
    if rsi is not None:
        summary["rsi_14"] = rsi
        summary["rsi_regime"] = (
            "OVERBOUGHT (>70)" if rsi >= 70.0 else ("OVERSOLD (<30)" if rsi <= 30.0 else "NEUTRAL (30-70)")
        )
    if vol is not None:
        summary["annualized_volatility_pct"] = vol
    if mdd is not None:
        summary["max_drawdown_30d_pct"] = mdd
    if atr is not None:
        summary["atr_14"] = atr
        summary["suggested_2atr_stop"] = round(closes[-1] - 2.0 * atr, 2)
    return summary


def compute_position_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
    max_order_value: float | None = None,
) -> dict[str, Any]:
    """Compute fixed-fractional position sizing (1% and 2% account risk rules) and R-multiples."""
    equity = max(float(equity), 0.0)
    entry = float(entry_price)
    stop = float(stop_price)
    per_share_risk = abs(entry - stop)
    direction = "LONG" if entry >= stop else "SHORT"

    if per_share_risk <= 0 or entry <= 0:
        return {
            "equity": equity,
            "entry_price": entry,
            "stop_price": stop,
            "direction": direction,
            "per_share_risk": 0.0,
            "risk_dollars": 0.0,
            "shares_1pct": 0,
            "notional_1pct": 0.0,
            "shares_2pct": 0,
            "notional_2pct": 0.0,
            "target_2r": entry,
            "target_3r": entry,
        }

    risk_1pct_dollars = equity * risk_pct
    risk_2pct_dollars = equity * (risk_pct * 2.0)

    shares_1pct = int(math.floor(risk_1pct_dollars / per_share_risk))
    shares_2pct = int(math.floor(risk_2pct_dollars / per_share_risk))

    # Cap by portfolio equity (100% cash) so we never suggest >1x leverage by default
    max_affordable = int(math.floor(equity / entry)) if equity > 0 else 0
    shares_1pct = min(shares_1pct, max_affordable)
    shares_2pct = min(shares_2pct, max_affordable)

    sign = 1.0 if direction == "LONG" else -1.0
    target_2r = round(entry + sign * 2.0 * per_share_risk, 2)
    target_3r = round(entry + sign * 3.0 * per_share_risk, 2)

    result: dict[str, Any] = {
        "equity": round(equity, 2),
        "entry_price": round(entry, 2),
        "stop_price": round(stop, 2),
        "direction": direction,
        "stop_distance_pct": round((per_share_risk / entry) * 100.0, 2),
        "per_share_risk": round(per_share_risk, 2),
        "risk_dollars": round(risk_1pct_dollars, 2),
        "shares_1pct": shares_1pct,
        "notional_1pct": round(shares_1pct * entry, 2),
        "shares_2pct": shares_2pct,
        "notional_2pct": round(shares_2pct * entry, 2),
        "target_2r": target_2r,
        "target_3r": target_3r,
    }
    if max_order_value and max_order_value > 0:
        result["max_order_cap_shares"] = int(math.floor(max_order_value / entry))
    return result
