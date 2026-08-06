---
name: technical-analysis
description: Use when analyzing stock price action, identifying trends, interpreting RSI/MACD/Bollinger signals, or making buy/sell timing decisions based on technical indicators.
---

# Technical Analysis Skill

## Indicator Interpretation

### RSI (Relative Strength Index, 14-period default)
- **< 30** → Oversold. Potential reversal buying opportunity. Confirm with volume spike.
- **30–50** → Bearish momentum. Avoid new longs unless trend reversing.
- **50–70** → Neutral-to-bullish. Trend following zone.
- **> 70** → Overbought. Consider tightening stops or taking profits.
- **Divergence**: Price makes new high but RSI doesn't → bearish divergence (strong sell signal).

### MACD
- **Signal line crossover above** → Bullish momentum building. Entry signal.
- **Signal line crossover below** → Bearish. Consider exit or short.
- **Histogram expanding** → Trend strengthening.
- **Histogram contracting** → Trend weakening, watch for reversal.
- Always pair with price action — MACD lags.

### Bollinger Bands
- **Price touches upper band** → Overbought short-term. Possible mean reversion.
- **Price touches lower band** → Oversold. Possible bounce.
- **Band squeeze (narrow bands)** → Volatility contraction → breakout incoming (direction unknown).
- **Band expansion** → Trend acceleration.

## Workflow: Full Technical Read

```
1. get_equity_quotes          → current price, daily change
2. get_equity_technical_indicators(symbol, "rsi")  → RSI
3. get_equity_technical_indicators(symbol, "macd") → MACD + signal
4. get_equity_historicals(symbol, interval="day", span="month") → 30d trend
5. Synthesize: trend direction + momentum + overbought/oversold
6. State: BULLISH / BEARISH / NEUTRAL with confidence level
7. If actionable: suggest entry price, stop-loss, target
```

## Output Format

Always present a structured summary:

| Signal | Value | Interpretation |
|--------|-------|----------------|
| Price  | $X.XX | +/-X% today    |
| RSI    | XX    | Neutral/OB/OS  |
| MACD   | Cross | Bullish/Bearish |
| Trend  | 30d   | Up/Down/Sideways |

**Verdict**: BULLISH / BEARISH / NEUTRAL  
**Entry**: $X.XX | **Stop**: $X.XX | **Target**: $X.XX

## Rules
- Never make a recommendation without checking at least RSI + price trend.
- Always note if market is closed and data is from the previous session.
- Flag earnings within 2 weeks as high-risk — avoid new positions before earnings.
