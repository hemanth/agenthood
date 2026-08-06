---
name: risk-management
description: Use when evaluating position sizing, portfolio concentration, stop-loss placement, drawdown risk, or when the user asks "should I buy more" or "how much should I invest".
---

# Risk Management Skill

## Position Sizing Rules

### The 1% Rule (default for agenthood)
- Never risk more than **1% of portfolio value** on a single trade.
- Risk = entry price - stop-loss price × quantity
- Example: $10,000 portfolio → max risk per trade = $100
  - AAPL at $300, stop at $290 → max 10 shares ($3,000 position)

### Portfolio Concentration Limits
- **Single stock**: max 10% of portfolio
- **Single sector**: max 25% of portfolio
- **Options/speculative**: max 5% of portfolio
- Flag if user is overconcentrated before recommending adds.

## Stop-Loss Placement

| Strategy | Stop Level |
|----------|-----------|
| Momentum trade | Below recent swing low |
| Breakout trade | Below breakout level |
| Mean reversion | 1.5× ATR below entry |
| Options | Exit at 50% loss of premium |

Always calculate ATR when placing stops:
```
get_equity_historicals → compute daily ranges → average = ATR
Stop = entry - (1.5 × ATR)
```

## Pre-Trade Checklist

Before recommending any order, verify:
- [ ] Position size within 1% risk rule
- [ ] Sector not already overweight
- [ ] Earnings not within 2 weeks (if directional bet)
- [ ] RSI not in extreme zone against the trade direction
- [ ] Buying power sufficient (check get_portfolio)
- [ ] Dry run mode is off if placing real order

## Portfolio Health Check Workflow
```
1. get_equity_positions           → current holdings
2. get_portfolio                  → total value + cash
3. For each position:
   - calculate % of portfolio
   - get RSI → flag overbought longs
4. Summarize: concentration, total risk exposure, cash %
5. Flag anything > 10% single stock or > 25% sector
```

## Output Format
```
⚖️ Risk Assessment: AAPL 10 shares @ $309
  Position size  : $3,090 (30.9% of $10k portfolio) ⚠️ HIGH
  Risk (to stop) : $190  (1.9% of portfolio)
  Recommended    : Reduce to 5 shares ($1,545 = 15%)
  Stop-loss      : $290.00 (1× ATR below entry)
  Max loss       : $95 (0.95% of portfolio) ✅
```

## Risk Warnings — Always Emit
- ⚠️ **Earnings within 14 days**: high volatility risk
- ⚠️ **Position > 10% portfolio**: concentration risk
- ⚠️ **RSI > 75**: overbought, stop should be tight
- ⚠️ **Options expiry < 7 days**: theta decay accelerates
- ⚠️ **Penny stock (< $5)**: high spread, manipulation risk
