---
name: options-playbook
description: Use when the user asks about options — buying calls/puts, covered calls, spreads, earnings plays, IV crush, or anything involving options chains, greeks, or expiration strategy.
---

# Options Playbook Skill

## Before Any Options Trade

Always gather:
```
1. get_option_chains(symbol)              → available expirations + strikes
2. get_option_instruments(...)            → specific contracts
3. get_option_quotes(...)                 → bid/ask, IV, greeks
4. get_equity_fundamentals(symbol)        → earnings date (IV crush risk!)
5. get_equity_technical_indicators(rsi)   → entry timing
```

**Always show**: breakeven, max profit, max loss, IV rank.

## Common Strategies

### Long Call (bullish, defined risk)
- Buy when: RSI < 60, bullish catalyst expected, IV rank < 40%
- Strike: ATM or 1 strike OTM
- Expiry: 30–60 DTE (avoid 0DTE for beginners)
- Max loss: premium paid | Max profit: unlimited
- Breakeven: strike + premium
- Exit: at 50% profit or 21 DTE (whichever first)

### Long Put (bearish, defined risk)
- Buy when: RSI > 65, distribution pattern, high IV is fine (protective)
- Strike: ATM or 1 strike OTM
- Expiry: 30–45 DTE
- Max loss: premium paid | Max profit: strike - premium (near zero)

### Covered Call (neutral-to-slightly-bullish, income)
- Requires 100 shares of underlying
- Sell OTM call (delta 0.20–0.30) against shares
- Collect premium as income
- Risk: shares called away if price rises above strike
- Best in: sideways or slowly rising market

### Cash-Secured Put (neutral-to-bullish, income)
- Sell OTM put, hold cash to buy shares if assigned
- Collect premium | Effective entry: strike - premium
- Risk: must buy shares at strike if price drops

## ⚠️ Earnings Warning
If earnings within **14 days**:
- IV is inflated → options are expensive
- IV crush after earnings can destroy long option value even if direction is right
- Use spreads (defined risk) not naked longs
- State this warning prominently before any earnings play

## Greeks Quick Reference
| Greek | Meaning | Watch for |
|-------|---------|-----------|
| Delta | Price sensitivity | 0.5 = ATM |
| Theta | Time decay per day | Hurts buyers |
| Vega | IV sensitivity | Hurts on IV crush |
| Gamma | Delta acceleration | High near expiry |

## Output Format for Options
```
📊 AAPL $200C 2026-09-19
  Bid/Ask : $3.20 / $3.40  (mid: $3.30)
  IV      : 42%  (IV Rank: 38% — fair)
  Delta   : 0.48 | Theta: -$0.08/day
  Breakeven: $203.30 (+1.6% from current)
  Max Loss: $330 (premium paid)
⚠️  Earnings: 2026-08-01 — IV crush risk after report
```
