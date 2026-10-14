"""Execution bodies for the dot-commands that hit MCP directly.

These bypass the LLM entirely: fetch, format, render.  Each returns the text to
render (or None when it already reported an error through *td*).
"""
from __future__ import annotations

import inspect
import json
import time

from rich.markup import escape

from .format import _format_mcp_response
from .render import _strip_emoji, render_response
from .theme import C_ERR, console
from .thinking import ThinkingDisplay
from .tickers import _resolve_ticker


async def pick_account_number(agent) -> str:
    """Return the account number to operate on (cached on agent when available)."""
    getter = getattr(agent, "_get_account_number", None)
    if inspect.iscoroutinefunction(getter):
        try:
            return await getter()
        except ValueError:
            return ""
    acct_raw = await agent._call_tool_direct("robinhood_get_accounts", {})
    acct_data = json.loads(acct_raw) if isinstance(acct_raw, str) else acct_raw
    accounts = (acct_data.get("data") or {}).get("accounts") or acct_data.get("accounts") or []

    for a in accounts:
        if a.get("agentic_allowed"):
            return a.get("account_number", "")
    for a in accounts:
        if a.get("is_default"):
            return a.get("account_number", "")
    if accounts:
        return accounts[0].get("account_number", "")
    return ""


async def run_account_view(agent, which: str) -> None:
    """Render .portfolio or .positions.  *which* is the __sentinel__ or short string."""
    t0 = time.monotonic()
    response = None
    is_port = which in ("__portfolio__", "portfolio")
    label = "portfolio" if is_port else "positions"
    async with ThinkingDisplay(console) as td:
        try:
            acct_num = await pick_account_number(agent)

            if is_port:
                td.step("fetching portfolio...")
                raw = await agent._call_tool_direct(
                    "robinhood_get_portfolio", {"account_number": acct_num}
                )
                response = _format_mcp_response(raw, "Portfolio")
            elif which in ("__positions__", "positions"):
                td.step("fetching positions...")
                raw = await agent._call_tool_direct(
                    "robinhood_get_equity_positions",
                    {"account_number": acct_num}
                )
                response = _format_mcp_response(raw, "Positions")
            td.step("done")
        except Exception as exc:
            response = f"Error: {exc}"
            td.step(f"error: {str(exc)[:60]}")
    if response:
        rec = getattr(agent, "record_turn", None)
        if callable(rec):
            rec(f".{label}", response)
        render_response(response, time.monotonic() - t0)


async def run_quote(agent, ticker_spec: str) -> None:
    """Render .quote TICKER [, TICKER2 ...]."""
    import re
    t0 = time.monotonic()
    raw_tokens = [t.strip() for t in re.split(r"[,\s]+", ticker_spec) if t.strip()]
    tickers = [await _resolve_ticker(agent, sym) for sym in raw_tokens]
    label = ", ".join(tickers)
    response = None
    async with ThinkingDisplay(console) as td:
        td.step(f"fetching {label} quote...")
        try:
            raw = await agent._call_tool_direct(
                "robinhood_get_equity_quotes", {"symbols": tickers}
            )
            response = _format_mcp_response(raw, f"{label} Quote")
            td.step("done")
        except Exception as exc:
            response = f"Error: {exc}"
            td.step(f"error: {str(exc)[:60]}")
    if response:
        rec = getattr(agent, "record_turn", None)
        if callable(rec):
            rec(f".quote {label}", response, tickers=tickers)
        render_response(response, time.monotonic() - t0)


async def run_analyze(agent, tickers_raw: list[str]) -> None:
    """Render .analyze TICKER [, TICKER2 ...] -- one panel per ticker."""
    # Resolve company names to symbols
    tickers = [await _resolve_ticker(agent, sym) for sym in tickers_raw]

    for ticker in tickers:
        t0 = time.monotonic()
        response = None
        async with ThinkingDisplay(console) as td:
            td.step(f"analyzing {ticker}...")
            try:
                _buf: list[str] = []
                _first_token = [True]

                def _on_token(token: str):
                    _buf.append(token)
                    if _first_token[0]:
                        _first_token[0] = False
                        td.step("generating analysis...")

                result = await agent.analyze(
                    ticker,
                    _progress=td.step,
                    _on_token=_on_token,
                )
                response = "".join(_buf) if _buf else result
                td.step("done")
            except Exception as exc:
                response = None
                console.print(f"  [{C_ERR}]{ticker}: {escape(str(exc))}[/{C_ERR}]")
        if response:
            render_response(_strip_emoji(response), time.monotonic() - t0)


async def run_discovery(agent, which: str) -> None:
    """Render .trending or .movers directly via Robinhood watchlists + quotes."""
    t0 = time.monotonic()
    response = None
    disc_syms: list[str] = []
    prefer_movers = which == "__movers__"
    prompt = "daily movers" if prefer_movers else "trending stocks"
    label = "daily movers" if prefer_movers else "trending stocks"
    async with ThinkingDisplay(console) as td:
        td.step(f"fetching {label}...")
        try:
            fetcher = getattr(agent, "_fetch_discovery_movers", None)
            if inspect.iscoroutinefunction(fetcher):
                res = await fetcher(prefer_movers=prefer_movers, _progress=td.step)
                if isinstance(res, tuple):
                    response, disc_syms = res[0], (res[1] or [])
                else:
                    response = str(res)
            else:
                response = await agent.chat(prompt, _progress=td.step)
            td.step("done")
        except Exception as exc:
            response = f"Error: {exc}"
            td.step(f"error: {str(exc)[:60]}")
    if response:
        rec = getattr(agent, "record_turn", None)
        if callable(rec):
            rec(f".{label.split()[0]}", response, tickers=disc_syms[:8] if disc_syms else None)
        render_response(_strip_emoji(response), time.monotonic() - t0)


async def run_risk(agent, sym_raw: str, stop_price: float | None = None) -> None:
    """Render .risk TICKER [STOP_PRICE] with fixed-fractional 1%/2% sizing and ATR stop."""
    from datetime import datetime, timedelta, timezone
    from ..core.quant import compute_position_size, summarize_historicals_quant

    t0 = time.monotonic()
    sym = await _resolve_ticker(agent, sym_raw)
    response = None
    async with ThinkingDisplay(console) as td:
        td.step(f"computing risk & position sizing for {sym}...")
        try:
            acct_num = await pick_account_number(agent)
            equity = 10000.0
            if acct_num:
                try:
                    port_raw = await agent._call_tool_direct(
                        "robinhood_get_portfolio", {"account_number": acct_num}
                    )
                    port_data = json.loads(port_raw) if isinstance(port_raw, str) else port_raw
                    inner = (port_data.get("data") or port_data) if isinstance(port_data, dict) else {}
                    eq_val = inner.get("total_value") or inner.get("equity_value")
                    if eq_val is not None:
                        equity = float(eq_val)
                except Exception:
                    pass

            q_raw = await agent._call_tool_direct("robinhood_get_equity_quotes", {"symbols": [sym]})
            q_data = json.loads(q_raw) if isinstance(q_raw, str) else q_raw
            q_inner = (q_data.get("data") or q_data) if isinstance(q_data, dict) else {}
            q_results = q_inner.get("results", [q_inner]) if isinstance(q_inner, dict) else []
            first_q = q_results[0].get("quote", q_results[0]) if q_results and isinstance(q_results[0], dict) else {}
            entry_price = float(first_q.get("last_trade_price") or 0.0)

            quant: dict = {}
            try:
                start = (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
                h_raw = await agent._call_tool_direct(
                    "robinhood_get_equity_historicals",
                    {"symbols": [sym], "interval": "day", "start_time": start},
                )
                h_data = json.loads(h_raw) if isinstance(h_raw, str) else h_raw
                quant = summarize_historicals_quant(h_data)
            except Exception:
                pass

            if stop_price is None or stop_price <= 0:
                stop_price = quant.get("suggested_2atr_stop") or round(entry_price * 0.95, 2)

            max_ord = getattr(getattr(agent, "config", None), "max_order_value", None)
            plan = compute_position_size(
                equity=equity,
                entry_price=entry_price,
                stop_price=stop_price,
                risk_pct=0.01,
                max_order_value=max_ord,
            )
            link = f"[{sym}](https://robinhood.com/stocks/{sym})"
            lines = [
                f"## {sym} Position Sizing & Risk Plan",
                "| Metric | Value |",
                "|---|---|",
                f"| Symbol | **{link}** |",
                f"| Portfolio Equity | ${plan['equity']:,.2f} |",
                f"| Entry Price | ${plan['entry_price']:,.2f} |",
                f"| Stop Price ({plan['stop_distance_pct']:.2f}% risk) | ${plan['stop_price']:,.2f} (${plan['per_share_risk']:,.2f}/sh) |",
                f"| **1% Risk Size** (${plan['risk_dollars']:,.2f} max loss) | **{plan['shares_1pct']} shares** (${plan['notional_1pct']:,.2f}) |",
                f"| **2% Risk Size** (${plan['risk_dollars'] * 2:,.2f} max loss) | **{plan['shares_2pct']} shares** (${plan['notional_2pct']:,.2f}) |",
                f"| 2R / 3R Profit Targets | ${plan['target_2r']:,.2f} / ${plan['target_3r']:,.2f} |",
            ]
            if quant.get("rsi_14") is not None:
                lines.append(f"| 14-Day Wilder RSI | {quant['rsi_14']} ({quant.get('rsi_regime', '')}) |")
            if quant.get("annualized_volatility_pct") is not None:
                lines.append(f"| 20-Day Realized Volatility | {quant['annualized_volatility_pct']}% annualized |")
            if quant.get("max_drawdown_30d_pct") is not None:
                lines.append(f"| 30-Day Max Drawdown | {quant['max_drawdown_30d_pct']}% |")
            if quant.get("sparkline_30d"):
                lines.append(f"| 30-Day Price Trend | `{quant['sparkline_30d']}` |")
            response = "\n".join(lines)
            td.step("done")
        except Exception as exc:
            response = f"Error: {exc}"
            td.step(f"error: {str(exc)[:60]}")
    if response:
        rec = getattr(agent, "record_turn", None)
        if callable(rec):
            rec(f".risk {sym}", response, tickers=[sym])
        render_response(_strip_emoji(response), time.monotonic() - t0)


