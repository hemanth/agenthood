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
    """Render .portfolio or .positions.  *which* is the __sentinel__ string."""
    t0 = time.monotonic()
    response = None
    async with ThinkingDisplay(console) as td:
        try:
            acct_num = await pick_account_number(agent)

            if which == "__portfolio__":
                td.step("fetching portfolio...")
                raw = await agent._call_tool_direct(
                    "robinhood_get_portfolio", {"account_number": acct_num}
                )
                response = _format_mcp_response(raw, "Portfolio")
            elif which == "__positions__":
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
        render_response(response, time.monotonic() - t0)


async def run_quote(agent, ticker: str) -> None:
    """Render .quote TICKER."""
    t0 = time.monotonic()
    async with ThinkingDisplay(console) as td:
        td.step(f"fetching {ticker} quote...")
        try:
            raw = await agent._call_tool_direct(
                "robinhood_get_equity_quotes", {"symbols": [ticker]}
            )
            response = _format_mcp_response(raw, f"{ticker} Quote")
            td.step("done")
        except Exception as exc:
            response = f"Error: {exc}"
            td.step(f"error: {str(exc)[:60]}")
    if response:
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
                    total = sum(len(t) for t in _buf)
                    if _first_token[0]:
                        _first_token[0] = False
                        td.step(f"generating... {total} chars")
                    else:
                        td.update(f"generating... {total} chars")

                await agent.analyze(
                    ticker,
                    _progress=td.step,
                    _on_token=_on_token,
                )
                if _buf:
                    response = "".join(_buf)
                td.step("done")
            except Exception as exc:
                response = None
                console.print(f"  [{C_ERR}]{ticker}: {escape(str(exc))}[/{C_ERR}]")
        if response:
            render_response(_strip_emoji(response), time.monotonic() - t0)
