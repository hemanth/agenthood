"""Resolve company names typed by the user into ticker symbols."""
from __future__ import annotations

from ..core.tickers import NAME_TO_TICKER as _NAME_TO_TICKER



async def _resolve_ticker(agent, sym: str) -> str:
    """Resolve a company name to a ticker symbol.

    1. If it looks like a ticker already (1-5 uppercase letters), return as-is.
    2. Check static name map.
    3. Fall back to Robinhood search MCP tool.
    """
    # Already a ticker (e.g. "AAPL", "PYPL")
    if len(sym) <= 5 and sym.isalpha() and sym.isupper():
        # Could still be a name like "APPLE" or "UBER" -- check map first
        if sym in _NAME_TO_TICKER:
            return _NAME_TO_TICKER[sym]
        return sym

    # Check name map
    up = sym.upper()
    if up in _NAME_TO_TICKER:
        return _NAME_TO_TICKER[up]

    # Fall back to Robinhood search
    try:
        raw = await agent._call_tool_direct("robinhood_search", {"query": sym})
        import json
        data = json.loads(raw) if isinstance(raw, str) else raw
        results = data.get("data", {}).get("results", [])
        if results:
            return results[0].get("symbol", sym)
    except Exception:
        pass

    return sym  # return as-is if all else fails
