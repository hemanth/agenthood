"""Resolve company names typed by the user into ticker symbols."""
from __future__ import annotations

# Common company name -> ticker mappings
_NAME_TO_TICKER: dict[str, str] = {
    "APPLE": "AAPL", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "MICROSOFT": "MSFT", "AMAZON": "AMZN", "META": "META",
    "FACEBOOK": "META", "TESLA": "TSLA", "NVIDIA": "NVDA",
    "NETFLIX": "NFLX", "PAYPAL": "PYPL", "AMD": "AMD",
    "INTEL": "INTC", "DISNEY": "DIS", "WALMART": "WMT",
    "JPMORGAN": "JPM", "BERKSHIRE": "BRK.B", "VISA": "V",
    "MASTERCARD": "MA", "SALESFORCE": "CRM", "UBER": "UBER",
    "SPOTIFY": "SPOT", "AIRBNB": "ABNB", "COINBASE": "COIN",
    "ROBINHOOD": "HOOD", "SNAP": "SNAP", "PINTEREST": "PINS",
    "PALANTIR": "PLTR", "SNOWFLAKE": "SNOW", "CROWDSTRIKE": "CRWD",
    "SHOPIFY": "SHOP", "SQUARE": "XYZ", "BLOCK": "XYZ",
}


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
