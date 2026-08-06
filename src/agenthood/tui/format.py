"""Render raw MCP JSON as markdown tables for the dot-command paths.

Used by .portfolio / .positions / .quote, which call MCP directly and never
go near the LLM.
"""
from __future__ import annotations


def _format_mcp_response(raw: str, title: str) -> str:
    """Format raw MCP JSON response into a readable markdown table."""
    import json
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return f"## {title}\n\n{str(raw)[:2000]}"

    # Navigate into nested data
    if isinstance(data, dict):
        data = data.get("data", data)

    # Portfolio response
    if title == "Portfolio" and isinstance(data, dict):
        lines = [f"## {title}", "| Metric | Value |", "|---|---|"]
        for key in ["total_value", "equity_value", "cash", "options_value",
                     "crypto_value", "buying_power"]:
            val = data.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                if isinstance(val, dict):  # buying_power is nested
                    val = val.get("buying_power", val)
                try:
                    lines.append(f"| {label} | ${float(val):,.2f} |")
                except (ValueError, TypeError):
                    lines.append(f"| {label} | {val} |")
        return "\n".join(lines) if len(lines) > 3 else f"## {title}\n\n{str(data)[:2000]}"

    # Positions response
    if title == "Positions" and isinstance(data, dict):
        results = data.get("results", [])
        if not results:
            return f"## {title}\n\nNo open positions."
        lines = [f"## {title}", "| Symbol | Qty | Avg Cost | Current | P/L |", "|---|---|---|---|---|"]
        for pos in results:
            sym = pos.get("symbol", "?")
            qty = pos.get("quantity", "0")
            avg = pos.get("average_buy_price", "?")
            cur = pos.get("current_price", "?")
            try:
                pl = (float(cur) - float(avg)) * float(qty)
                pl_str = f"${pl:+,.2f}"
            except (ValueError, TypeError):
                pl_str = "N/A"
            try:
                avg = f"${float(avg):,.2f}"
                cur = f"${float(cur):,.2f}"
            except (ValueError, TypeError):
                pass
            lines.append(f"| {sym} | {qty} | {avg} | {cur} | {pl_str} |")
        return "\n".join(lines)

    # Quote response
    if "Quote" in title and isinstance(data, dict):
        results = data.get("results", [data])
        if not results:
            return f"## {title}\n\nNo data."
        lines = [f"## {title}", "| Metric | Value |", "|---|---|"]
        q = results[0] if isinstance(results[0], dict) else results[0]
        if isinstance(q, dict):
            q = q.get("quote", q)
        for key in ["symbol", "last_trade_price", "previous_close",
                     "bid_price", "ask_price", "last_extended_hours_trade_price"]:
            val = q.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                try:
                    if key != "symbol":
                        val = f"${float(val):,.2f}"
                except (ValueError, TypeError):
                    pass
                lines.append(f"| {label} | {val} |")
        return "\n".join(lines) if len(lines) > 3 else f"## {title}\n\n{str(data)[:2000]}"

    # Fallback: dump as readable text
    return f"## {title}\n\n{json.dumps(data, indent=2)[:2000]}"
