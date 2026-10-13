"""Format raw Robinhood JSON into clean markdown -- no LLM call needed.

Used as a presentation fallback when the model returns raw tool JSON instead of
formatting it itself.
"""
from __future__ import annotations


def _format_tool_result(tool_name: str, raw_json: str) -> str | None:
    """Format raw Robinhood JSON into clean markdown. Returns None if unrecognized."""
    import json as _json
    from datetime import datetime, timezone
    try:
        data = _json.loads(raw_json)
    except (_json.JSONDecodeError, TypeError):
        return None

    results = data.get("results") if isinstance(data, dict) else None
    if results is None and isinstance(data, list):
        results = data
    first = results[0] if (results and isinstance(results, list)) else {}

    def _price(v: str | None) -> str:
        """Clean price: $58.53 not $58.530000."""
        if not v:
            return "-"
        try:
            return f"${float(v):,.2f}"
        except (ValueError, TypeError):
            return str(v)

    def _time(ts: str | None) -> str:
        """Humanize timestamp."""
        if not ts:
            return "-"
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            diff = now - dt
            if diff.total_seconds() < 60:
                return "just now"
            if diff.total_seconds() < 3600:
                return f"{int(diff.total_seconds() / 60)}m ago"
            if diff.total_seconds() < 86400:
                return f"{int(diff.total_seconds() / 3600)}h ago"
            return dt.strftime("%b %d, %H:%M")
        except Exception:
            return str(ts)[:16]

    # -- Quotes --
    if "quote" in first:
        parts = []
        for entry in results:
            q = entry.get("quote", {})
            sym = q.get("symbol", "?")
            cur = float(q.get("last_trade_price", 0) or 0)
            prev = float(q.get("adjusted_previous_close", 0) or 0)
            chg = cur - prev if prev else 0
            pct = (chg / prev * 100) if prev else 0
            sign = "+" if chg >= 0 else ""
            bid = q.get("bid_price")
            ask = q.get("ask_price")
            state = q.get("state", "?")
            trade_time = q.get("venue_last_trade_time")

            # Compute a preliminary signal / recommendation based on momentum
            if pct > 1.5:
                signal_fmt = "[bold green]BUY[/bold green]"
            elif pct < -1.5:
                signal_fmt = "[bold red]SELL[/bold red]"
            else:
                signal_fmt = "[bold yellow]HOLD[/bold yellow]"

            chg_color = "green" if chg >= 0 else "red"
            chg_fmt = f"[{chg_color}]{sign}{chg:.2f} ({sign}{pct:.2f}%)[/{chg_color}]"

            lines = [
                f"Ticker     : [bold cyan]{sym}[/bold cyan]",
                f"Last Price : [bold]{_price(str(cur))}[/bold] ({chg_fmt})",
                f"Verdict    : {signal_fmt}",
                f"Bid / Ask  : {_price(bid)} / {_price(ask)}",
                f"Prev Close : {_price(str(prev))}",
                f"Last Trade : {_time(trade_time)}",
                f"Status     : {state.title()}",
            ]
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # -- Portfolio --
    p_data = data.get("data", {}) if isinstance(data, dict) and "data" in data else data
    if isinstance(p_data, dict) and "total_value" in p_data:
        val = _price(p_data.get("total_value"))
        eq = _price(p_data.get("equity_value"))
        crypto = _price(p_data.get("crypto_value"))
        cash = _price(p_data.get("cash"))
        bp = _price(p_data.get("buying_power", {}).get("buying_power") if isinstance(p_data.get("buying_power"), dict) else p_data.get("buying_power"))
        lines = [
            f"Portfolio Value : [bold cyan]{val}[/bold cyan]",
            f"Equities        : [bold]{eq}[/bold]",
            f"Crypto          : [bold]{crypto}[/bold]",
            f"Cash            : [bold]{cash}[/bold]",
            f"Buying Power    : [bold green]{bp}[/bold green]"
        ]
        return "\n".join(lines)

    # -- Accounts --
    if "account_number" in first:
        lines = ["**Accounts**", "",
                 "| Account | Type | Nickname | Agentic |",
                 "|---------|------|----------|---------|"]
        for a in results:
            num = a.get("account_number", "?")
            masked = f"****{num[-4:]}" if len(num) > 4 else num
            btype = a.get("brokerage_account_type", "?").replace("_", " ").title()
            nick = a.get("nickname") or "-"
            agentic = "Yes" if a.get("agentic_allowed") else "No"
            lines.append(f"| {masked} | {btype} | {nick} | {agentic} |")
        return "\n".join(lines)

    # -- Search --
    if "symbol" in first and "name" in first:
        lines = ["| Symbol | Name |",
                 "|--------|------|"]
        for r in results:
            sym = r.get("symbol", "?")
            name = r.get("simple_name") or r.get("name", "?")
            lines.append(f"| **{sym}** | {name} |")
        return "\n".join(lines)

    return None
