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
        lines = [
            f"## {title}",
            "| # | Symbol | Qty | Avg Cost | Current | P/L |",
            "|---|---|---|---|---|---|",
        ]
        for idx, pos in enumerate(results, start=1):
            sym = pos.get("symbol", "?")
            link = f"[{sym}](https://robinhood.com/stocks/{sym})" if sym != "?" else sym
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
            lines.append(f"| {idx} | **{link}** | {qty} | {avg} | {cur} | {pl_str} |")
        return "\n".join(lines)

    # Quote response
    if "Quote" in title and isinstance(data, dict):
        results = data.get("results", [data])
        if not results:
            return f"## {title}\n\nNo data."

        def _extract_q(entry):
            q = entry.get("quote", entry) if isinstance(entry, dict) else {}
            sym = str(q.get("symbol", "?"))
            is_crypto = "mark_price" in q or (len(sym) > 3 and sym.endswith("USD") and not sym.endswith("-USD"))
            if len(sym) > 3 and sym.endswith("USD") and not sym.endswith("-USD"):
                sym = sym[:-3]
            elif sym.endswith("-USD"):
                sym = sym[:-4]
                is_crypto = True

            last_raw = q.get("last_trade_price") or q.get("mark_price")
            prev_raw = q.get("adjusted_previous_close") or q.get("previous_close") or q.get("open_price")
            bid_raw = q.get("bid_price")
            ask_raw = q.get("ask_price")
            ext_raw = q.get("last_extended_hours_trade_price")

            try:
                cur = float(last_raw) if last_raw is not None else None
            except (ValueError, TypeError):
                cur = None
            try:
                prev = float(prev_raw) if prev_raw is not None else None
            except (ValueError, TypeError):
                prev = None

            chg_str = "N/A"
            verdict = "HOLD"
            if cur is not None and prev and prev > 0:
                chg = cur - prev
                pct = (chg / prev) * 100
                sign = "+" if chg >= 0 else "-"
                dec = 4 if 0 < abs(cur) < 1.0 else 2
                chg_str = f"{sign}${abs(chg):,.{dec}f} ({'+' if chg >= 0 else ''}{pct:.2f}%)"
                if pct > 1.5:
                    verdict = "BUY"
                elif pct < -1.5:
                    verdict = "SELL"

            def _fmt_p(v):
                if v is None:
                    return "-"
                try:
                    f = float(v)
                    if 0 < abs(f) < 0.01:
                        return f"${f:,.6f}"
                    if 0 < abs(f) < 1.0:
                        return f"${f:,.4f}"
                    return f"${f:,.2f}"
                except (ValueError, TypeError):
                    return str(v)

            return {
                "symbol": sym,
                "is_crypto": is_crypto,
                "last": _fmt_p(last_raw),
                "prev": _fmt_p(prev_raw),
                "bid": _fmt_p(bid_raw),
                "ask": _fmt_p(ask_raw),
                "ext": _fmt_p(ext_raw) if ext_raw is not None else None,
                "change": chg_str,
                "verdict": verdict,
            }

        parsed_quotes = [_extract_q(r) for r in results if isinstance(r, dict)]
        if not parsed_quotes:
            return f"## {title}\n\nNo data."

        def _sym_link(pq: dict) -> str:
            sym = pq["symbol"]
            if sym == "?":
                return sym
            path = "crypto" if pq.get("is_crypto") else "stocks"
            return f"[{sym}](https://robinhood.com/{path}/{sym})"

        # Multi-ticker side-by-side table
        if len(parsed_quotes) > 1:
            lines = [
                f"## {title}",
                "| # | Symbol | Price | Day Change | Bid / Ask | Prev Close | Verdict |",
                "|---|---|---|---|---|---|---|",
            ]
            for idx, pq in enumerate(parsed_quotes, start=1):
                link = _sym_link(pq)
                lines.append(
                    f"| {idx} | **{link}** | {pq['last']} | {pq['change']} | "
                    f"{pq['bid']} / {pq['ask']} | {pq['prev']} | **{pq['verdict']}** |"
                )
            lines.append("\n*Tip: type `1`..`N` for instant quote or `.a 1` to analyze row #1.*")
            return "\n".join(lines)

        # Single-ticker detailed table
        pq = parsed_quotes[0]
        link = _sym_link(pq)
        lines = [
            f"## {title}",
            "| Metric | Value |",
            "|---|---|",
            f"| Symbol | **{link}** |",
            f"| Last Trade Price | {pq['last']} |",
        ]
        if pq["change"] != "N/A":
            lines.append(f"| Day Change | {pq['change']} |")
        if pq["prev"] != "-":
            lines.append(f"| Previous Close | {pq['prev']} |")
        if pq["bid"] != "-":
            lines.append(f"| Bid Price | {pq['bid']} |")
        if pq["ask"] != "-":
            lines.append(f"| Ask Price | {pq['ask']} |")
        if pq["ext"]:
            lines.append(f"| Extended Hours | {pq['ext']} |")
        lines.append(f"| Verdict | **{pq['verdict']}** |")
        return "\n".join(lines)

    # Fallback: dump as readable text
    return f"## {title}\n\n{json.dumps(data, indent=2)[:2000]}"

