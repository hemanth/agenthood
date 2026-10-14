"""Dot-command dispatch.

``handle_meta`` runs BEFORE the agent's own pipeline, so dot-commands never
reach the LLM.  Its return value tells the REPL what to do next:

  True                      -- fully handled, nothing more to do
  False                     -- not a dot-command, pass to the agent
  "__portfolio__"           -- run the portfolio action
  "__positions__"           -- run the positions action
  "__watches__"             -- show active in-REPL and daemon watches
  "__update__"              -- run the self-upgrade
  ("__quote__", "AAPL")     -- run the quote action for that ticker (or comma-joined tickers)
  ["AAPL", "NVDA"]          -- run the analyze action for those tickers
"""
from __future__ import annotations

import difflib
import re

from rich.markup import escape

from .models import _resolve_model, _short_model, apply_model_switch, show_model_picker
from .render import _show_table, print_help
from .sessions import print_sessions
from .theme import C_ACCENT, C_DIM, C_GREEN, C_STEP, C_WARN, console

KNOWN_DOT_COMMANDS = (
    ".analyze", ".a",
    ".portfolio", ".port", ".p",
    ".positions", ".pos",
    ".quote", ".q",
    ".trending", ".t",
    ".movers", ".mv",
    ".risk",
    ".watches", ".watch",
    ".history", ".status",
    ".tools", ".help",
    ".model", ".m",
    ".sessions", ".session",
    ".new", ".reset", ".clear",
    ".cost", ".memory", ".mem", ".findings",
    ".verbose", ".update", ".upgrade",
)


def _active_context_ticker(agent) -> str | None:
    """Return the most recently discussed ticker on *agent*, if any."""
    lt = getattr(agent, "last_ticker", None)
    if isinstance(lt, str) and lt.strip():
        return lt.strip().upper()
    recent = getattr(agent, "_recent_tickers", None)
    if isinstance(recent, list) and recent:
         last = recent[-1]
         if isinstance(last, str) and last.strip():
             return last.strip().upper()
    return None


def _resolve_numeric_ticker(token: str, agent) -> str:
    """If *token* is a 1-based index into the last table or recent tickers, resolve it to that symbol."""
    clean = token.lstrip(".")
    if clean.isdigit():
        idx = int(clean) - 1
        table_syms = getattr(agent, "_table_tickers", None)
        if isinstance(table_syms, list) and 0 <= idx < len(table_syms):
            sym = table_syms[idx]
            if isinstance(sym, str) and sym.strip():
                return sym.strip().upper()
        recent = getattr(agent, "_recent_tickers", None)
        if isinstance(recent, list) and 0 <= idx < len(recent):
            sym = recent[idx]
            if isinstance(sym, str) and sym.strip():
                return sym.strip().upper()
    return token.upper()


def handle_meta(cmd: str, agent) -> bool | str | tuple | list:
    stripped = cmd.strip()
    lower    = stripped.lower()

    # Numeric quick-select: bare "1".."9" or ".1"..".9" quotes the Nth recent ticker
    if re.fullmatch(r"\.?[1-9]\d?", stripped):
        resolved = _resolve_numeric_ticker(stripped, agent)
        if resolved != stripped.upper():
            return ("__quote__", resolved)
        if stripped.startswith("."):
            console.print(f"  [{C_WARN}]No ticker at index {stripped.lstrip('.')} in recent context.[/{C_WARN}]")
            return True
        return False

    if lower in (".help", ".h"):
        print_help()
        return True

    # .model [arg] / .m [arg]
    if lower in (".model", ".m") or lower.startswith(".model ") or lower.startswith(".m "):
        parts = stripped.split(None, 1)
        if len(parts) == 2:
            arg = parts[1].strip()
            new_model = _resolve_model(arg)
            if new_model:
                apply_model_switch(agent, new_model)
                console.print(f"  [{C_GREEN}]model -> {_short_model(new_model)}[/{C_GREEN}]")
            return True
        new_model = show_model_picker(agent.config.model)
        if new_model:
            apply_model_switch(agent, new_model)
        return True

    if lower in (".sessions", ".session"):
        print_sessions()
        return True

    # .update -- pip upgrade from PyPI (async, so the REPL runs it)
    if lower in (".update", ".upgrade"):
        return "__update__"

    if lower in (".new", ".reset", ".clear"):
        agent.clear_history()
        console.print(f"  [{C_DIM}]Conversation history cleared. New session started.[/{C_DIM}]")
        return True

    if lower == ".cost":
        _show_table(agent.cost(), "cost")
        return True

    if lower in (".memory", ".mem"):
        _show_table(agent.memory_stats(), "memory")
        return True

    if lower == ".findings":
        findings = agent.findings(limit=10)
        if not findings:
            console.print(f"  [{C_DIM}]No findings yet.[/{C_DIM}]")
        else:
            for f in findings:
                console.print(f"  [{C_STEP}]{escape(str(f.get('text', f)))}[/{C_STEP}]")
        return True

    if lower == ".history":
        history = getattr(agent, "_history", None) or []
        if not history:
            console.print(f"  [{C_DIM}]No conversation history in this session yet.[/{C_DIM}]")
        else:
            rows = {}
            turn_idx = 0
            for role, text in history[-12:]:
                if role == "user":
                    turn_idx += 1
                    rows[f"#{turn_idx} you"] = str(text)[:80]
                else:
                    first_line = str(text).strip().splitlines()[0] if str(text).strip() else ""
                    rows[f"#{turn_idx} agent"] = first_line[:80]
            _show_table(rows, "recent history")
        return True

    if lower == ".status":
        cfg = getattr(agent, "config", None)
        ctx_sym = _active_context_ticker(agent) or "-"
        hist_len = len(getattr(agent, "_history", None) or []) // 2
        raw_max = getattr(cfg, "max_order_value", 0) if cfg else 0
        max_str = f"${raw_max:,.0f}" if isinstance(raw_max, (int, float)) else str(raw_max)
        info = {
            "model": str(getattr(cfg, "model", "auto") or "auto") if cfg else "auto",
            "mode": "DRY-RUN" if getattr(cfg, "dry_run", True) else "LIVE",
            "max_order": max_str,
            "context_ticker": ctx_sym,
            "turns": str(hist_len),
        }
        analyze_model = getattr(cfg, "analyze_model", "") if cfg else ""
        if isinstance(analyze_model, str) and analyze_model:
            info["analyze_model"] = analyze_model
        _show_table(info, "status")
        return True


    # .watches / .watch
    if lower == ".watches" or lower == ".watch":
        return "__watches__"
    if lower.startswith(".watch "):
        # Allow `.watch AAPL below 180` as a dot-command shorthand for natural watch
        from ..watcher import parse_watch_prompt
        spec = parse_watch_prompt(
            "watch " + stripped.split(None, 1)[1],
            context_ticker=_active_context_ticker(agent),
        )
        if spec:
            return {"__watch__": spec}
        console.print(f"  [{C_WARN}]usage: .watch TICKER below|above PRICE[/{C_WARN}]")
        return True

    # .portfolio -- direct MCP call
    if lower in (".portfolio", ".port", ".p"):
        return "__portfolio__"

    # .positions -- direct MCP call
    if lower in (".positions", ".pos"):
        return "__positions__"

    # .trending / .t -- live trending stocks table
    if lower in (".trending", ".t"):
        return "__trending__"

    # .movers -- live daily movers table
    if lower in (".movers", ".mv"):
        return "__movers__"

    # .risk [TICKER] [STOP_PRICE] -- position sizing & risk calculator
    if lower == ".risk" or lower.startswith(".risk "):
        parts = stripped.split()
        if len(parts) == 1:
            ctx_sym = _active_context_ticker(agent)
            if ctx_sym:
                return ("__risk__", ctx_sym, None)
            console.print(f"  [{C_WARN}]usage: .risk TICKER [STOP_PRICE][/{C_WARN}]")
            return True
        sym = _resolve_numeric_ticker(parts[1], agent)
        stop_price = None
        if len(parts) >= 3:
            try:
                stop_price = float(parts[2].lstrip("$"))
            except ValueError:
                pass
        return ("__risk__", sym, stop_price)

    # .quote [TICKER] / .q [TICKER] -- direct MCP call
    if lower in (".quote", ".q") or lower.startswith(".quote ") or lower.startswith(".q "):
        parts = stripped.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            ctx_sym = _active_context_ticker(agent)
            if ctx_sym:
                return ("__quote__", ctx_sym)
            console.print(f"  [{C_WARN}]usage: .quote TICKER[/{C_WARN}]")
            return True
        raw_syms = [
            _resolve_numeric_ticker(t.strip(), agent)
            for t in re.split(r"[,\s]+", parts[1].strip())
            if t.strip()
        ]
        return ("__quote__", ",".join(raw_syms) if len(raw_syms) > 1 else raw_syms[0])

    # .analyze [TICKER[, TICKER2, ...]] / .a [TICKER...]
    if lower in (".analyze", ".a") or lower.startswith(".analyze ") or lower.startswith(".a "):
        parts = stripped.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            ctx_sym = _active_context_ticker(agent)
            if ctx_sym:
                return [ctx_sym]
            console.print(f"  [{C_WARN}]usage: .analyze TICKER [, TICKER2, ...][/{C_WARN}]")
            return True
        raw = parts[1].strip()
        tickers = [
            _resolve_numeric_ticker(t.strip(), agent)
            for t in re.split(r'[,\s]+', raw)
            if t.strip()
        ]
        return tickers  # list signals multi-analyze

    # .tools -- diagnostic: show loaded MCP tools
    if lower == ".tools":
        tools = getattr(agent, '_all_tools', None) or []
        if not tools:
            console.print(f"  [{C_WARN}]No MCP tools loaded. Check MCP connection/token.[/{C_WARN}]")
        else:
            console.print(f"  [{C_GREEN}]{len(tools)} tools loaded:[/{C_GREEN}]")
            for t in sorted(tools, key=lambda x: x.name):
                console.print(f"    [{C_DIM}]{t.name}[/{C_DIM}]")
        return True

    # Catch any other dot-command locally so typos never leak to the LLM
    if stripped.startswith(".") and len(stripped) > 1 and not stripped[1].isdigit():
        token = lower.split()[0]
        matches = difflib.get_close_matches(token, KNOWN_DOT_COMMANDS, n=1, cutoff=0.5)
        if matches:
            console.print(
                f"  [{C_WARN}]Unknown command {escape(token)!r} -- "
                f"did you mean [{C_ACCENT}]{matches[0]}[/{C_ACCENT}]? (type ? for help)[/{C_WARN}]"
            )
        else:
            console.print(
                f"  [{C_WARN}]Unknown command {escape(token)!r} -- type ? for help[/{C_WARN}]"
            )
        return True

    return False

