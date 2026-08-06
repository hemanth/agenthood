"""Dot-command dispatch.

``handle_meta`` runs BEFORE the agent's own pipeline, so dot-commands never
reach the LLM.  Its return value tells the REPL what to do next:

  True                      -- fully handled, nothing more to do
  False                     -- not a dot-command, pass to the agent
  "__portfolio__"           -- run the portfolio action
  "__positions__"           -- run the positions action
  "__update__"              -- run the self-upgrade
  ("__quote__", "AAPL")     -- run the quote action for that ticker
  ["AAPL", "NVDA"]          -- run the analyze action for those tickers
"""
from __future__ import annotations

import re

from rich.markup import escape

from .models import _resolve_model, _short_model, apply_model_switch, show_model_picker
from .render import _show_table
from .sessions import print_sessions
from .theme import C_DIM, C_GREEN, C_STEP, C_WARN, console


def handle_meta(cmd: str, agent) -> bool:
    stripped = cmd.strip()
    lower    = stripped.lower()

    # .model [arg]
    if lower == ".model" or lower.startswith(".model "):
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

    if lower == ".sessions":
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

    if lower == ".memory":
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

    # .portfolio -- direct MCP call
    if lower in (".portfolio", ".port", ".p"):
        return "__portfolio__"

    # .positions -- direct MCP call
    if lower in (".positions", ".pos"):
        return "__positions__"

    # .quote TICKER -- direct MCP call
    if lower.startswith(".quote") or lower.startswith(".q "):
        parts = stripped.split(None, 1)
        if len(parts) < 2:
            console.print(f"  [{C_WARN}]usage: .quote TICKER[/{C_WARN}]")
            return True
        return ("__quote__", parts[1].strip().upper())

    # .analyze TICKER[, TICKER2, ...]
    if lower.startswith(".analyze"):
        parts = stripped.split(None, 1)
        if len(parts) < 2:
            console.print(f"  [{C_WARN}]usage: .analyze TICKER [, TICKER2, ...][/{C_WARN}]")
            return True
        # Parse comma or space separated tickers
        raw = parts[1].strip()
        tickers = [t.strip().upper() for t in re.split(r'[,\s]+', raw) if t.strip()]
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

    return False
