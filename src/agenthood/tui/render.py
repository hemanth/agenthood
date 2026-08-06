"""Everything that draws to the terminal: banners, panels, help, status bar."""
from __future__ import annotations

import re
from typing import Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .models import _short_model
from .theme import (
    C_ACCENT, C_DIM, C_DIMMER, C_ERR, C_FG, C_LABEL, C_TRADE_L, C_TRADE_R,
    console,
)


# ── Utility ───────────────────────────────────────────────────────────────────

def _strip_emoji(s: str) -> str:
    return re.sub(
        r"[\U00010000-\U0010ffff\U0001F300-\U0001F9FF"
        r"\u2600-\u27BF\uFE00-\uFE0F\u200d\u20e3]",
        "",
        s,
    )


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner(model: str, dry_run: bool, max_order: float,
                 token_info: str, session_name: Optional[str]) -> None:
    """Clean Rich Panel banner — sharp, minimal, no ASCII art."""
    mode_label = "dry-run" if dry_run else "live"
    mode_color = C_TRADE_L if dry_run else C_TRADE_R
    short = _short_model(model)

    body = Text()
    body.append(short, style=C_ACCENT)
    body.append("  ·  ")
    body.append(mode_label, style=mode_color)
    body.append(f"  ·  cap ${max_order:.0f}", style=C_DIM)
    if session_name:
        body.append(f"  ·  {session_name}", style=C_DIMMER)
    body.append("\n")
    body.append("?", style=C_ACCENT)
    body.append(" help", style=C_DIMMER)
    body.append("   .model", style=C_ACCENT)
    body.append(" switch", style=C_DIMMER)
    body.append("   .sessions", style=C_ACCENT)
    body.append(" list", style=C_DIMMER)
    body.append("   Ctrl+R", style=C_ACCENT)
    body.append(" search", style=C_DIMMER)

    title = Text()
    title.append("agenthood", style=f"bold {C_FG}")

    console.print()
    console.print(
        Panel(body, title=title, title_align="left",
              border_style=C_DIMMER, padding=(0, 1)),
    )
    console.print()


# ── Response rendering ────────────────────────────────────────────────────────

def _render_with_code_blocks(text: str):
    """Split on ```lang\\n...``` fences; return a Rich renderable."""
    from rich.markdown import Markdown

    fence_re = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    if not fence_re.search(text):
        # Check for Rich markup tags (e.g. [bold cyan], [bold green], [green])
        if "[" in text and "]" in text and any(tag in text for tag in ("green", "red", "yellow", "cyan", "bold")):
            return Text.from_markup(text)
        # Check for markdown features (tables, headers)
        has_markdown = any(marker in text for marker in ("|---|", "## "))
        if has_markdown:
            return Markdown(text)
        return Text(text)

    renderables: list = []
    i = 0
    for m in fence_re.finditer(text):
        before = text[i:m.start()]
        if before.strip():
            renderables.append(Markdown(before.rstrip()))
        lang = m.group(1) or "text"
        renderables.append(Syntax(m.group(2), lang, theme="monokai",
                                  background_color="default", word_wrap=True))
        i = m.end()
    after = text[i:]
    if after.strip():
        renderables.append(Markdown(after.lstrip()))
    return Group(*renderables) if renderables else Text(text)


def render_response(text: str, elapsed: float) -> None:
    text = _strip_emoji(text)
    style = C_ERR if (text.lstrip().startswith("!") or "[ERROR]" in text[:30]) else C_ACCENT
    console.print(
        Panel(
            _render_with_code_blocks(text),
            border_style=style,
            padding=(0, 2),
            title=f"[{C_LABEL}]agenthood[/{C_LABEL}]",
            title_align="left",
            subtitle=f"[{C_DIMMER}]{elapsed:.2f}s[/{C_DIMMER}]",
            subtitle_align="right",
        )
    )


# ── Help ──────────────────────────────────────────────────────────────────────

_HELP_EXAMPLES = [
    "what's paypal looking like?",
    "show me AAPL and NVDA side by side",
    "is TSLA overbought? check the RSI",
    "BTC and ETH prices",
    "my portfolio",
    "any earnings this week?",
    "should I buy more MSFT here?",
]

_HELP_META = [
    (".analyze",    "full analysis of a ticker (or multiple: .analyze PYPL, GOOGL)"),
    (".portfolio",  "view portfolio summary (.port, .p)"),
    (".positions",  "view open positions (.pos)"),
    (".quote",      "quick quote for a ticker (.q AAPL)"),
    (".tools",      "list loaded MCP tools"),
    (".model",      "switch model"),
    (".sessions",   "list sessions"),
    (".new",        "new conversation"),
    (".cost",       "token usage"),
    (".memory",     "remembered facts"),
    (".verbose",    "debug mode"),
    (".update",     "upgrade agenthood from PyPI"),
    ("exit",        "quit"),
]

_HELP_KEYS = [
    ("Ctrl+R",     "reverse-search history"),
    ("Up / Down",  "navigate history"),
    ("Tab",        "complete .commands"),
    ("\\",         "continue line (multi-line input)"),
]


def print_help() -> None:
    # Examples section
    ex = Text()
    ex.append("just ask anything — ", style=C_DIMMER)
    ex.append("no special syntax needed\n\n", style=C_DIM)
    for e in _HELP_EXAMPLES:
        ex.append("  ❯  ", style=C_DIMMER)
        ex.append(e, style=C_FG)
        ex.append("\n")

    # Meta commands
    ex.append("\n")
    mt = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    mt.add_column(style=C_ACCENT, no_wrap=True)
    mt.add_column(style=C_DIM)
    for cmd, desc in _HELP_META:
        mt.add_row(cmd, desc)

    # Key bindings
    kb = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    kb.add_column(style=C_ACCENT, no_wrap=True)
    kb.add_column(style=C_DIM)
    for key, desc in _HELP_KEYS:
        kb.add_row(key, desc)

    console.print()
    console.print(Panel(
        Group(ex, Rule(style=C_DIMMER), mt, Rule(style=C_DIMMER), kb),
        border_style=C_DIMMER,
        title=Text("help", style=C_LABEL),
        title_align="left",
        padding=(0, 1),
    ))
    console.print()


# ── Tables / status ───────────────────────────────────────────────────────────

def _show_table(data: dict, title: str) -> None:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style=C_LABEL)
    t.add_column(style=C_FG)
    for k, v in (data or {}).items():
        t.add_row(str(k), str(v))
    console.print(Panel(t, border_style=C_DIM,
                        title=f"[{C_LABEL}]{title}[/{C_LABEL}]", title_align="left"))


def _analyze_provider(analyze_base: str) -> str:
    """Human-readable provider name from an api_base URL."""
    if "groq.com" in analyze_base:
        return "groq"
    if "openrouter" in analyze_base:
        return "openrouter"
    if "openai.com" in analyze_base:
        return "openai"
    if "localhost" in analyze_base or "127.0.0.1" in analyze_base:
        return "local"
    from urllib.parse import urlparse
    return urlparse(analyze_base).hostname or "custom"


def _print_status(agent, turn: int, cost_usd: float) -> None:
    cfg   = agent.config
    trade_label = "DRY-RUN" if cfg.dry_run else "LIVE"
    trade_style = C_TRADE_L if cfg.dry_run else C_TRADE_R
    cost  = f"  ${cost_usd:.4f}" if cost_usd else ""
    fallbacks = getattr(cfg, 'model_fallbacks', [])
    line  = Text("  ")
    line.append("model ", style=C_DIM)
    line.append(_short_model(cfg.model), style=C_ACCENT)
    if fallbacks:
        line.append(f" +{len(fallbacks)} fallback{'s' if len(fallbacks) > 1 else ''}", style=C_DIM)
    # Show analyze provider if configured
    analyze_model = getattr(cfg, 'analyze_model', '')
    analyze_base = getattr(cfg, 'analyze_api_base', '')
    if analyze_model and analyze_base:
        line.append(" | ", style=C_DIM)
        line.append("analyze ", style=C_DIM)
        line.append(_short_model(analyze_model), style=C_ACCENT)
        line.append(f" ({_analyze_provider(analyze_base)})", style=C_DIM)
    line.append("  ", style=C_DIM)
    line.append(trade_label, style=trade_style)
    line.append(f"  turn {turn}{cost}", style=C_DIM)
    console.print(line)
