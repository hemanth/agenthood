"""agenthood TUI -- terminal interface built with Rich.

Design principles (distilled from agy, codex, claude CLIs):
  agy    -- compact status bar, tool trace shown inline, always feels fast
  codex  -- numbered model picker, minimal chrome, task-first layout
  claude -- live thinking panel, beautiful panels, context-aware footer

Features:
  - Persistent readline history  (~/.agenthood/history)
  - Arrow keys + Ctrl+R reverse search  (readline / libedit)
  - Session management  (--session NAME to continue, --new-session for fresh)
  - Numbered model picker  (.model command, hot-swap, no restart needed)
  - Live thinking panel  (transient spinner + step trace)
  - Syntax-highlighted response panels  (fenced code blocks detected)
  - Elapsed time shown per response

No emojis. No em-dashes. ASCII via Rich only.
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
import threading
import readline
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich.markup import escape
from rich.syntax import Syntax
from rich import box


# ── Directories ───────────────────────────────────────────────────────────────
from .config import DATA_DIR as _DATA_DIR, SESS_DIR as _SESS_DIR
_HIST_FILE  = _DATA_DIR / "history"

# ── Consoles ──────────────────────────────────────────────────────────────────
console = Console(highlight=False, markup=True)

# ── Colour tokens ─────────────────────────────────────────────────────────────
C_ACCENT  = "bright_cyan"
C_DIM     = "grey50"
C_DIMMER  = "grey35"
C_FG      = "bright_white"
C_WARN    = "yellow"
C_ERR     = "red"
C_GREEN   = "bright_green"
C_STEP    = "grey70"
C_LABEL   = "grey42"
C_TRADE_L = "yellow"
C_TRADE_R = "red"

# ── ASCII spinner ─────────────────────────────────────────────────────────────
_SPIN = ["|", "/", "-", "\\"]

# ── Curated model catalogue ───────────────────────────────────────────────────
MODELS: list[tuple[str, str, str, str, str]] = [
    # (model_id, display_label, provider, speed, note)
    # --- OpenRouter free (ranked by tool-calling benchmark) ------------------
    ("meta-llama/llama-3.3-70b-instruct:free","Llama 3.3 70B",     "openrouter", "fast",   "#1 free: best JSON tool-call accuracy"),
    ("openai/gpt-oss-120b:free",              "GPT-OSS 120B",      "openrouter", "fast",   "#2 free: built for agentic tool-use"),
    ("openai/gpt-oss-20b:free",               "GPT-OSS 20B",       "openrouter", "fast",   "#3 free: fast, reliable tool-calls"),
    ("qwen/qwen3-coder:free",                 "Qwen3 Coder",       "openrouter", "fast",   "#4 free: strong agent/function-call"),
    ("deepseek/deepseek-r1-0528:free",        "DeepSeek R1-0528",  "openrouter", "medium", "#5 free: best reasoning + tool calls"),
    ("google/gemini-flash-1.5:free",          "Gemini 1.5 Flash",  "openrouter", "fast",   "#6 free: long context, fast"),
    ("mistralai/mistral-7b-instruct:free",    "Mistral 7B",        "openrouter", "fast",   "#7 free: lightweight"),
    # --- OpenRouter paid ------------------------------------------------------
    ("openai/gpt-4o",                         "GPT-4o",            "openrouter", "medium", "best all-round"),
    ("openai/o3-mini",                        "o3-mini",           "openrouter", "medium", "strong reasoning"),
    ("anthropic/claude-sonnet-4-5",           "Claude Sonnet 4.5", "openrouter", "medium", "excellent prose"),
    ("anthropic/claude-opus-4-5",             "Claude Opus 4.5",   "openrouter", "slow",   "most capable"),
    ("google/gemini-2.0-flash-001",           "Gemini 2.0 Flash",  "openrouter", "fast",   "long context"),
    ("google/gemini-2.5-pro",                 "Gemini 2.5 Pro",    "openrouter", "medium", "deep reasoning"),
    # --- Ollama local ---------------------------------------------------------
    ("qwen3:latest",                          "Qwen3",             "ollama",     "fast",   "local, strong tool-calls"),
    ("llama3.3:latest",                       "Llama 3.3",         "ollama",     "fast",   "local, general"),
    ("deepseek-r1:latest",                    "DeepSeek R1",       "ollama",     "slow",   "local, reasoning"),
    ("mistral:latest",                        "Mistral 7B",        "ollama",     "fast",   "local, lightweight"),
    ("codestral:latest",                      "Codestral",         "ollama",     "medium", "local, code-optimised"),
]

_PROVIDER_COL = {"openrouter": C_ACCENT,  "ollama": C_GREEN}
_SPEED_COL    = {"fast": C_GREEN, "medium": C_WARN, "slow": C_ERR}


# ── Readline: history + tab-completion + key bindings ────────────────────────

# All dot-commands that can be tab-completed
_DOT_COMMANDS = [
    ".model", ".sessions", ".new", ".reset", ".cost", ".memory", ".findings", ".verbose",
]

# Common intent phrases for completion
_INTENT_PHRASES = [
    "buy ", "sell ", "should I buy ", "should I sell ",
    "compare ", "analyze ", "what do you think about ",
    "show my portfolio", "show my positions", "show my orders",
    "portfolio", "positions", "orders", "watchlist",
    "cancel order", "what is ",
]


def _make_completer():
    """Return a readline completer that handles:
      - .command  tab-expands dot-commands
      - .model N  tab-expands to numbered model names
      - TICKER    tab-expands common two-token shorthands
      - anything  falls back to history-based completion
    """
    # Build a flat list of all completion candidates
    model_labels = [m[1].lower() for m in MODELS]   # display names
    model_ids    = [m[0] for m in MODELS]             # full ids
    model_nums   = [str(i) for i in range(1, len(MODELS)+1)]
    ticker_sfx   = [
        "options", "calls", "puts", "rsi", "macd",
        "earnings", "chart", "history", "fundamentals",
        "analyst", "ratings", "news",
    ]

    def completer(text: str, state: int):
        buf  = readline.get_line_buffer()
        word = text  # the current incomplete word

        # --- .model <N|name> completion ------------------------------------
        if buf.lower().startswith(".model "):
            arg = buf[7:]
            candidates = model_nums + model_labels + model_ids
            matches = [c for c in candidates if c.lower().startswith(arg.lower())]
            if state < len(matches):
                # Replace just the arg part
                return matches[state]
            return None

        # --- dot-command completion ----------------------------------------
        if buf.startswith("."):
            matches = [c for c in _DOT_COMMANDS if c.startswith(buf)]
            if state < len(matches):
                return matches[state]
            return None

        # --- TICKER <suffix> completion ------------------------------------
        # If the buffer looks like "WORD " (all-caps word + space), offer suffixes
        parts = buf.split()
        if len(parts) == 1 and parts[0].isupper() and len(parts[0]) <= 5:
            matches = [s for s in ticker_sfx if s.startswith(word)]
            if state < len(matches):
                return matches[state]
            return None

        # --- intent phrase completion at start of line --------------------
        if not buf or word == buf:  # beginning of line
            matches = [
                p for p in _INTENT_PHRASES + _DOT_COMMANDS
                if p.startswith(word)
            ]
            if state < len(matches):
                return matches[state]
            return None

        return None

    return completer


def setup_readline() -> None:
    """Configure libedit readline for agenthood REPL.

    Binding order matters on libedit:
      1. 'bind -e'  -- sets emacs keymap (enables Up/Down arrows, Ctrl+A/E etc)
      2. Ctrl+R     -- reverse incremental search (must come AFTER bind -e)
      3. Tab        -- rl_complete (must come AFTER bind -e)
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing history (silently skip if not found)
    try:
        readline.read_history_file(str(_HIST_FILE))
    except (FileNotFoundError, OSError):
        pass

    readline.set_history_length(1000)

    is_libedit = "libedit" in (readline.__doc__ or "")
    if is_libedit:
        readline.parse_and_bind("bind -e")                    # 1. emacs keymaps first
        readline.parse_and_bind("bind ^R em-inc-search-prev") # 2. Ctrl+R search
        readline.parse_and_bind("bind ^S em-inc-search-next") # 3. Ctrl+S forward
        readline.parse_and_bind("bind ^I rl_complete")        # 4. Tab complete
    else:
        # GNU readline
        readline.parse_and_bind("set editing-mode emacs")
        readline.parse_and_bind(r'"\C-r": reverse-search-history')
        readline.parse_and_bind("tab: complete")

    # Install completer
    readline.set_completer(_make_completer())
    # Don't split on '.' so ".model" is treated as one word
    readline.set_completer_delims(" \t\n")


def save_readline() -> None:
    """Flush readline history to disk."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        readline.write_history_file(str(_HIST_FILE))
    except Exception:
        pass


def _add_history(line: str) -> None:
    """Add *line* to readline history, purging meta-commands.

    readline's input() auto-appends every entered line before we see it.
    For dot-commands we must also remove that auto-added entry so
    up-arrow navigation never surfaces .model / .sessions etc.
    """
    s = line.strip()
    if not s:
        return
    is_meta = s.startswith(".") or s.lower() in ("help", "exit", "quit", "q")
    n = readline.get_current_history_length()
    if is_meta:
        # Remove the entry readline just auto-added (always the last item)
        if n > 0 and readline.get_history_item(n) == s:
            readline.remove_history_item(n - 1)  # remove_history_item is 0-indexed
        return
    # Deduplicate: skip if identical to the previous entry
    if n > 0 and readline.get_history_item(n) == s:
        return
    readline.add_history(s)


# ── Session management ────────────────────────────────────────────────────────

def list_sessions() -> list[str]:
    """Return sorted list of saved session names."""
    _SESS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in _SESS_DIR.glob("*.json"))


def print_sessions() -> None:
    sessions = list_sessions()
    if not sessions:
        console.print(f"  [{C_DIM}]No saved sessions.  Use --session NAME to start one.[/{C_DIM}]")
        return
    t = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {C_DIM}",
              padding=(0, 2))
    t.add_column("Name",     style=C_FG,    no_wrap=True)
    t.add_column("Modified", style=C_DIMMER, no_wrap=True)
    for name in sessions:
        p = _SESS_DIR / f"{name}.json"
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        t.add_row(name, mtime)
    console.print(Panel(t, border_style=C_DIM,
                        title=f"[{C_LABEL}]sessions[/{C_LABEL}]", title_align="left"))


def session_path(name: str) -> Path:
    _SESS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESS_DIR / f"{name}.json"


# ── Status bar / banner ───────────────────────────────────────────────────────

def _short_model(model: str) -> str:
    return model.split("/")[-1].replace(":free", "").replace(":latest", "")


def print_banner(model: str, dry_run: bool, max_order: float,
                 token_info: str, session_name: Optional[str]) -> None:
    """Clean Rich Panel banner — sharp, minimal, no ASCII art."""
    from rich.panel import Panel

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


# ── Thinking panel ────────────────────────────────────────────────────────────

class ThinkingDisplay:
    """Transient Live panel: spinner + step trace.  Erases itself on exit."""

    def __init__(self, con: Console):
        self._con    = con
        self._steps: list[str] = []
        self._start  = time.monotonic()
        self._frame  = 0
        self._done   = False
        self._lock   = threading.Lock()
        self._live: Optional[Live]         = None
        self._task: Optional[asyncio.Task] = None

    def step(self, msg: str) -> None:
        with self._lock:
            self._steps.append(msg)

    def update(self, msg: str) -> None:
        """Replace the last step message (for live counters)."""
        with self._lock:
            if self._steps:
                self._steps[-1] = msg
            else:
                self._steps.append(msg)

    def _render(self) -> Panel:
        elapsed = time.monotonic() - self._start
        spin    = _SPIN[self._frame % len(_SPIN)]
        self._frame += 1

        body = Text()
        body.append(f"  {spin} ", style=C_ACCENT)
        body.append("thinking  ", style=C_DIM)
        body.append(f"{elapsed:.1f}s", style=C_LABEL)

        with self._lock:
            steps = list(self._steps)

        for i, s in enumerate(steps):
            body.append("\n")
            body.append("  +-- " if i < len(steps) - 1 else "  `-- ", style=C_DIMMER)
            if ":" in s:
                tag, rest = s.split(":", 1)
                body.append(tag + ":", style=C_ACCENT)
                body.append(rest, style=C_STEP)
            else:
                body.append(s, style=C_STEP)

        return Panel(body, border_style=C_DIMMER, padding=(0, 1))

    async def _spin_loop(self) -> None:
        while not self._done:
            if self._live:
                self._live.update(self._render())
            await asyncio.sleep(0.10)

    async def __aenter__(self) -> "ThinkingDisplay":
        self._live = Live(self._render(), console=self._con,
                          refresh_per_second=10, transient=True)
        self._live.__enter__()
        self._task = asyncio.create_task(self._spin_loop())
        return self

    def stop(self) -> None:
        """Stop the spinner and remove the display without exiting the context."""
        self._done = True
        if self._live:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None

    async def __aexit__(self, *_) -> None:
        self._done = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._live:
            self._live.__exit__(None, None, None)


# ── Utility ───────────────────────────────────────────────────────────────────

def _strip_emoji(s: str) -> str:
    return re.sub(
        r"[\U00010000-\U0010ffff\U0001F300-\U0001F9FF"
        r"\u2600-\u27BF\uFE00-\uFE0F\u200d\u20e3]",
        "",
        s,
    )


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

def _render_with_code_blocks(text: str):
    """Split on ```lang\\n...``` fences; return a Rich renderable."""
    from rich.markdown import Markdown

    fence_re = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    if not fence_re.search(text):
        # No code fences -- check for markdown features (tables, headers, bold)
        has_markdown = any(marker in text for marker in ("|---|", "## ", "**"))
        if has_markdown:
            return Markdown(text)
        return Text(text)

    from rich.console import Group
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


# ── Model picker ──────────────────────────────────────────────────────────────

def show_model_picker(current: str) -> Optional[str]:
    """Interactive numbered model table.  Returns new model id or None."""
    console.print()
    t = Table(box=box.SIMPLE, show_header=True,
              header_style=f"bold {C_DIM}", padding=(0, 2), expand=False)
    t.add_column("#",        style=C_DIMMER, width=4,  no_wrap=True)
    t.add_column("Model",    style=C_FG,     min_width=20, no_wrap=True)
    t.add_column("Provider", width=11,        no_wrap=True)
    t.add_column("Speed",    width=7,         no_wrap=True)
    t.add_column("Note",     style=C_DIMMER,  no_wrap=True)

    last_prov = None
    for idx, (mid, label, prov, speed, note) in enumerate(MODELS, 1):
        if prov != last_prov:
            t.add_row("", "", "", "", "")
            last_prov = prov
        mark = f"[{C_ACCENT}]*[/{C_ACCENT}] " if mid == current else "  "
        t.add_row(
            str(idx),
            f"{mark}{label}",
            f"[{_PROVIDER_COL.get(prov, C_DIM)}]{prov}[/{_PROVIDER_COL.get(prov, C_DIM)}]",
            f"[{_SPEED_COL.get(speed, C_DIM)}]{speed}[/{_SPEED_COL.get(speed, C_DIM)}]",
            note,
        )

    console.print(Panel(
        t,
        border_style=C_DIM,
        title=f"[{C_LABEL}]model picker[/{C_LABEL}]",
        title_align="left",
        subtitle=f"[{C_DIMMER}]number / name prefix / full id  --  empty to cancel[/{C_DIMMER}]",
        subtitle_align="right",
    ))

    try:
        raw = input("\x1b[38;5;75m❯\x1b[0m  ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]")
        return None

    if not raw:
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]")
        return None

    if raw.isdigit():
        n = int(raw)
        if 1 <= n <= len(MODELS):
            mid, label = MODELS[n - 1][0], MODELS[n - 1][1]
            console.print(f"  [{C_GREEN}]switched to {label}[/{C_GREEN}]")
            return mid
        console.print(f"  [{C_WARN}]! invalid number[/{C_WARN}]")
        return None

    raw_low = raw.lower()
    for mid, label, *_ in MODELS:
        if mid.lower().startswith(raw_low) or label.lower().startswith(raw_low):
            console.print(f"  [{C_GREEN}]switched to {label}[/{C_GREEN}]")
            return mid

    # Accept arbitrary model id  (has / or : or multiple -)
    if "/" in raw or ":" in raw or raw.count("-") >= 2:
        console.print(f"  [{C_GREEN}]switched to {raw}[/{C_GREEN}]")
        return raw

    console.print(f"  [{C_WARN}]! no match for {raw!r}[/{C_WARN}]")
    return None


def apply_model_switch(agent, new_model: str) -> None:
    """Hot-swap LLM model on running agent (no restart needed)."""
    try:
        if agent._agent is not None:
            agent._agent.model = new_model
        agent.config.model = new_model
    except Exception as exc:
        console.print(f"  [{C_WARN}]! model switch error: {exc}[/{C_WARN}]")


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

    from rich.rule import Rule
    console.print()
    console.print(Panel(
        Group(ex, Rule(style=C_DIMMER), mt, Rule(style=C_DIMMER), kb),
        border_style=C_DIMMER,
        title=Text("help", style=C_LABEL),
        title_align="left",
        padding=(0, 1),
    ))
    console.print()


# ── Meta commands ─────────────────────────────────────────────────────────────

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


def _resolve_model(arg: str) -> Optional[str]:
    if arg.isdigit():
        n = int(arg)
        if 1 <= n <= len(MODELS):
            label = MODELS[n - 1][1]
            console.print(f"  [{C_GREEN}]switched to {label}[/{C_GREEN}]")
            return MODELS[n - 1][0]
        console.print(f"  [{C_WARN}]! invalid number[/{C_WARN}]")
        return None
    raw_low = arg.lower()
    for mid, label, *_ in MODELS:
        if mid.lower().startswith(raw_low) or label.lower().startswith(raw_low):
            return mid
    if "/" in arg or ":" in arg or arg.count("-") >= 2:
        return arg
    console.print(f"  [{C_WARN}]! no match for {arg!r}[/{C_WARN}]")
    return None


def _show_table(data: dict, title: str) -> None:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style=C_LABEL)
    t.add_column(style=C_FG)
    for k, v in (data or {}).items():
        t.add_row(str(k), str(v))
    console.print(Panel(t, border_style=C_DIM,
                        title=f"[{C_LABEL}]{title}[/{C_LABEL}]", title_align="left"))


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
        # Extract provider name from api_base (e.g. "groq" from "https://api.groq.com/...")
        provider = ""
        if "groq.com" in analyze_base:
            provider = "groq"
        elif "openrouter" in analyze_base:
            provider = "openrouter"
        elif "openai.com" in analyze_base:
            provider = "openai"
        elif "localhost" in analyze_base or "127.0.0.1" in analyze_base:
            provider = "local"
        else:
            from urllib.parse import urlparse
            provider = urlparse(analyze_base).hostname or "custom"
        line.append(" | ", style=C_DIM)
        line.append("analyze ", style=C_DIM)
        line.append(_short_model(analyze_model), style=C_ACCENT)
        line.append(f" ({provider})", style=C_DIM)
    line.append("  ", style=C_DIM)
    line.append(trade_label, style=trade_style)
    line.append(f"  turn {turn}{cost}", style=C_DIM)
    console.print(line)


# ── Main REPL ─────────────────────────────────────────────────────────────────

async def tui_repl(agent, session_name: Optional[str] = None) -> None:
    """Main TUI REPL.  Call with optional session_name to persist context."""
    from agenthood.token import token_status

    cfg  = agent.config
    tok  = token_status()

    # Load session memory if continuing
    sess_file = session_path(session_name) if session_name else None
    if sess_file and sess_file.exists():
        console.print(f"  [{C_DIM}]continuing session {session_name!r}[/{C_DIM}]")

    setup_readline()
    print_banner(
        model=cfg.model or "auto",
        dry_run=cfg.dry_run,
        max_order=cfg.max_order_value,
        token_info=tok,
        session_name=session_name,
    )

    turn          = 0
    verbose       = False
    total_cost_usd = 0.0

    # ── Price watcher state ────────────────────────────────────────────────────
    _watch_tasks: list[asyncio.Task] = []
    _watch_queue: asyncio.Queue      = asyncio.Queue()

    async def _watcher_price_fn(ticker: str) -> float:
        """Fetch latest bid price for a ticker via the Robinhood quote tool."""
        import re as _re
        import json as _json
        try:
            raw = await agent._call_tool_direct(
                "robinhood_get_equity_quotes", {"symbols": [ticker.upper()]}
            )
            txt = str(raw)
            # Try JSON first
            try:
                data = _json.loads(txt)
                for key in ("last_trade_price", "bid_price", "ask_price", "price"):
                    v = data.get(key)
                    if v is not None:
                        return float(v)
            except Exception:
                pass
            # Fall back: scrape first numeric-looking price
            m = _re.search(r'"?(?:price|last|bid)[":]?\s*"?(\d[\d,]*\.?\d*)"?', txt, _re.I)
            if m:
                return float(m.group(1).replace(",", ""))
            m = _re.search(r'\$(\d[\d,]*(?:\.\d+)?)', txt)
            if m:
                return float(m.group(1).replace(",", ""))
        except Exception:
            pass
        # Last resort: use chat() and scrape the panel output
        result = await agent.chat(ticker)
        m = _re.search(r'\$(\d[\d,]*(?:\.\d+)?)', str(result))
        if m:
            return float(m.group(1).replace(",", ""))
        raise ValueError(f"Cannot parse price for {ticker} from: {result!r}")

    async def _on_watcher_trigger(watcher, price) -> None:  # noqa: F811
        await _watch_queue.put((watcher, price))

    def _start_watcher(spec: dict) -> None:
        from agenthood.watcher import PriceWatcher
        w = PriceWatcher(
            ticker     = spec["ticker"],
            threshold  = spec["threshold"],
            direction  = spec["direction"],
            action     = spec["action"],
            shares     = spec["shares"],
            interval   = spec.get("interval", 30),
            get_price  = _watcher_price_fn,
            on_trigger = _on_watcher_trigger,
        )
        task = asyncio.create_task(w.run())
        _watch_tasks.append(task)
        dir_word   = "below" if w.direction == "below" else "above"
        action_str = (
            "alert you"
            if w.action == "alert"
            else f"{w.action} {w.shares} shares"
        )
        render_response(
            f"Watching {w.ticker} \u2014 will {action_str} when price goes "
            f"{dir_word} ${w.threshold:,.2f}.  Polling every {w.interval}s.",
            0.0,
        )

    def read_input() -> str:
        """Read one logical line from stdin.

        Multi-line: a line ending with \\ continues on the next prompt.
        The continuation prompt aligns under the chevron.
        The joined result is added to history as a single entry.
        """
        PROMPT_MAIN = "\n\x1b[38;5;75m❯\x1b[0m  "   # bright blue chevron, agy-style
        PROMPT_CONT = "  \x1b[90m·\x1b[0m  "         # dim dot for continuation lines
        lines: list[str] = []
        while True:
            prompt = PROMPT_MAIN if not lines else PROMPT_CONT
            try:
                chunk = input(prompt)
            except (EOFError, KeyboardInterrupt):
                raise
            if chunk.endswith("\\"):
                lines.append(chunk[:-1].rstrip())
            else:
                lines.append(chunk)
                break
        full = " ".join(lines)
        _add_history(full)
        return full.strip()

    async def _read_input_async() -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, read_input)

    async def _drain_notifications_during_input(input_future: asyncio.Future) -> None:
        """Print any triggered watcher notifications while readline is blocking."""
        while not input_future.done():
            try:
                w_obj, w_price = _watch_queue.get_nowait()
                # print notification above current line
                sys.stdout.write('\n')
                sys.stdout.write(f'  ⚡ WATCHER: {w_obj.ticker} hit ${w_price:,.2f}\n')
                sys.stdout.flush()
            except asyncio.QueueEmpty:
                pass
            await asyncio.sleep(0.5)

    _last_interrupt: float = 0.0   # timestamp of most recent Ctrl+C

    from agenthood.daemon import get_pending_triggers

    try:
        while True:
            # ── Check background daemon triggers ──────────────────────────────
            for trigger in get_pending_triggers():
                spec = trigger['spec']
                price = trigger['price']
                console.print()
                console.print(
                    f"  [{C_WARN}]  DAEMON WATCHER TRIGGERED  "
                    f"{spec['ticker']} hit ${price:,.2f} "
                    f"(threshold {spec['direction']} ${spec['threshold']:,.2f})[/{C_WARN}]"
                )

            # ── Input ─────────────────────────────────────────────────────────
            try:
                # user_input = read_input()
                input_task = asyncio.create_task(_read_input_async())
                drain_task = asyncio.create_task(_drain_notifications_during_input(input_task))
                await asyncio.wait([input_task, drain_task], return_when=asyncio.FIRST_COMPLETED)
                drain_task.cancel()
                user_input = await input_task
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - _last_interrupt < 2.0:
                    # Second Ctrl+C within 2 s → exit
                    console.print(f"\n  [{C_DIM}]session ended  {turn} turns[/{C_DIM}]")
                    break
                _last_interrupt = now
                console.print(f"\n  [{C_DIM}]Ctrl+C again to exit  (or type exit)[/{C_DIM}]")
                continue
            except EOFError:
                console.print(f"\n  [{C_DIM}]session ended  {turn} turns[/{C_DIM}]")
                break

            if not user_input:
                continue

            # ── Drain triggered watchers ──────────────────────────────────────
            while not _watch_queue.empty():
                try:
                    w_obj, w_price = _watch_queue.get_nowait()
                    dir_word  = "below" if w_obj.direction == "below" else "above"
                    console.print()
                    console.print(
                        f"  [{C_WARN}]  WATCHER TRIGGERED  "
                        f"{w_obj.ticker} hit ${w_price:,.2f} "
                        f"(threshold {dir_word} ${w_obj.threshold:,.2f})[/{C_WARN}]"
                    )
                    if w_obj.action != "alert":
                        console.print(
                            f"  [{C_DIM}]  Proposed: "
                            f"{w_obj.action} {w_obj.shares} {w_obj.ticker} "
                            f"— confirm? (yes / no)[/{C_DIM}]"
                        )
                        try:
                            confirm = input("  confirm> ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            confirm = "no"
                        if confirm in ("yes", "y"):
                            # Route through agent safety review
                            order_prompt = (
                                f"{w_obj.action} {w_obj.shares} {w_obj.ticker} "
                                f"at market — triggered by price watcher"
                            )
                            console.print(
                                f"  [{C_DIM}]routing through safety review...[/{C_DIM}]"
                            )
                            try:
                                resp = await agent.chat(order_prompt)
                                console.print()
                                render_response(
                                    _strip_emoji(str(resp)), 0.0
                                )
                            except Exception as w_exc:
                                console.print(
                                    f"  [{C_ERR}]Order error: {escape(str(w_exc))}[/{C_ERR}]"
                                )
                        else:
                            console.print(f"  [{C_DIM}]  Order cancelled.[/{C_DIM}]")
                except asyncio.QueueEmpty:
                    break

            if user_input in ("?", "help", ".help", "?help"):
                print_help()
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                # Cancel background watchers before exiting
                for t in _watch_tasks:
                    t.cancel()
                console.print(f"  [{C_DIM}]session ended  {turn} turns[/{C_DIM}]")
                break
            if user_input.lower() == ".verbose":
                verbose = not verbose
                console.print(f"  [{C_DIM}]verbose {'on' if verbose else 'off'}[/{C_DIM}]")
                continue
            meta_result = handle_meta(user_input, agent)
            if meta_result is True:
                _print_status(agent, turn, total_cost_usd)
                continue

            # .portfolio / .positions / .quote -- direct MCP calls
            if isinstance(meta_result, str) and meta_result.startswith("__"):
                turn += 1
                t0 = time.monotonic()
                response = None
                async with ThinkingDisplay(console) as td:
                    try:
                        # Get account number first
                        import json as _json
                        acct_raw = await agent._call_tool_direct("robinhood_get_accounts", {})
                        acct_data = _json.loads(acct_raw) if isinstance(acct_raw, str) else acct_raw
                        accounts = acct_data.get("data", {}).get("accounts", [])
                        # Prefer agentic account, then default, then first
                        acct_num = ""
                        for a in accounts:
                            if a.get("agentic_allowed"):
                                acct_num = a.get("account_number", "")
                                break
                        if not acct_num:
                            for a in accounts:
                                if a.get("is_default"):
                                    acct_num = a.get("account_number", "")
                                    break
                        if not acct_num and accounts:
                            acct_num = accounts[0].get("account_number", "")

                        if meta_result == "__portfolio__":
                            td.step("fetching portfolio...")
                            raw = await agent._call_tool_direct(
                                "robinhood_get_portfolio", {"account_number": acct_num}
                            )
                            response = _format_mcp_response(raw, "Portfolio")
                        elif meta_result == "__positions__":
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
                    elapsed = time.monotonic() - t0
                    render_response(response, elapsed)
                _print_status(agent, turn, total_cost_usd)
                continue

            if isinstance(meta_result, tuple) and meta_result[0] == "__quote__":
                turn += 1
                ticker = meta_result[1]
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
                    elapsed = time.monotonic() - t0
                    render_response(response, elapsed)
                _print_status(agent, turn, total_cost_usd)
                continue
            if isinstance(meta_result, list):
                # .analyze TICKER [, TICKER2, ...]
                tickers_raw = meta_result
                turn += 1

                # Resolve company names to symbols
                tickers = []
                for sym in tickers_raw:
                    resolved = await _resolve_ticker(agent, sym)
                    tickers.append(resolved)

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
                        elapsed = time.monotonic() - t0
                        render_response(_strip_emoji(response), elapsed)
                _print_status(agent, turn, total_cost_usd)
                continue

            # ── Thinking + inference ──────────────────────────────────────────
            turn += 1
            t0       = time.monotonic()
            response = None
            error    = None

            async with ThinkingDisplay(console) as td:
                td.step("routing query...")
                try:
                    response = await agent.chat(
                        user_input,
                        _progress=td.step,
                    )
                    td.step("done")
                except KeyboardInterrupt:
                    td.step("cancelled")
                    console.print(f"\n  [{C_DIM}]cancelled[/{C_DIM}]")
                    _last_interrupt = time.monotonic()
                    continue
                except Exception as exc:
                    error = str(exc)
                    td.step(f"error: {error[:80]}")

            elapsed = time.monotonic() - t0

            # Cost tracking
            try:
                c = agent.cost()
                if isinstance(c, dict) and "total_cost" in c:
                    total_cost_usd = float(c["total_cost"])
            except Exception:
                pass

            # ── Output ─────────────────────────────────────────────────────────
            console.print()
            if error:
                console.print(f"  [{C_ERR}]! Error: {escape(error)}[/{C_ERR}]")
            elif response is not None:
                if isinstance(response, dict) and "__watch__" in response:
                    # Watcher sentinel — start background task
                    _start_watcher(response["__watch__"])
                    turn -= 1  # don't count as a regular turn
                elif isinstance(response, dict):
                    msg = response.get("error", str(response))
                    console.print(f"  [{C_WARN}]! {escape(_strip_emoji(msg))}[/{C_WARN}]")
                elif not str(response).strip():
                    # Empty response — model couldn't handle this input
                    render_response(
                        "I'm a trading assistant. "
                        "Ask me about stocks, crypto, options, or your portfolio.",
                        elapsed,
                    )
                else:
                    render_response(_strip_emoji(str(response)), elapsed)
            elif cfg.stream:
                console.print(f"  [{C_LABEL}]agenthood >[/{C_LABEL}]  ", end="")
                try:
                    await agent.stream(user_input)
                except Exception as exc:
                    console.print(f"\n  [{C_ERR}]! {escape(str(exc))}[/{C_ERR}]")

    finally:
        # Always save history and session
        save_readline()
        if sess_file and session_name:
            # Persist session name (agentu session data lives in agent internals)
            sess_file.parent.mkdir(parents=True, exist_ok=True)
            import json
            sess_file.write_text(json.dumps({
                "name":  session_name,
                "turns": turn,
                "model": cfg.model,
            }, indent=2))
            console.print(f"  [{C_DIMMER}]session saved: {session_name}[/{C_DIMMER}]")
