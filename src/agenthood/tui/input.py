"""Readline: history, tab-completion, key bindings, and line reading."""
from __future__ import annotations

import readline

from ..config import DATA_DIR as _DATA_DIR
from .models import MODELS

_HIST_FILE = _DATA_DIR / "history"

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


PROMPT_MAIN = "\n\x1b[38;5;75m❯\x1b[0m  "   # bright blue chevron, agy-style
PROMPT_CONT = "  \x1b[90m·\x1b[0m  "         # dim dot for continuation lines


def read_input() -> str:
    """Read one logical line from stdin.

    Multi-line: a line ending with \\ continues on the next prompt.
    The continuation prompt aligns under the chevron.
    The joined result is added to history as a single entry.
    """
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
