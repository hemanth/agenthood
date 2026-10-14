"""Readline: history, tab-completion, key bindings, and line reading."""
from __future__ import annotations

import readline

from ..config import DATA_DIR as _DATA_DIR
from .models import MODELS
from .theme import PROMPT_CONT, PROMPT_MAIN

from ..core.tickers import NAME_TO_TICKER as _NAME_TO_TICKER

_HIST_FILE = _DATA_DIR / "history"

# All dot-commands that can be tab-completed. Keep in sync with handle_meta --
# the six most-used commands were missing here, so Tab did nothing for them.
_DOT_COMMANDS = [
    ".analyze", ".a", ".portfolio", ".p", ".positions", ".pos",
    ".quote", ".q", ".watches", ".watch", ".history", ".status",
    ".tools", ".help", ".model", ".sessions", ".new", ".reset", ".clear",
    ".cost", ".memory", ".findings", ".verbose", ".update",
    ".trending", ".t", ".movers", ".risk", ".earnings",
]

# Dynamic ticker pool updated during the session (recent tickers, positions, watchlists)
_DYNAMIC_TICKERS: list[str] = []


def set_completion_context(tickers: list[str] | None = None) -> None:
    """Update the dynamic ticker list prioritized by the readline tab completer."""
    global _DYNAMIC_TICKERS
    if not tickers:
        return
    merged: list[str] = []
    for sym in tickers:
        s = str(sym).strip().upper()
        if s and s not in merged:
            merged.append(s)
    for sym in _DYNAMIC_TICKERS:
        if sym not in merged:
            merged.append(sym)
    _DYNAMIC_TICKERS = merged[:40]


# Common intent phrases for completion
_INTENT_PHRASES = [
    "buy ", "sell ", "should I buy ", "should I sell ",
    "compare ", "analyze ", "what do you think about ",
    "show my portfolio", "show my positions", "show my orders",
    "best stock to buy", "trending stocks", "daily movers",
    "portfolio", "positions", "orders", "watchlist",
    "cancel order", "what is ", "watch ",
]


def _make_completer():
    """Return a readline completer that handles:
      - .command            tab-expands dot-commands
      - .model N|name       tab-expands numbered/named models
      - .q / .a / .watch X  tab-expands recent tickers, popular tickers, and company names
      - watch X below|above tab-expands watch direction and interval keywords
      - TICKER <suffix>     tab-expands common two-token shorthands (case-insensitive)
      - anything            falls back to intent phrases and dot-commands
    """
    model_labels = [m[1].lower() for m in MODELS]
    model_ids    = [m[0] for m in MODELS]
    model_nums   = [str(i) for i in range(1, len(MODELS) + 1)]
    ticker_sfx   = [
        "price", "quote", "options", "calls", "puts", "rsi", "macd",
        "earnings", "chart", "history", "fundamentals",
        "analyst", "ratings", "news", "risk", "below", "above",
    ]
    static_tickers = sorted(set(_NAME_TO_TICKER.values()))
    company_names  = sorted(k.lower() for k in _NAME_TO_TICKER.keys())
    watch_dirs     = ["below", "above"]

    def _ticker_candidates(prefix: str) -> list[str]:
        up = prefix.upper()
        low = prefix.lower()
        out: list[str] = []
        for sym in _DYNAMIC_TICKERS + static_tickers:
            if sym.startswith(up) and sym not in out:
                out.append(sym)
        for name in company_names:
            if name.startswith(low):
                mapped = _NAME_TO_TICKER.get(name.upper())
                if mapped and mapped not in out:
                    out.append(mapped)
                if name not in out:
                    out.append(name)
        return out

    def completer(text: str, state: int):
        buf  = readline.get_line_buffer()
        word = text  # the current incomplete word
        low_buf = buf.lower()

        # --- .model <N|name> completion ------------------------------------
        if low_buf.startswith(".model ") or low_buf.startswith(".m "):
            arg = buf.split(None, 1)[1] if len(buf.split(None, 1)) > 1 else ""
            candidates = model_nums + model_labels + model_ids
            matches = [c for c in candidates if c.lower().startswith(arg.lower())]
            return matches[state] if state < len(matches) else None

        # --- .watch / watch <TICKER> <below|above> completion --------------
        if low_buf.startswith((".watch ", "watch ")):
            parts = buf.split()
            trailing_space = buf.endswith(" ")
            # Completing the 2nd token (direction: below / above)
            if (len(parts) == 2 and trailing_space) or (len(parts) == 3 and not trailing_space):
                matches = [d for d in watch_dirs if d.startswith(word.lower())]
                return matches[state] if state < len(matches) else None
            # Completing the 1st token (ticker)
            if (len(parts) == 1 and trailing_space) or (len(parts) == 2 and not trailing_space):
                matches = _ticker_candidates(word)
                return matches[state] if state < len(matches) else None

        # --- .q / .quote / .a / .analyze / .risk <TICKER> completion -------
        if low_buf.startswith((".q ", ".quote ", ".a ", ".analyze ", ".risk ", "analyze ", "compare ", "buy ", "sell ")):
            clean_word = word.lstrip(",")
            matches = _ticker_candidates(clean_word)
            return matches[state] if state < len(matches) else None

        # --- dot-command completion ----------------------------------------
        if buf.startswith("."):
            matches = [c for c in _DOT_COMMANDS if c.startswith(low_buf)]
            return matches[state] if state < len(matches) else None

        # --- TICKER <suffix> completion (case-insensitive) -----------------
        parts = buf.split()
        if parts:
            first_up = parts[0].upper()
            is_known_sym = (
                (len(first_up) <= 5 and first_up.isalpha())
                or first_up in _NAME_TO_TICKER
            )
            if is_known_sym and ((len(parts) == 1 and buf.endswith(" ")) or (len(parts) == 2 and not buf.endswith(" "))):
                matches = [s for s in ticker_sfx if s.startswith(word.lower())]
                return matches[state] if state < len(matches) else None

        # --- beginning of line: intent phrases + dot-commands + tickers ----
        if not buf or word == buf:
            matches = [
                p for p in _INTENT_PHRASES + _DOT_COMMANDS
                if p.lower().startswith(word.lower())
            ]
            if word:
                for t in _ticker_candidates(word):
                    if t not in matches:
                        matches.append(t)
            return matches[state] if state < len(matches) else None

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


def read_input() -> str:
    """Read one logical line from stdin.

    Multi-line: a line ending with \\ continues on the next prompt.
    The continuation prompt aligns under the chevron.
    The joined result is added to history as a single entry.
    """
    lines: list[str] = []
    while True:
        if lines:
            prompt = PROMPT_CONT
        else:
            # Printed rather than embedded in the prompt: a newline inside the
            # prompt string throws off readline's cursor arithmetic.
            print()
            prompt = PROMPT_MAIN
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
