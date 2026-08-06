"""Shared console and colour tokens for the agenthood TUI.

No emojis. No em-dashes. ASCII via Rich only.
"""
from __future__ import annotations

from rich.console import Console

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

_PROVIDER_COL = {"openrouter": C_ACCENT,  "ollama": C_GREEN}
_SPEED_COL    = {"fast": C_GREEN, "medium": C_WARN, "slow": C_ERR}


# ── Readline prompts ──────────────────────────────────────────────────────────
# Readline counts every byte of the prompt when working out where the cursor
# sits, so raw ANSI makes it believe the prompt is ~15 columns wider than it
# looks. Recalling history then redraws over the prompt and mangles the line.
# \001 and \002 mark spans readline must skip when measuring width.

def invisible(seq: str) -> str:
    """Wrap a non-printing escape sequence so readline excludes it from width."""
    return f"\001{seq}\002"


_ANSI_BLUE  = "\x1b[38;5;75m"
_ANSI_GREY  = "\x1b[90m"
_ANSI_RESET = "\x1b[0m"

# Deliberately carry no leading newline: a newline inside the prompt breaks the
# same width calculation. Callers print the blank line themselves.
PROMPT_MAIN = f"{invisible(_ANSI_BLUE)}❯{invisible(_ANSI_RESET)}  "
PROMPT_CONT = f"  {invisible(_ANSI_GREY)}·{invisible(_ANSI_RESET)}  "
