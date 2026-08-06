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
