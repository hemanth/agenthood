"""Saved session files under ~/.agenthood/sessions."""
from __future__ import annotations

import time
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table

from ..config import SESS_DIR as _SESS_DIR
from .theme import C_DIM, C_DIMMER, C_FG, C_LABEL, console


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
