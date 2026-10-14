"""Saved session files under ~/.agenthood/sessions."""
from __future__ import annotations

import json
import time
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table

from ..config import SESS_DIR as _SESS_DIR
from .theme import C_ACCENT, C_DIM, C_DIMMER, C_FG, C_LABEL, console


def list_sessions() -> list[str]:
    """Return sorted list of saved session names."""
    _SESS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in _SESS_DIR.glob("*.json"))


def save_session_state(sess_file: Path, session_name: str, turn: int, agent) -> None:
    """Persist session metadata, multi-turn history, and active tickers to disk."""
    sess_file.parent.mkdir(parents=True, exist_ok=True)
    history = getattr(agent, "_history", None) or []
    recent_tickers = getattr(agent, "_recent_tickers", None) or []
    cfg = getattr(agent, "config", None)
    model = getattr(cfg, "model", "auto") if cfg else "auto"

    serialised_history = [
        [str(r), str(t)]
        for r, t in history[-40:]
        if isinstance(r, str) and isinstance(t, str)
    ]
    payload = {
        "name": session_name,
        "turns": int(turn),
        "model": str(model or "auto"),
        "history": serialised_history,
        "recent_tickers": [str(s) for s in recent_tickers[-10:]],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sess_file.write_text(json.dumps(payload, indent=2))


def load_session_state(sess_file: Path, agent) -> int:
    """Restore history, recent tickers, and turn count from *sess_file* onto *agent*."""
    if not sess_file or not sess_file.exists():
        return 0
    try:
        data = json.loads(sess_file.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0

    raw_hist = data.get("history")
    if isinstance(raw_hist, list):
        restored = [
            (str(item[0]), str(item[1]))
            for item in raw_hist
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]
        agent._history = restored[-40:]

    raw_tickers = data.get("recent_tickers")
    if isinstance(raw_tickers, list):
        agent._recent_tickers = [str(s).upper() for s in raw_tickers if str(s).strip()][-10:]

    turns = data.get("turns", 0)
    return int(turns) if isinstance(turns, int) else 0


def print_sessions() -> None:
    sessions = list_sessions()
    if not sessions:
        console.print(f"  [{C_DIM}]No saved sessions.  Use --session NAME to start one.[/{C_DIM}]")
        return
    t = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {C_DIM}",
              padding=(0, 2))
    t.add_column("Name",     style=C_FG,     no_wrap=True)
    t.add_column("Turns",    style=C_DIM,    no_wrap=True, justify="right")
    t.add_column("Ticker",   style=C_ACCENT, no_wrap=True)
    t.add_column("Modified", style=C_DIMMER, no_wrap=True)
    for name in sessions:
        p = _SESS_DIR / f"{name}.json"
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        turns_str = "-"
        ticker_str = "-"
        try:
            meta = json.loads(p.read_text())
            if isinstance(meta, dict):
                turns_str = str(meta.get("turns", "-"))
                rt = meta.get("recent_tickers") or []
                if rt:
                    ticker_str = str(rt[-1])
        except Exception:
            pass
        t.add_row(name, turns_str, ticker_str, mtime)
    console.print(Panel(t, border_style=C_DIM,
                        title=f"[{C_LABEL}]sessions[/{C_LABEL}]", title_align="left"))


def session_path(name: str) -> Path:
    _SESS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESS_DIR / f"{name}.json"

