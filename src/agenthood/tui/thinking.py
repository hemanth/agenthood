"""Transient live "thinking" panel: spinner + step trace."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .theme import C_ACCENT, C_DIM, C_DIMMER, C_LABEL, C_STEP, _SPIN


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
