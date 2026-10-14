"""In-REPL price watchers: start, poll, notify, and confirm triggered orders."""
from __future__ import annotations

import asyncio
import sys

from rich.markup import escape

from ..watcher import PriceWatcher, extract_quote_price
from .render import _show_table, _strip_emoji, render_response
from .theme import C_DIM, C_ERR, C_WARN, console


class WatchController:
    """Owns the background watcher tasks and their trigger queue."""

    def __init__(self, agent):
        self._agent = agent
        self._tasks: list[asyncio.Task] = []
        self._watchers: list[PriceWatcher] = []
        self._queue: asyncio.Queue = asyncio.Queue()

    @property
    def active_count(self) -> int:
        return sum(
            1 for w, t in zip(self._watchers, self._tasks)
            if not w.cancelled and not w.triggered and not t.done()
        )

    # -- price source ---------------------------------------------------------

    async def _price_fn(self, ticker: str) -> float:
        """Fetch the latest trade price for a ticker via the Robinhood quote tool."""
        raw = await self._agent._call_tool_direct(
            "robinhood_get_equity_quotes", {"symbols": [ticker.upper()]}
        )
        price = extract_quote_price(raw)
        if price is None:
            raise ValueError(f"Cannot parse price for {ticker} from: {str(raw)[:200]!r}")
        return price

    async def _on_trigger(self, watcher, price) -> None:
        await self._queue.put((watcher, price))

    # -- lifecycle ------------------------------------------------------------

    def start(self, spec: dict) -> None:
        """Spawn a background watcher from a parsed watch spec."""
        w = PriceWatcher(
            ticker     = spec["ticker"],
            threshold  = spec["threshold"],
            direction  = spec["direction"],
            action     = spec["action"],
            shares     = spec["shares"],
            interval   = spec.get("interval", 30),
            get_price  = self._price_fn,
            on_trigger = self._on_trigger,
        )
        self._watchers.append(w)
        self._tasks.append(asyncio.create_task(w.run()))
        dir_word   = "below" if w.direction == "below" else "above"
        action_str = (
            "alert you"
            if w.action == "alert"
            else f"{w.action} {w.shares} shares"
        )
        render_response(
            f"Watching {w.ticker} -- will {action_str} when price goes "
            f"{dir_word} ${w.threshold:,.2f}.  Polling every {w.interval}s.",
            0.0,
        )

    def show_watches(self) -> None:
        """Display active in-REPL and daemon watches."""
        from ..daemon import list_watches
        rows: dict[str, str] = {}
        idx = 0
        for w, t in zip(self._watchers, self._tasks):
            if not w.cancelled and not w.triggered and not t.done():
                idx += 1
                rows[f"repl #{idx}"] = f"{w.label} (every {w.interval}s)"
        for dw in list_watches():
            wid = dw.get("id", "?")
            act = dw.get("action", "alert")
            sh = dw.get("shares", 0)
            sym = dw.get("ticker", "?")
            direction = dw.get("direction", "below")
            thresh = dw.get("threshold", 0)
            act_str = "alert" if act == "alert" else f"{act} {sh}"
            rows[f"daemon #{wid}"] = f"{act_str} {sym} when {direction} ${float(thresh):,.2f}"
        if not rows:
            console.print(f"  [{C_DIM}]No active watches. Example: watch NVDA below $180[/{C_DIM}]")
            return
        _show_table(rows, "active watches")

    def cancel_all(self) -> None:
        for w in self._watchers:
            w.cancelled = True
        for t in self._tasks:
            t.cancel()

    # -- notifications --------------------------------------------------------

    async def drain_while_waiting(self, input_future: asyncio.Future) -> None:
        """Print triggered watcher notifications while readline is blocking."""
        while not input_future.done():
            try:
                w_obj, w_price = self._queue.get_nowait()
                # print notification above current line
                sys.stdout.write('\n')
                sys.stdout.write(f'  [!] WATCHER: {w_obj.ticker} hit ${w_price:,.2f}\n')
                sys.stdout.flush()
            except asyncio.QueueEmpty:
                pass
            await asyncio.sleep(0.5)

    async def drain_triggered(self) -> None:
        """Report every queued trigger, confirming any order it proposes."""
        while not self._queue.empty():
            try:
                w_obj, w_price = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            dir_word = "below" if w_obj.direction == "below" else "above"
            console.print()
            console.print(
                f"  [{C_WARN}]  WATCHER TRIGGERED  "
                f"{w_obj.ticker} hit ${w_price:,.2f} "
                f"(threshold {dir_word} ${w_obj.threshold:,.2f})[/{C_WARN}]"
            )
            if w_obj.action == "alert":
                continue

            console.print(
                f"  [{C_DIM}]  Proposed: "
                f"{w_obj.action} {w_obj.shares} {w_obj.ticker} "
                f"-- confirm? (yes / no)[/{C_DIM}]"
            )
            try:
                confirm = input("  confirm> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                confirm = "no"
            if confirm not in ("yes", "y"):
                console.print(f"  [{C_DIM}]  Order cancelled.[/{C_DIM}]")
                continue

            # Route through agent safety review
            order_prompt = (
                f"{w_obj.action} {w_obj.shares} {w_obj.ticker} "
                f"at market -- triggered by price watcher"
            )
            console.print(f"  [{C_DIM}]routing through safety review...[/{C_DIM}]")
            try:
                resp = await self._agent.chat(order_prompt)
                console.print()
                render_response(_strip_emoji(str(resp)), 0.0)
            except Exception as w_exc:
                console.print(f"  [{C_ERR}]Order error: {escape(str(w_exc))}[/{C_ERR}]")

