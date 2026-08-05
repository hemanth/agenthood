"""
agenthood.watcher
~~~~~~~~~~~~~~~~~
Background price-watcher tasks.

Usage::

    watcher = PriceWatcher(
        ticker="AAPL",
        threshold=180.0,
        direction="below",      # "below" | "above"
        action="buy",           # "buy" | "sell" | "alert"
        shares=10,
        interval=30,            # seconds between polls
        on_trigger=callback,    # async callable(watcher, current_price)
        get_price=fetch_fn,     # async callable(ticker) -> float
    )
    task = asyncio.create_task(watcher.run())
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

GetPriceFn  = Callable[[str], Awaitable[float]]
OnTriggerFn = Callable[["PriceWatcher", float], Awaitable[None]]


@dataclass
class PriceWatcher:
    """Polls ``ticker`` every ``interval`` seconds until the price condition is met."""

    ticker:    str
    threshold: float
    direction: str          # "below" | "above"
    action:    str          # "buy" | "sell" | "alert"
    shares:    int          # 0 for alert-only
    interval:  int          # poll interval in seconds (default 30)
    get_price: GetPriceFn
    on_trigger: OnTriggerFn

    # internal
    started_at: float = field(default_factory=time.monotonic, init=False)
    cancelled:  bool  = field(default=False, init=False)
    triggered:  bool  = field(default=False, init=False)

    @property
    def label(self) -> str:
        sym = ("below" if self.direction == "below" else "above")
        if self.action == "alert":
            return f"alert when {self.ticker} {sym} ${self.threshold:,.2f}"
        return f"{self.action} {self.shares} {self.ticker} when {sym} ${self.threshold:,.2f}"

    async def run(self) -> None:
        """Poll loop — runs until triggered, cancelled, or an unrecoverable error."""
        logger.info("Watcher started: %s  interval=%ds", self.label, self.interval)
        while not self.cancelled and not self.triggered:
            try:
                price = await self.get_price(self.ticker)
                logger.debug("Watcher %s: current price $%.2f", self.ticker, price)

                hit = (
                    (self.direction == "below" and price <= self.threshold) or
                    (self.direction == "above" and price >= self.threshold)
                )
                if hit:
                    self.triggered = True
                    await self.on_trigger(self, price)
                    return

            except asyncio.CancelledError:
                logger.info("Watcher cancelled: %s", self.label)
                return
            except Exception as exc:
                logger.warning("Watcher poll error (%s): %s", self.ticker, exc)
                # Don't abort on transient errors — keep polling

            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                return

        logger.info("Watcher done: %s", self.label)


# ── Natural-language parser ────────────────────────────────────────────────────

_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')
_PRICE_RE  = re.compile(r'\$?([\d,]+(?:\.\d+)?)')

def parse_watch_prompt(prompt: str) -> Optional[dict]:
    """
    Parse a natural-language watch prompt into a watcher spec dict.

    Supported patterns (case-insensitive)::

        "watch AAPL and buy 10 when it drops below $180"
        "alert me when TSLA hits $250"
        "buy 5 NVDA if price falls below 800"
        "sell 20 MSFT when above $450"
        "watch BTC below 80000"

    Returns a dict with keys: ticker, threshold, direction, action, shares
    or None if the prompt cannot be parsed.
    """
    p = prompt.strip()
    pl = p.lower()

    # Must be a watch/alert/condition prompt
    _WATCH_KEYWORDS = (
        "watch", "alert", "notify", "when it", "when price", "if price",
        "if it", "drops below", "falls below", "rises above", "goes above",
        "hits ", "reaches ", "crosses",
        # bare "when above / below" and "if above / below"
        "when above", "when below", "if above", "if below",
        "above $", "below $",
    )
    if not any(kw in pl for kw in _WATCH_KEYWORDS):
        return None

    # Direction
    direction = "below"
    if any(kw in pl for kw in ("above", "rises", "goes above", "hits", "reaches")):
        direction = "above"
    if any(kw in pl for kw in ("below", "drops", "falls", "under")):
        direction = "below"

    # Action
    action = "alert"
    shares = 0
    buy_match  = re.search(r'\bbuy\s+(\d+)\b', pl)
    sell_match = re.search(r'\bsell\s+(\d+)\b', pl)
    if buy_match:
        action = "buy"
        shares = int(buy_match.group(1))
    elif sell_match:
        action = "sell"
        shares = int(sell_match.group(1))

    # Ticker — first uppercase word that looks like a ticker
    tickers = _TICKER_RE.findall(p)
    # Filter stop words that happen to be all-caps
    _STOP = {"I", "A", "THE", "AND", "OR", "IF", "WHEN", "BUY", "SELL",
              "WATCH", "ALERT", "NOTIFY", "ME", "MY", "IT", "AT"}
    tickers = [t for t in tickers if t not in _STOP]
    if not tickers:
        return None
    ticker = tickers[0]

    # Threshold price — last number in the string (most likely to be the price)
    prices = _PRICE_RE.findall(p)
    if not prices:
        return None
    try:
        threshold = float(prices[-1].replace(",", ""))
    except ValueError:
        return None

    # Optional poll interval: "every 5 minutes" / "every 30 seconds" / "every 2m"
    interval = 30
    iv_m = re.search(r'every\s+(\d+)\s*(s(?:ec)?|m(?:in)?|h(?:our)?)', pl)
    if iv_m:
        n, unit = int(iv_m.group(1)), iv_m.group(2)
        if unit.startswith("h"):
            interval = n * 3600
        elif unit.startswith("m"):
            interval = n * 60
        else:
            interval = n

    return dict(ticker=ticker, threshold=threshold, direction=direction,
                action=action, shares=shares, interval=interval)
