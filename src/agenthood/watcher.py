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
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# ── Types & Quote Price Extraction ────────────────────────────────────────────

GetPriceFn  = Callable[[str], Awaitable[float]]
OnTriggerFn = Callable[["PriceWatcher", float], Awaitable[None]]

# Order matters: the last actual trade is the truth, the rest are stand-ins for
# when a venue has not printed one yet.
_PRICE_KEYS = (
    "last_trade_price",
    "last_non_reg_trade_price",
    "bid_price",
    "ask_price",
)


def extract_quote_price(raw) -> float | None:
    """Pull the last trade price out of a Robinhood equity-quotes response.

    The payload is ``{"data": {"results": [{"quote": {...}, "close": {...}}]}}``.
    Reading the top level instead means never matching, which used to send every
    watcher poll down a regex scrape and then a full LLM round trip -- slow, and
    it could latch onto a bid, an ask, or an unrelated number.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None

    data = raw.get("data", raw)
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        for key in (*_PRICE_KEYS, "price"):
            val = (data if isinstance(data, dict) else {}).get(key)
            if val not in (None, ""):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None
    entry = results[0]
    if not isinstance(entry, dict):
        return None

    quote = entry.get("quote", entry)
    for key in _PRICE_KEYS:
        val = (quote or {}).get(key)
        if val not in (None, ""):
            try:
                return float(val)
            except (ValueError, TypeError):
                continue

    close = entry.get("close") or {}
    if close.get("price") not in (None, ""):
        try:
            return float(close["price"])
        except (ValueError, TypeError):
            pass
    return None


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
_INTERVAL_RE = re.compile(r'every\s+(\d+)\s*(s(?:ec\w*)?|m(?:in\w*)?|h(?:our\w*)?)\b')

_QUESTION_START_RE = re.compile(
    r"^(is|are|was|were|why|what|how|when\s+did|did|do|does|which|who|where|should\s+i)\b",
    re.IGNORECASE,
)
_POLITE_WATCH_RE = re.compile(
    r"^(can|could|would|will)\s+you\s+(please\s+)?(watch|alert|notify|buy|sell)\b",
    re.IGNORECASE,
)

_WATCH_STOPWORDS = {
    "AM", "AN", "AS", "AT", "BE", "BY", "DO", "IF", "IN", "IS",
    "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "THE", "AND", "FOR", "ARE", "WAS", "CAN", "YOU", "HIS", "HER", "OUR",
    "WATCH", "ALERT", "NOTIFY", "BUY", "SELL", "WHEN", "IT", "ITS", "THAT",
    "THIS", "DROPS", "DROP", "FALLS", "FALL", "RISES", "RISE", "GOES", "GO",
    "ABOVE", "BELOW", "UNDER", "OVER", "HITS", "HIT", "REACHES", "REACH",
    "CROSSES", "CROSS", "PRICE", "STOCK", "STOCKS", "SHARE", "SHARES",
    "EVERY", "UNTIL", "PLEASE", "COULD", "WOULD", "WILL", "WITH", "FROM",
    "SEC", "SECS", "MIN", "MINS", "HOUR", "HOURS",
}


def parse_watch_prompt(
    prompt: str,
    context_ticker: Optional[str] = None,
) -> Optional[dict]:
    """
    Parse a natural-language watch prompt into a watcher spec dict.

    Supported patterns (case-insensitive)::

        "watch AAPL and buy 10 when it drops below $180"
        "alert me when TSLA hits $250"
        "buy 5 NVDA if price falls below 800"
        "sell 20 MSFT when above $450"
        "watch BTC below 80000"
        "watch it below $180"  (with context_ticker="NVDA")

    Returns a dict with keys: ticker, threshold, direction, action, shares
    or None if the prompt cannot be parsed.
    """
    from .core.tickers import NAME_TO_TICKER

    p = prompt.strip()
    if not p:
        return None
    pl = p.lower()

    # Reject interrogative questions ("is AAPL above $200?", "why did NVDA drop below $100?")
    # unless phrased as a polite request ("can you watch AAPL below $180?").
    if _QUESTION_START_RE.match(pl) and not _POLITE_WATCH_RE.match(pl):
        return None
    if p.endswith("?") and not _POLITE_WATCH_RE.match(pl) and not pl.startswith(("watch ", "alert ", "notify ")):
        return None

    # Must be a watch/alert/condition prompt
    _WATCH_KEYWORDS = (
        "watch", "alert", "notify", "when it", "when price", "if price",
        "if it", "drops below", "falls below", "rises above", "goes above",
        "falls under", "drops under", "goes under",
        "hits ", "reaches ", "crosses",
        # bare "when above / below" and "if above / below"
        "when above", "when below", "if above", "if below",
        "when under", "if under",
        "above $", "below $", "under $",
    )
    if not any(kw in pl for kw in _WATCH_KEYWORDS):
        return None

    # Direction
    direction = "below"
    if any(kw in pl for kw in ("above", "rises", "goes above", "hits", "reaches", "over")):
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

    # Optional poll interval: "every 5 minutes" / "every 30 seconds" / "every 2m".
    # Parsed BEFORE the price and ticker so interval units don't pollute either.
    interval = 30
    iv_m = _INTERVAL_RE.search(pl)
    if iv_m:
        n, unit = int(iv_m.group(1)), iv_m.group(2)
        if unit.startswith("h"):
            interval = n * 3600
        elif unit.startswith("m"):
            interval = n * 60
        else:
            interval = n
        p = p[:iv_m.start()] + p[iv_m.end():]

    # Ticker resolution:
    # 1. Known company/crypto names (handles 6+ letter names like "Nvidia", "Bitcoin", and 5-letter "Apple")
    # 2. Explicitly uppercase 1-5 letter tokens in the original prompt
    # 3. Any remaining 1-5 letter non-stopword tokens
    # 4. Context ticker if prompt uses "it" / "that"
    ticker: Optional[str] = None
    words = [w.strip("?!.,;:$'\"()[]{}") for w in p.split()]
    words = [w for w in words if w]

    # Check known company names first
    for w in words:
        up = w.upper()
        if up in NAME_TO_TICKER and up not in _WATCH_STOPWORDS:
            ticker = NAME_TO_TICKER[up]
            break

    # Next, check words the user typed in uppercase
    if ticker is None:
        for w in words:
            up = w.upper()
            if w.isupper() and _TICKER_RE.fullmatch(up) and up not in _WATCH_STOPWORDS:
                ticker = NAME_TO_TICKER.get(up, up)
                break

    # Next, check any 1-5 letter non-stopword token
    if ticker is None:
        for t in _TICKER_RE.findall(p.upper()):
            if t not in _WATCH_STOPWORDS:
                ticker = NAME_TO_TICKER.get(t, t)
                break

    # Finally, check pronoun reference to context_ticker
    if ticker is None and context_ticker and re.search(r"\b(it|that)\b", pl):
        ticker = context_ticker.upper()

    if not ticker:
        return None

    # Threshold price — last number in what's left (most likely to be the price)
    prices = _PRICE_RE.findall(p)
    if not prices:
        return None
    try:
        threshold = float(prices[-1].replace(",", ""))
    except ValueError:
        return None

    return dict(ticker=ticker, threshold=threshold, direction=direction,
                action=action, shares=shares, interval=interval)

