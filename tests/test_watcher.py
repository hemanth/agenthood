"""
tests/test_watcher.py
~~~~~~~~~~~~~~~~~~~~~
Unit + integration tests for the price-watcher feature.

Coverage:
  - parse_watch_prompt()  —  all natural-language patterns
  - PriceWatcher.run()    —  trigger / no-trigger / cancel / error recovery
  - _direct_dispatch()    —  watch sentinel routed correctly
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agenthood.watcher import PriceWatcher, parse_watch_prompt


# ─────────────────────────────────────────────────────────────────────────────
# parse_watch_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestParseWatchPrompt:
    # ── buy / sell ────────────────────────────────────────────────────────────

    def test_buy_below_dollar(self):
        spec = parse_watch_prompt("watch AAPL and buy 10 when it drops below $180")
        assert spec is not None
        assert spec["ticker"]    == "AAPL"
        assert spec["threshold"] == 180.0
        assert spec["direction"] == "below"
        assert spec["action"]    == "buy"
        assert spec["shares"]    == 10

    def test_buy_below_no_dollar(self):
        spec = parse_watch_prompt("buy 5 NVDA if price falls below 800")
        assert spec is not None
        assert spec["ticker"]    == "NVDA"
        assert spec["threshold"] == 800.0
        assert spec["direction"] == "below"
        assert spec["action"]    == "buy"
        assert spec["shares"]    == 5

    def test_sell_above(self):
        spec = parse_watch_prompt("sell 20 MSFT when above $450")
        assert spec is not None
        assert spec["ticker"]    == "MSFT"
        assert spec["threshold"] == 450.0
        assert spec["direction"] == "above"
        assert spec["action"]    == "sell"
        assert spec["shares"]    == 20

    def test_buy_above_rises(self):
        spec = parse_watch_prompt("buy 3 TSLA when it rises above $250")
        assert spec is not None
        assert spec["ticker"]    == "TSLA"
        assert spec["threshold"] == 250.0
        assert spec["direction"] == "above"
        assert spec["action"]    == "buy"
        assert spec["shares"]    == 3

    # ── alert ─────────────────────────────────────────────────────────────────

    def test_alert_hits(self):
        spec = parse_watch_prompt("alert me when TSLA hits $250")
        assert spec is not None
        assert spec["ticker"]    == "TSLA"
        assert spec["threshold"] == 250.0
        assert spec["direction"] == "above"
        assert spec["action"]    == "alert"
        assert spec["shares"]    == 0

    def test_alert_drops_below(self):
        spec = parse_watch_prompt("notify me if BTC drops below 80000")
        assert spec is not None
        assert spec["ticker"]    == "BTC"
        assert spec["threshold"] == 80000.0
        assert spec["direction"] == "below"
        assert spec["action"]    == "alert"

    def test_watch_bare(self):
        spec = parse_watch_prompt("watch BTC below 80000")
        assert spec is not None
        assert spec["ticker"]    == "BTC"
        assert spec["threshold"] == 80000.0
        assert spec["direction"] == "below"

    # ── interval parsing ──────────────────────────────────────────────────────

    def test_interval_minutes(self):
        spec = parse_watch_prompt("watch AAPL every 5 minutes buy 10 below $180")
        assert spec is not None
        assert spec["interval"] == 300

    def test_interval_seconds(self):
        spec = parse_watch_prompt("watch AAPL every 10 sec below $180")
        assert spec is not None
        assert spec["interval"] == 10

    def test_interval_hours(self):
        spec = parse_watch_prompt("alert when AAPL drops below $170 every 1 hour")
        assert spec is not None
        assert spec["interval"] == 3600

    def test_interval_default(self):
        spec = parse_watch_prompt("watch AAPL below $180")
        assert spec is not None
        assert spec["interval"] == 30

    # ── comma-formatted price ─────────────────────────────────────────────────

    def test_comma_price(self):
        spec = parse_watch_prompt("alert me when AMZN drops below $1,800")
        assert spec is not None
        assert spec["threshold"] == 1800.0

    # ── non-matching prompts ──────────────────────────────────────────────────

    def test_no_match_regular_quote(self):
        assert parse_watch_prompt("AAPL") is None

    def test_no_match_portfolio(self):
        assert parse_watch_prompt("show my portfolio") is None

    def test_no_match_joke(self):
        assert parse_watch_prompt("tell me a joke") is None

    def test_no_ticker(self):
        # "below $100" with no recognisable ticker
        assert parse_watch_prompt("when it drops below $100") is None

    def test_no_price(self):
        assert parse_watch_prompt("watch AAPL until it drops") is None

    def test_watchlist_not_matched(self):
        # "my watchlist" must NOT be parsed as a price watch
        assert parse_watch_prompt("show my watchlist") is None


# ─────────────────────────────────────────────────────────────────────────────
# PriceWatcher.run()
# ─────────────────────────────────────────────────────────────────────────────

def _make_watcher(
    *,
    ticker="AAPL",
    threshold=180.0,
    direction="below",
    action="alert",
    shares=0,
    interval=0,      # 0 = no sleep between polls in tests
    prices: list,    # prices returned on successive polls
    trigger_cb=None,
) -> tuple[PriceWatcher, list]:
    """Helper: build a PriceWatcher with a scripted get_price sequence."""
    calls: list[float] = []
    triggered: list[tuple] = []

    price_iter = iter(prices)

    async def _get_price(sym: str) -> float:
        p = next(price_iter)
        calls.append(p)
        return p

    async def _on_trigger(w, p):
        triggered.append((w, p))
        if trigger_cb:
            trigger_cb(w, p)

    w = PriceWatcher(
        ticker=ticker, threshold=threshold, direction=direction,
        action=action, shares=shares, interval=interval,
        get_price=_get_price, on_trigger=_on_trigger,
    )
    return w, triggered


class TestPriceWatcher:

    @pytest.mark.asyncio
    async def test_triggers_below(self):
        """Price drops below threshold → triggers on first matching poll."""
        w, triggered = _make_watcher(
            direction="below", threshold=180.0,
            prices=[185.0, 182.0, 179.5],
        )
        await w.run()
        assert w.triggered
        assert len(triggered) == 1
        _, p = triggered[0]
        assert p == pytest.approx(179.5)

    @pytest.mark.asyncio
    async def test_triggers_above(self):
        """Price rises above threshold → triggers."""
        w, triggered = _make_watcher(
            direction="above", threshold=200.0,
            prices=[195.0, 198.0, 201.0],
        )
        await w.run()
        assert w.triggered
        assert triggered[0][1] == pytest.approx(201.0)

    @pytest.mark.asyncio
    async def test_triggers_exactly_at_threshold(self):
        """Price exactly equal to threshold counts as triggered."""
        w, triggered = _make_watcher(
            direction="below", threshold=180.0,
            prices=[181.0, 180.0],
        )
        await w.run()
        assert w.triggered

    @pytest.mark.asyncio
    async def test_no_trigger_while_above(self):
        """Price stays above threshold for below-watcher -- never triggers."""
        poll_count = 0

        async def _get_price(sym: str) -> float:
            nonlocal poll_count
            prices = [190.0, 185.0, 182.0]  # never <= 180
            if poll_count < len(prices):
                p = prices[poll_count]
                poll_count += 1
                return p
            # After all prices exhausted, stall until cancelled
            await asyncio.sleep(999)
            return 999.0  # unreachable

        triggered: list = []

        async def _on_trigger(w, p):
            triggered.append((w, p))

        w = PriceWatcher(
            ticker="AAPL", threshold=180.0, direction="below",
            action="alert", shares=0, interval=0,
            get_price=_get_price, on_trigger=_on_trigger,
        )
        task = asyncio.create_task(w.run())
        # Let it poll through all 3 prices
        for _ in range(30):
            await asyncio.sleep(0.01)
            if poll_count >= 3:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert not w.triggered
        assert triggered == []
        assert poll_count == 3  # prove all 3 prices were actually checked

    @pytest.mark.asyncio
    async def test_cancelled_before_trigger(self):
        """Cancelling the task stops the run loop cleanly — no CancelledError escapes."""
        call_count = 0

        async def _slow_price(_):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(10)   # long enough that cancel fires first
            return 200.0

        triggered = []

        async def _on_trigger(w, p):
            triggered.append(p)

        w = PriceWatcher(
            ticker="AAPL", threshold=180.0, direction="below",
            action="alert", shares=0, interval=10,
            get_price=_slow_price, on_trigger=_on_trigger,
        )
        task = asyncio.create_task(w.run())
        await asyncio.sleep(0)   # let task start the first poll (sleeping)
        task.cancel()
        # Watcher catches CancelledError internally and returns cleanly —
        # the task finishes normally (no exception propagates).
        await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        assert not w.triggered
        assert triggered == []

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """A poll error is swallowed and polling continues."""
        call_count = 0

        async def _flaky_price(_):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("timeout")
            return 175.0   # below 180 on 2nd attempt → trigger

        triggered = []

        async def _on_trigger(w, p):
            triggered.append(p)

        w = PriceWatcher(
            ticker="AAPL", threshold=180.0, direction="below",
            action="alert", shares=0, interval=0,
            get_price=_flaky_price, on_trigger=_on_trigger,
        )
        await w.run()
        assert w.triggered
        assert triggered == [pytest.approx(175.0)]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_triggered_flag_set(self):
        w, _ = _make_watcher(direction="below", threshold=100.0, prices=[90.0])
        assert not w.triggered
        await w.run()
        assert w.triggered

    @pytest.mark.asyncio
    async def test_trigger_not_fired_twice(self):
        """Watcher exits after first trigger — on_trigger called exactly once."""
        w, triggered = _make_watcher(
            direction="below", threshold=100.0,
            prices=[90.0, 85.0, 80.0],
        )
        await w.run()
        assert len(triggered) == 1

    def test_label_alert(self):
        w, _ = _make_watcher(
            ticker="AAPL", threshold=180.0, direction="below",
            action="alert", shares=0, prices=[],
        )
        assert "AAPL" in w.label
        assert "below" in w.label
        assert "180" in w.label

    def test_label_buy(self):
        w, _ = _make_watcher(
            ticker="TSLA", threshold=200.0, direction="above",
            action="buy", shares=5, prices=[],
        )
        assert "buy" in w.label
        assert "5" in w.label
        assert "TSLA" in w.label


# ─────────────────────────────────────────────────────────────────────────────
# Integration: _direct_dispatch returns __watch__ sentinel
# ─────────────────────────────────────────────────────────────────────────────

class TestWatchDispatch:
    """Tests that agent._direct_dispatch() returns the __watch__ sentinel."""

    @pytest.fixture
    async def hood(self):
        """Minimal AgentHood with mocked internals."""
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
        from agenthood.agent import AgentHood
        hood = object.__new__(AgentHood)
        hood._history = []
        hood.config = MagicMock()
        return hood

    @pytest.mark.asyncio
    async def test_watch_buy_below_returns_sentinel(self, hood):
        result = await hood._direct_dispatch(
            "watch AAPL and buy 10 when it drops below $180"
        )
        assert isinstance(result, dict)
        assert "__watch__" in result
        spec = result["__watch__"]
        assert spec["ticker"]    == "AAPL"
        assert spec["action"]    == "buy"
        assert spec["shares"]    == 10
        assert spec["threshold"] == 180.0
        assert spec["direction"] == "below"

    @pytest.mark.asyncio
    async def test_watch_alert_above_returns_sentinel(self, hood):
        result = await hood._direct_dispatch("alert when TSLA hits $300")
        assert isinstance(result, dict)
        assert "__watch__" in result
        spec = result["__watch__"]
        assert spec["ticker"]    == "TSLA"
        assert spec["action"]    == "alert"
        assert spec["threshold"] == 300.0
        assert spec["direction"] == "above"

    @pytest.mark.asyncio
    async def test_non_watch_returns_none_or_string(self, hood):
        """Regular ticker prompt should NOT return a __watch__ dict."""
        # We can't run full dispatch without MCP, so just verify watch
        # detection doesn't fire on a plain ticker
        from agenthood.watcher import parse_watch_prompt
        assert parse_watch_prompt("AAPL") is None
        assert parse_watch_prompt("show my portfolio") is None
        assert parse_watch_prompt("buy 5 AAPL at $180") is None  # no watch keyword

    @pytest.mark.asyncio
    async def test_with_custom_interval(self, hood):
        result = await hood._direct_dispatch(
            "watch NVDA every 5 minutes below $900"
        )
        assert isinstance(result, dict)
        assert "__watch__" in result
        assert result["__watch__"]["interval"] == 300

    @pytest.mark.asyncio
    async def test_watchlist_not_a_watcher(self):
        """'watchlist' must not be parsed as a price watch."""
        from agenthood.watcher import parse_watch_prompt
        assert parse_watch_prompt("show my watchlist") is None
        assert parse_watch_prompt("my watchlist") is None


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases for parse_watch_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestParseEdgeCases:

    def test_crypto_btc(self):
        spec = parse_watch_prompt("alert me when BTC drops below 80000")
        assert spec is not None
        assert spec["ticker"] == "BTC"
        assert spec["threshold"] == 80000.0

    def test_eth(self):
        spec = parse_watch_prompt("buy 2 ETH when below 3500")
        assert spec is not None
        assert spec["ticker"] == "ETH"
        assert spec["shares"] == 2

    def test_case_insensitive_keywords(self):
        spec = parse_watch_prompt("WATCH AAPL AND BUY 10 WHEN IT DROPS BELOW $180")
        assert spec is not None
        assert spec["ticker"]  == "AAPL"
        assert spec["action"]  == "buy"
        assert spec["shares"]  == 10

    def test_decimal_threshold(self):
        spec = parse_watch_prompt("alert when AAPL drops below $179.99")
        assert spec is not None
        assert spec["threshold"] == pytest.approx(179.99)

    def test_large_price_with_comma(self):
        spec = parse_watch_prompt("buy 1 AMZN when below $1,800.00")
        assert spec is not None
        assert spec["threshold"] == pytest.approx(1800.0)


# ── Interval must not be mistaken for the price ───────────────────────────────

class TestIntervalDoesNotEatPrice:
    """Regression: the threshold was the LAST number in the string.

    'watch AAPL below $180 every 5 minutes' produced threshold=5.0, so the
    watcher fired immediately on a documented syntax.
    """

    def test_minutes_suffix(self):
        s = parse_watch_prompt("watch AAPL below $180 every 5 minutes")
        assert s["threshold"] == 180.0
        assert s["interval"] == 300

    def test_seconds_suffix(self):
        s = parse_watch_prompt("watch BTC below 80000 every 30 seconds")
        assert s["threshold"] == 80000.0
        assert s["interval"] == 30

    def test_hour_suffix(self):
        s = parse_watch_prompt("sell 20 MSFT when above $450 every 1 hour")
        assert s["threshold"] == 450.0
        assert s["interval"] == 3600

    def test_short_unit(self):
        s = parse_watch_prompt("buy 5 NVDA if price falls below 800 every 2m")
        assert s["threshold"] == 800.0
        assert s["interval"] == 120

    def test_no_interval_keeps_default(self):
        s = parse_watch_prompt("watch AAPL below $180")
        assert s["threshold"] == 180.0
        assert s["interval"] == 30


# ── Intent refinements: questions, stopwords, company names, pronoun context ──

class TestWatchIntentRefinements:

    def test_questions_are_not_watch_prompts(self):
        assert parse_watch_prompt("is AAPL above $200?") is None
        assert parse_watch_prompt("why did NVDA drop below $100?") is None
        assert parse_watch_prompt("what happens if TSLA falls below $200?") is None
        assert parse_watch_prompt("did MSFT go above $400?") is None
        assert parse_watch_prompt("which stocks are below $50?") is None

    def test_polite_requests_still_parse_as_watches(self):
        s1 = parse_watch_prompt("can you watch AAPL below $180?")
        assert s1 is not None and s1["ticker"] == "AAPL" and s1["threshold"] == 180.0
        s2 = parse_watch_prompt("please alert me when TSLA hits $250")
        assert s2 is not None and s2["ticker"] == "TSLA" and s2["threshold"] == 250.0

    def test_stopwords_do_not_shadow_real_ticker(self):
        s1 = parse_watch_prompt("when price of AAPL drops below $180")
        assert s1 is not None and s1["ticker"] == "AAPL"
        s2 = parse_watch_prompt("buy 10 shares of NVDA when below 800")
        assert s2 is not None and s2["ticker"] == "NVDA" and s2["shares"] == 10
        s3 = parse_watch_prompt("alert me if stock price of MSFT falls under 400")
        assert s3 is not None and s3["ticker"] == "MSFT"

    def test_company_names_resolve_to_tickers_in_watch(self):
        s1 = parse_watch_prompt("watch Apple below $180")
        assert s1 is not None and s1["ticker"] == "AAPL"
        s2 = parse_watch_prompt("alert me when Nvidia hits $150")
        assert s2 is not None and s2["ticker"] == "NVDA"
        s3 = parse_watch_prompt("notify me if Bitcoin drops below 80000")
        assert s3 is not None and s3["ticker"] == "BTC"

    def test_pronoun_watch_uses_context_ticker(self):
        s = parse_watch_prompt("watch it below $180", context_ticker="NVDA")
        assert s is not None
        assert s["ticker"] == "NVDA"
        assert s["threshold"] == 180.0
        assert s["direction"] == "below"

