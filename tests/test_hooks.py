"""Tests for trading_guard — the pre_tool safety hook.

All tests are pure async unit tests using pytest-asyncio.
No MCP connection or LLM required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pytest_asyncio
from agentu import HookAction
from agenthood.hooks import trading_guard, _matches

# ── _matches helper ───────────────────────────────────────────────────────────

class TestMatches:
    def test_bare_tool_name(self):
        assert _matches("place_equity_order", {"place_equity_order"})

    def test_prefixed_tool_name(self):
        assert _matches("robinhood_place_equity_order", {"place_equity_order"})

    def test_no_match(self):
        assert not _matches("get_equity_quotes", {"place_equity_order"})

    def test_multi_prefix(self):
        assert _matches("rh_place_equity_order", {"place_equity_order"})

    def test_exact_match_in_set(self):
        assert _matches("robinhood_review_equity_order", {"review_equity_order"})


# ── Dry-run blocking ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestDryRunBlocking:

    async def test_blocks_place_equity_order(self):
        guard = trading_guard(dry_run=True, max_order_value=1000, allowed_symbols=[])
        result = await guard("robinhood_place_equity_order", {"symbol": "AAPL"}, None)
        assert result.action == HookAction.DENY
        assert "DRY RUN" in result.reason

    async def test_blocks_place_option_order(self):
        guard = trading_guard(dry_run=True, max_order_value=1000, allowed_symbols=[])
        result = await guard("robinhood_place_option_order", {}, None)
        assert result.action == HookAction.DENY

    async def test_blocks_exercise_option(self):
        guard = trading_guard(dry_run=True, max_order_value=1000, allowed_symbols=[])
        result = await guard("robinhood_exercise_option", {}, None)
        assert result.action == HookAction.DENY

    async def test_allows_read_tools_in_dry_run(self):
        guard = trading_guard(dry_run=True, max_order_value=1000, allowed_symbols=[])
        result = await guard("robinhood_get_equity_quotes", {"symbols": ["AAPL"]}, None)
        assert result.action == HookAction.ALLOW

    async def test_allows_review_in_dry_run(self):
        guard = trading_guard(dry_run=True, max_order_value=1000, allowed_symbols=[])
        result = await guard("robinhood_review_equity_order",
                             {"symbol": "AAPL", "quantity": 1, "price": 100}, None)
        assert result.action == HookAction.ALLOW

    async def test_dry_run_false_allows_place(self):
        guard = trading_guard(dry_run=False, max_order_value=1000, allowed_symbols=[])
        result = await guard("robinhood_place_equity_order",
                             {"symbol": "AAPL", "quantity": 1, "price": 100}, None)
        assert result.action == HookAction.ALLOW


# ── Notional cap ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestNotionalCap:

    async def test_blocks_over_cap(self):
        guard = trading_guard(dry_run=False, max_order_value=500, allowed_symbols=[])
        result = await guard("place_equity_order",
                             {"quantity": 10, "price": 100}, None)  # $1000 > $500
        assert result.action == HookAction.DENY
        assert "500" in result.reason

    async def test_allows_at_cap(self):
        guard = trading_guard(dry_run=False, max_order_value=500, allowed_symbols=[])
        result = await guard("place_equity_order",
                             {"quantity": 5, "price": 100}, None)  # $500 = cap
        assert result.action == HookAction.ALLOW

    async def test_allows_below_cap(self):
        guard = trading_guard(dry_run=False, max_order_value=500, allowed_symbols=[])
        result = await guard("place_equity_order",
                             {"quantity": 1, "price": 50}, None)  # $50 < $500
        assert result.action == HookAction.ALLOW

    async def test_dollar_amount_field(self):
        guard = trading_guard(dry_run=False, max_order_value=100, allowed_symbols=[])
        result = await guard("place_equity_order",
                             {"dollar_amount": 200}, None)  # $200 > $100
        assert result.action == HookAction.DENY

    async def test_zero_cap_means_no_cap(self):
        guard = trading_guard(dry_run=False, max_order_value=0, allowed_symbols=[])
        result = await guard("place_equity_order",
                             {"quantity": 1000, "price": 1000}, None)
        assert result.action == HookAction.ALLOW

    async def test_no_price_no_block(self):
        guard = trading_guard(dry_run=False, max_order_value=500, allowed_symbols=[])
        result = await guard("place_equity_order", {"quantity": 10}, None)
        assert result.action == HookAction.ALLOW  # price=0 → notional=0

    async def test_limit_price_field_used(self):
        guard = trading_guard(dry_run=False, max_order_value=100, allowed_symbols=[])
        result = await guard("place_equity_order",
                             {"quantity": 2, "limit_price": 100}, None)  # $200 > $100
        assert result.action == HookAction.DENY


# ── Symbol allowlist ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSymbolAllowlist:

    async def test_blocks_unlisted_symbol_on_order(self):
        guard = trading_guard(dry_run=False, max_order_value=9999,
                              allowed_symbols=["AAPL", "MSFT"])
        result = await guard("place_equity_order",
                             {"symbol": "NVDA", "quantity": 1, "price": 1}, None)
        assert result.action == HookAction.DENY
        assert "NVDA" in result.reason

    async def test_allows_listed_symbol(self):
        guard = trading_guard(dry_run=False, max_order_value=9999,
                              allowed_symbols=["AAPL", "MSFT"])
        result = await guard("place_equity_order",
                             {"symbol": "AAPL", "quantity": 1, "price": 1}, None)
        assert result.action == HookAction.ALLOW

    async def test_case_insensitive_allowlist(self):
        guard = trading_guard(dry_run=False, max_order_value=9999,
                              allowed_symbols=["aapl"])
        result = await guard("place_equity_order",
                             {"symbol": "AAPL", "quantity": 1, "price": 1}, None)
        assert result.action == HookAction.ALLOW

    async def test_empty_allowlist_allows_all(self):
        guard = trading_guard(dry_run=False, max_order_value=9999, allowed_symbols=[])
        result = await guard("place_equity_order",
                             {"symbol": "DOGE", "quantity": 1, "price": 1}, None)
        assert result.action == HookAction.ALLOW

    async def test_allowlist_also_applies_to_review(self):
        guard = trading_guard(dry_run=False, max_order_value=9999,
                              allowed_symbols=["AAPL"])
        result = await guard("robinhood_review_equity_order",
                             {"symbol": "TSLA", "quantity": 1, "price": 1}, None)
        assert result.action == HookAction.DENY

    async def test_allowlist_does_not_apply_to_reads(self):
        guard = trading_guard(dry_run=False, max_order_value=9999,
                              allowed_symbols=["AAPL"])
        result = await guard("robinhood_get_equity_quotes",
                             {"symbols": ["TSLA"]}, None)
        assert result.action == HookAction.ALLOW

    async def test_underlying_symbol_field(self):
        guard = trading_guard(dry_run=False, max_order_value=9999,
                              allowed_symbols=["AAPL"])
        result = await guard("place_option_order",
                             {"underlying_symbol": "NVDA"}, None)
        assert result.action == HookAction.DENY

    async def test_chain_symbol_field(self):
        guard = trading_guard(dry_run=False, max_order_value=9999,
                              allowed_symbols=["AAPL"])
        result = await guard("place_option_order",
                             {"chain_symbol": "TSLA"}, None)
        assert result.action == HookAction.DENY


# ── Interaction: dry_run + allowlist + cap ────────────────────────────────────

@pytest.mark.asyncio
class TestCombinedRules:

    async def test_dry_run_wins_before_cap(self):
        """Dry-run check happens before cap check."""
        guard = trading_guard(dry_run=True, max_order_value=1,
                              allowed_symbols=["AAPL"])
        result = await guard("place_equity_order",
                             {"symbol": "AAPL", "quantity": 1, "price": 0.50}, None)
        assert result.action == HookAction.DENY
        assert "DRY RUN" in result.reason

    async def test_cancel_always_allowed(self):
        guard = trading_guard(dry_run=True, max_order_value=1, allowed_symbols=["X"])
        result = await guard("robinhood_cancel_equity_order", {"order_id": "123"}, None)
        assert result.action == HookAction.ALLOW

    async def test_read_tools_always_allowed(self):
        guard = trading_guard(dry_run=True, max_order_value=0, allowed_symbols=["X"])
        for tool in ["get_equity_quotes", "get_portfolio", "get_equity_historicals",
                     "get_option_chains", "get_equity_fundamentals",
                     "get_open_equity_orders", "get_equity_positions"]:
            result = await guard(tool, {}, None)
            assert result.action == HookAction.ALLOW, f"{tool} should be allowed"
