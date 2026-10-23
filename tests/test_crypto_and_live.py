"""TDD tests for crypto trading, order preview/confirmation safety, and --live/.live mode."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agenthood.config import AgentHoodConfig
from agenthood.core.agent import AgentHood
from agenthood.core.formatting import _format_tool_result
from agenthood.core.toolargs import wrap_tool
from agenthood.hooks import HookAction, trading_guard
from agenthood.tui.commands import handle_meta
from agenthood.tui.format import _format_mcp_response


# ── 1. Safety hooks (hooks.py) ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trading_guard_blocks_place_crypto_order_in_dry_run():
    guard = trading_guard(dry_run=True, max_order_value=500.0)
    res = await guard(
        "robinhood_place_crypto_order",
        {"rhs_account_number": "802162008", "symbol": "DOGE", "side": "buy", "type": "limit", "dollar_amount": "1.00"},
    )
    assert res.action == HookAction.DENY
    assert "DRY RUN" in (res.reason or "")


@pytest.mark.asyncio
async def test_trading_guard_allows_preview_crypto_order_in_dry_run():
    guard = trading_guard(dry_run=True, max_order_value=500.0)
    res = await guard(
        "robinhood_preview_crypto_order",
        {"rhs_account_number": "802162008", "symbol": "DOGE", "side": "buy", "type": "limit", "dollar_amount": "1.00"},
    )
    assert res.action == HookAction.ALLOW


@pytest.mark.asyncio
async def test_trading_guard_enforces_max_order_value_on_dollar_amount():
    guard = trading_guard(dry_run=False, max_order_value=50.0)
    res = await guard(
        "robinhood_place_crypto_order",
        {"rhs_account_number": "802162008", "symbol": "BTC", "side": "buy", "type": "market", "dollar_amount": "100.00"},
    )
    assert res.action == HookAction.DENY
    assert "$100.00" in (res.reason or "")


@pytest.mark.asyncio
async def test_trading_guard_allowed_symbols_handles_crypto_pair_suffix():
    guard = trading_guard(dry_run=False, max_order_value=500.0, allowed_symbols=["DOGE", "BTC"])
    ok = await guard("robinhood_place_crypto_order", {"symbol": "DOGE-USD", "dollar_amount": "1.00"})
    assert ok.action == HookAction.ALLOW
    blocked = await guard("robinhood_place_crypto_order", {"symbol": "SHIB-USD", "dollar_amount": "1.00"})
    assert blocked.action == HookAction.DENY
    assert "SHIB" in (blocked.reason or "")


# ── 2. Crypto argument coercion (toolargs.py) ────────────────────────────────

@pytest.mark.asyncio
async def test_wrap_tool_normalizes_crypto_quote_symbols_and_rhs_account():
    captured_quotes = {}

    async def fake_quotes(**kwargs):
        captured_quotes.update(kwargs)
        return {"ok": True}

    t_quotes = MagicMock()
    t_quotes.name = "robinhood_get_crypto_quotes"
    t_quotes.parameters = {"symbols": "['array']: Pair symbols"}
    t_quotes.function = fake_quotes
    wrap_tool(t_quotes)

    await t_quotes.function(symbols=["DOGE", "BTC-USD", "ETHUSD"])
    assert captured_quotes["symbols"] == ["DOGE-USD", "BTC-USD", "ETHUSD"]

    captured_preview = {}

    async def fake_preview(**kwargs):
        captured_preview.update(kwargs)
        return {"ok": True}

    t_prev = MagicMock()
    t_prev.name = "robinhood_preview_crypto_order"
    t_prev.parameters = {
        "rhs_account_number": "string: Numeric brokerage account number",
        "symbol": "string: Crypto symbol",
    }
    t_prev.function = fake_preview
    wrap_tool(t_prev)

    await t_prev.function(account_number="802162008", symbol="DOGE")
    assert captured_preview["rhs_account_number"] == "802162008"
    assert "account_number" not in captured_preview


# ── 3. Crypto quote & preview formatting ─────────────────────────────────────

def test_format_crypto_quotes_sub_dollar_precision():
    raw = json.dumps({
        "data": {
            "results": [
                {
                    "symbol": "DOGEUSD",
                    "bid_price": "0.077871",
                    "ask_price": "0.079358",
                    "mark_price": "0.078615",
                    "open_price": "0.080295",
                    "routing": "Market Maker Routing",
                    "updated_at": "2026-10-22T19:49:41.268-04:00",
                }
            ]
        }
    })
    fmt1 = _format_tool_result("robinhood_get_crypto_quotes", raw)
    assert fmt1 is not None
    assert "DOGE" in fmt1
    assert "$0.0786" in fmt1

    fmt2 = _format_mcp_response(raw, "DOGE Quote")
    assert "DOGE" in fmt2
    assert "$0.0786" in fmt2
    assert "https://robinhood.com/crypto/DOGE" in fmt2


def test_format_crypto_order_preview():
    raw = json.dumps({
        "data": {
            "order": {
                "currency_code": "DOGE",
                "side": "buy",
                "type": "limit",
                "time_in_force": "gtc",
                "quantity": "12.59",
                "price": "0.0794",
                "routing": "Market Maker Routing",
                "net_rounded_estimated_notional": "1",
            },
            "estimated_fee": "0",
            "crypto_account_number": "311266774926",
            "customer_approval_required": False,
        }
    })
    fmt = _format_tool_result("robinhood_preview_crypto_order", raw)
    assert fmt is not None
    assert "DOGE" in fmt
    assert "12.59" in fmt
    assert "$0.0794" in fmt


# ── 4. TUI .live / .dry dot-commands ─────────────────────────────────────────

def test_handle_meta_live_and_dry_toggle():
    agent = AgentHood(AgentHoodConfig(dry_run=True))
    assert agent.config.dry_run is True

    res = handle_meta(".live", agent)
    assert res is True
    assert agent.config.dry_run is False

    res2 = handle_meta(".dry", agent)
    assert res2 is True
    assert agent.config.dry_run is True


# ── 5. End-to-End Crypto Preview -> Confirmation Flow in AgentHood.chat() ────

@pytest.mark.asyncio
async def test_chat_buy_crypto_previews_first_and_requires_confirmation():
    ah = AgentHood(AgentHoodConfig(dry_run=False))
    ah._agent = MagicMock()
    ah._agent.infer = AsyncMock(return_value={"result": "fallback"})

    calls = []

    async def fake_call_tool_direct(name, params):
        calls.append((name, params))
        if name == "robinhood_get_accounts":
            return json.dumps({
                "data": {
                    "accounts": [
                        {
                            "account_number": "802162008",
                            "rhs_account_number": "802162008",
                            "rhc_account_number": "311266774926",
                            "agentic_allowed": True,
                        }
                    ]
                }
            })
        if name == "robinhood_get_crypto_quotes":
            return json.dumps({
                "data": {
                    "results": [
                        {
                            "symbol": "DOGEUSD",
                            "bid_price": "0.0778",
                            "ask_price": "0.0794",
                            "mark_price": "0.0786",
                            "open_price": "0.0802",
                            "routing": "Market Maker Routing",
                        }
                    ]
                }
            })
        if name == "robinhood_preview_crypto_order":
            return json.dumps({
                "data": {
                    "order": {
                        "id": "prev-1",
                        "currency_code": params.get("symbol", "DOGE"),
                        "side": params.get("side", "buy"),
                        "type": params.get("type", "limit"),
                        "time_in_force": "gtc",
                        "quantity": "12.59",
                        "price": params.get("limit_price", "0.0794"),
                        "routing": "Market Maker Routing",
                        "net_rounded_estimated_notional": params.get("dollar_amount", "1.00"),
                    },
                    "estimated_fee": "0",
                    "crypto_account_number": "311266774926",
                    "customer_approval_required": False,
                }
            })
        if name == "robinhood_place_crypto_order":
            return json.dumps({
                "data": {
                    "order": {
                        "id": "live-order-123",
                        "currency_code": params.get("symbol", "DOGE"),
                        "side": params.get("side", "buy"),
                        "type": params.get("type", "limit"),
                        "state": "confirmed",
                        "quantity": "12.59",
                        "price": params.get("limit_price", "0.0794"),
                        "net_rounded_estimated_notional": params.get("dollar_amount", "1.00"),
                    }
                }
            })
        raise ValueError(f"Unexpected tool: {name}")

    ah._call_tool_direct = fake_call_tool_direct

    # Turn 1: User says "just buy a cryto coin" -> MUST preview, NEVER place!
    reply1 = await ah.chat("just buy a cryto coin")
    tool_names_turn1 = [c[0] for c in calls]
    assert "robinhood_preview_crypto_order" in tool_names_turn1
    assert "robinhood_place_crypto_order" not in tool_names_turn1
    assert "Confirm? (yes/no)" in reply1
    assert "DOGE" in reply1
    assert "Risk Warning" in reply1 or "high-risk" in reply1.lower() or "volatile" in reply1.lower()

    # Turn 2: User confirms "yes" -> places the previewed crypto order!
    reply2 = await ah.chat("yes")
    tool_names_turn2 = [c[0] for c in calls]
    assert "robinhood_place_crypto_order" in tool_names_turn2
    assert "live-order-123" in reply2 or "DOGE" in reply2
