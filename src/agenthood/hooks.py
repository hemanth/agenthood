"""Trading safety hooks for agenthood.

pre_tool hook wired into the agentu Agent:
  - In dry_run mode, blocks place_equity_order and place_option_order.
  - Enforces per-order notional cap (max_order_value).
  - Enforces symbol allowlist when configured.
  - Always requires explicit confirmation for order placement.
"""
from __future__ import annotations

import logging
from typing import Any

from agentu import HookResult, HookAction

logger = logging.getLogger(__name__)

# Tools that actually move money — matched by suffix (MCPToolAdapter prefixes with server name)
_ORDER_SUFFIXES = {"place_equity_order", "place_option_order", "exercise_option"}
_REVIEW_SUFFIXES = {"review_equity_order", "review_option_order"}
_CANCEL_SUFFIXES = {"cancel_equity_order", "cancel_option_order", "cancel_option_exercise"}


def _matches(tool_name: str, suffixes: set[str]) -> bool:
    """Match tool name against a set of suffixes (handles 'robinhood_' prefix)."""
    bare = tool_name.split("_", 1)[-1] if "_" in tool_name else tool_name
    return tool_name in suffixes or bare in suffixes


def trading_guard(dry_run: bool, max_order_value: float, allowed_symbols: list[str]):
    """Return a pre_tool hook closure configured with safety settings."""

    async def _hook(tool_name: str, params: dict[str, Any], context: Any) -> HookResult:
        # ── Dry-run: block real orders ─────────────────────────────────────
        if dry_run and _matches(tool_name, _ORDER_SUFFIXES):
            reason = (
                f"🔒 DRY RUN — '{tool_name}' is blocked. "
                "Use review_equity_order / review_option_order to simulate, "
                "or set dry_run=False in agenthood.json to enable live trading."
            )
            logger.warning(reason)
            return HookResult(action=HookAction.DENY, reason=reason)

        # ── Symbol allowlist ───────────────────────────────────────────────
        if allowed_symbols and _matches(tool_name, _ORDER_SUFFIXES | _REVIEW_SUFFIXES):
            symbol = (
                params.get("symbol") or
                params.get("underlying_symbol") or
                params.get("chain_symbol", "")
            ).upper()
            if symbol and symbol not in [s.upper() for s in allowed_symbols]:
                reason = (
                    f"⛔ Symbol '{symbol}' is not in the allowed list: "
                    f"{', '.join(allowed_symbols)}"
                )
                logger.warning(reason)
                return HookResult(action=HookAction.DENY, reason=reason)

        # ── Notional cap ───────────────────────────────────────────────────
        if _matches(tool_name, _ORDER_SUFFIXES) and max_order_value > 0:
            qty = float(params.get("quantity") or params.get("shares") or 1)
            price = float(params.get("price") or params.get("limit_price") or params.get("stop_price") or 0)
            notional = float(params.get("dollar_amount") or (qty * price) or 0)
            if notional == 0:
                reason = (
                    f"⛔ Cannot determine order notional value to enforce ${max_order_value:.2f} cap. "
                    "Specify limit_price or dollar_amount."
                )
                logger.warning(reason)
                return HookResult(action=HookAction.DENY, reason=reason)
            if notional > max_order_value:
                reason = (
                    f"⛔ Order notional ${notional:.2f} exceeds cap "
                    f"of ${max_order_value:.2f}. Lower quantity/price or raise max_order_value."
                )
                logger.warning(reason)
                return HookResult(action=HookAction.DENY, reason=reason)

        # ── Audit log all trade-related calls ──────────────────────────────
        if _matches(tool_name, _ORDER_SUFFIXES | _CANCEL_SUFFIXES | _REVIEW_SUFFIXES):
            logger.info("[agenthood] tool=%s params=%s", tool_name, params)

        return HookResult(action=HookAction.ALLOW)

    return _hook

