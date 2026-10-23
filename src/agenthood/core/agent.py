"""AgentHood -- the agent class.

Wraps agentu.Agent and wires every available agentu feature to a
Robinhood MCP backend.  The LLM discovers tools via search_tools(),
calls them directly, and formats its own output.

The wiring itself lives in :mod:`agenthood.core.builder`; this module holds the
AgentHood facade -- chat, analyze, direct tool access, memory, the loop-engineering
passthroughs, and lifecycle (token refresh, shutdown).

Feature matrix:

  Deferred tool search       -- all 53 tools behind search_tools
  Skills (progressive)       -- technical-analysis, options-playbook, risk-management
  Codemode                   -- LLM writes Python to call tools
  Streaming                  -- token-by-token via agent.stream()
  Cache                      -- basic / smart / offline / distributed presets
  Guardrails                 -- input length cap + output content filter
  Permissions                -- ask-writes / plan / auto modes
  Context compaction         -- auto-summarise long sessions
  Memory consolidation       -- background LLM synthesis of key facts
  CostTracker middleware     -- API spend tracking per session
  Rationale recording        -- audit trail for every trade decision
  Inbox watcher              -- drop signal files -> auto-ingested as memory
  Sandbox                    -- safe subprocess for codemode execution
  OpenTelemetry              -- GenAI span export
  Notifications              -- Apprise push after findings
  Scheduling                 -- autonomous portfolio scans on cron/interval
  Ralph loop                 -- PROMPT.md checkbox autonomous loop
  Delegate (maker-checker)   -- sub-agent review for high-stakes trades
  best_of                    -- N parallel runs, judge picks best
  Serve                      -- REST + WebSocket API server
  Sessions                   -- persistent conversation history
  Structured output          -- Pydantic schema from chat()
  Evaluate                   -- batch test suite
  Token refresh              -- background Robinhood JWT hot-swap
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Type

from agentu import Agent

from ..config import AgentHoodConfig, load_config
from ..token import TokenManager
from .builder import build_agent
from .facade import AgentuFacade
from .formatting import _format_tool_result
from .llm import analyze_llm_config, converse, llm_completion, stream_completion
from .market_data import fetch_market_data
from .prompts import SYSTEM_PROMPT, analyze_prompt
from .response import _extract_response
from .tickers import KNOWN_CRYPTO_SYMBOLS, extract_tickers, is_crypto_symbol

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # only show warnings/errors unless caller enables debug

# Last resort only: the tool router missed AND the direct conversational call
# also failed (network down, bad key). Never leak the router's own sentinel --
# it reads as a crash, not an answer.
SCOPE_MESSAGE = (
    "I'm AgentHood, your trading assistant. Ask me about a ticker, "
    "your portfolio, or place an order."
)

# Substrings agentu uses when tool selection finds nothing. "tool none not found"
# is the router failing to name a tool at all -- a genuine "Tool XYZ not found"
# names the tool, so real errors still surface.
_NO_TOOL_MARKERS = (
    "no appropriate tool",
    "no tool found",
    "no tool selected",
    "tool none not found",
    "tool 'none' not found",
    "tool null not found",
    "tool 'null' not found",
    "tool search_tools not found",
)


def _is_no_tool_sentinel(text: str) -> bool:
    """True if *text* is agentu reporting a routing miss rather than an answer."""
    t = text.strip().lower()
    return bool(t) and any(marker in t for marker in _NO_TOOL_MARKERS)


import re as _re

_META_OR_GREETING_RE = _re.compile(
    r"^(hi|hey|hello|yo|hiya|thanks|thank you|thx|ok|okay|cool|great|bye|"
    r"what can you do\??|who are you\??|help\??|how do you work\??)$",
    _re.IGNORECASE,
)
_CONFIRM_YES_RE = _re.compile(
    r"^(yes|y|yep|yeah|sure|confirm|confirmed|do it|place it|go ahead|proceed|execute)$",
    _re.IGNORECASE,
)
_CONFIRM_NO_RE = _re.compile(
    r"^(no|n|nope|nah|cancel|abort|stop|nevermind|never mind|don'?t)$",
    _re.IGNORECASE,
)
_FOLLOWUP_PRONOUN_RE = _re.compile(
    r"\b(it|its|that|this|those|them|they|the stock|same one)\b",
    _re.IGNORECASE,
)
_FOLLOWUP_COMP_RE = _re.compile(
    r"(\b(compare|vs\.?|versus|what about|how about|instead)\b|^(and|also|now)\b)",
    _re.IGNORECASE,
)
_FOLLOWUP_ATTR_RE = _re.compile(
    r"\b(p/e|pe\s+ratio|rsi|macd|sma|ema|bollinger|earnings|fundamentals?|technicals?|"
    r"options?|calls?|puts?|news|ratings?|analyst|chart|historicals?|dividend|"
    r"market\s+cap|52w|high|low|verdict|why|explain)\b",
    _re.IGNORECASE,
)
_PORTFOLIO_INTENT_RE = _re.compile(
    r"\b(my\s+portfolio|show\s+(me\s+)?(my\s+)?portfolio|portfolio\s+(value|summary)|"
    r"buying\s+power|account\s+value|cash\s+balance)\b",
    _re.IGNORECASE,
)
_POSITIONS_INTENT_RE = _re.compile(
    r"\b(my\s+positions?|show\s+(me\s+)?(my\s+)?positions?|open\s+positions?|"
    r"my\s+holdings?|what\s+(stocks?|shares?)\s+do\s+i\s+(own|hold))\b",
    _re.IGNORECASE,
)
_DISCOVERY_INTENT_RE = _re.compile(
    r"(\b(best|top|good|hot|trending|popular|biggest)\s+(stocks?|tickers?|equities|plays?|picks?|movers?|gainers?|losers?)\b|"
    r"\b(stocks?|tickers?|what)\s+(should\s+i\s+|to\s+)(buy|trade|invest|watch)\b|"
    r"\b(what\s+should\s+i\s+(buy|trade|invest)|recommend\s+(a\s+)?stock|stock\s+recommendations?|"
    r"daily\s+movers?|market\s+movers?)\b)",
    _re.IGNORECASE,
)
_SIMPLE_QUOTE_INTENT_RE = _re.compile(
    r"^(what('?s|\s+is)\s+[a-z0-9.,\s]+(trading\s+at|price|worth|at)(\s+right\s+now|\s+now|\s+today)?\??|"
    r"price\s+of\s+[a-z0-9.,\s]+\??|"
    r"quote\s+(for\s+)?[a-z0-9.,\s]+\??)$",
    _re.IGNORECASE,
)
_CRYPTO_ORDER_VERB_RE = _re.compile(r"\b(buy|sell|purchase|order)\b", _re.IGNORECASE)
_CRYPTO_WORD_RE = _re.compile(r"\b(crypto|cryto|cryptocurrency|coin|coins)\b", _re.IGNORECASE)
_DOLLAR_AMT_RE = _re.compile(r"(?:\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:dollars?|usd|bucks?)\b)", _re.IGNORECASE)
_QTY_BEFORE_SYM_RE = _re.compile(r"\b(?:buy|sell|purchase)\s+(\d+(?:\.\d+)?)\s+(?!dollars?|usd|bucks?)([a-zA-Z]{2,10})\b", _re.IGNORECASE)
_LIMIT_AT_RE = _re.compile(r"\b(?:at|@|limit)\s*\$?\s*(\d+(?:\.\d+)?)\b", _re.IGNORECASE)


def _is_order_confirm_pending(history: list[tuple[str, str]]) -> bool:
    """Return True if the most recent assistant message asked to confirm an order."""
    if not history:
        return False
    for role, text in reversed(history):
        if role == "assistant":
            low = str(text).lower()
            return (
                "confirm?" in low
                or "(yes/no)" in low
                or "(yes / no)" in low
                or "about to place" in low
                or "order review" in low
            )
    return False


def _needs_multiturn_context(
    prompt: str,
    history: list[tuple[str, str]],
    last_ticker: str | None,
) -> bool:
    """Return True when *prompt* depends on prior conversation turns."""
    if not history:
        return False
    stripped = prompt.strip()
    if not stripped or _META_OR_GREETING_RE.match(stripped):
        return False
    if _FOLLOWUP_PRONOUN_RE.search(stripped) or _FOLLOWUP_COMP_RE.search(stripped):
        return True
    if last_ticker and not extract_tickers(stripped) and _FOLLOWUP_ATTR_RE.search(stripped):
        return True
    return False


def _build_contextual_prompt(
    prompt: str,
    history: list[tuple[str, str]],
    recent_tickers: list[str],
) -> str:
    """Attach compact multi-turn context so agent.infer() can resolve references."""
    lines = ["[Conversation Context]"]
    if recent_tickers:
        lines.append(f"Active ticker(s): {', '.join(recent_tickers[-3:])} (most recent: {recent_tickers[-1]})")
    for role, text in history[-4:]:
        snippet = " ".join(str(text).strip().split())[:300]
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {snippet}")
    lines.append("[End Context]")
    lines.append(f"Current request: {prompt}")
    return "\n".join(lines)


# -- AgentHood ----------------------------------------------------------------

class AgentHood(AgentuFacade):
    """Robinhood trading agent powered by agentu + Robinhood MCP."""

    def __init__(self, config: AgentHoodConfig | None = None):
        self.config = config or load_config()
        self._agent: Agent | None = None
        self._mcp_adapter = None
        self._all_tools: list = []  # all MCP tools (for _call_tool_direct)
        self._rules_tmp: Path | None = None
        self._token_manager = TokenManager()
        self._refresh_task: asyncio.Task | None = None
        self._cost_tracker = None   # filled in start()
        self._session_manager = None
        self._account_number: str | None = None
        self._rhs_account_number: str | None = None
        self._pending_crypto_order: dict[str, Any] | None = None
        self._history: list[tuple[str, str]] = []
        self._recent_tickers: list[str] = []

    def set_dry_run(self, dry_run: bool) -> None:
        """Toggle dry_run mode dynamically and re-bind safety hooks on the active agent."""
        self.config.dry_run = bool(dry_run)
        if self._agent is not None:
            try:
                from ..hooks import trading_guard
                guard = trading_guard(
                    dry_run=self.config.dry_run,
                    max_order_value=self.config.max_order_value,
                    allowed_symbols=self.config.allowed_symbols,
                )
                self._agent.with_hooks(pre_tool=guard)
            except Exception as exc:
                logger.debug("Could not re-bind trading_guard: %s", exc)

    @property
    def last_ticker(self) -> str | None:
        """Most recently discussed ticker symbol in this session, or None."""
        if not hasattr(self, "_recent_tickers") or not self._recent_tickers:
            return None
        return self._recent_tickers[-1]

    def _track_tickers(self, symbols: list[str]) -> None:
        if not hasattr(self, "_recent_tickers"):
            self._recent_tickers = []
        cleaned = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
        if not cleaned:
            return
        # Store the latest batch of tickers so _table_tickers maps 1..N to the last displayed table
        if len(cleaned) > 1:
            self._table_tickers = list(cleaned)
        for s in cleaned:
            if s in self._recent_tickers:
                self._recent_tickers.remove(s)
            self._recent_tickers.append(s)
        if len(self._recent_tickers) > 10:
            self._recent_tickers = self._recent_tickers[-10:]
        try:
            from agenthood.tui.input import set_completion_context
            set_completion_context(self._recent_tickers)
        except Exception:
            pass

    def record_turn(
        self,
        user_text: str,
        assistant_text: str,
        tickers: list[str] | None = None,
    ) -> None:
        """Record a user/assistant turn in session history and update active tickers."""
        if not hasattr(self, "_history"):
            self._history = []
        if tickers:
            self._track_tickers(tickers)
        else:
            found = extract_tickers(user_text, limit=5)
            if found:
                self._track_tickers(found)

        self._history.append(("user", user_text))
        self._history.append(("assistant", assistant_text))
        if len(self._history) > 40:
            target_idx = len(self._history) - 40
            safe_idx = target_idx
            for i in range(target_idx, len(self._history)):
                turn = self._history[i]
                if isinstance(turn, (list, tuple)) and turn[0] == "user":
                    safe_idx = i
                    break
            self._history = self._history[safe_idx:]

    async def start(self) -> "AgentHood":
        """Connect to Robinhood MCP and prepare the agent. Returns self for chaining."""
        cfg = self.config

        built = await build_agent(cfg)
        agent = built.agent
        self._all_tools = built.tools     # keep reference for _call_tool_direct
        self._cost_tracker = built.cost_tracker
        self._rules_tmp = built.rules_tmp
        self._mcp_adapter = built.adapter
        self._agent = agent

        # Background token refresh
        self._refresh_task = asyncio.create_task(self._token_refresh_loop())

        # -- Autonomous scan schedule ------------------------------------------
        if cfg.scan_interval > 0:
            agent.with_schedule(
                every=max(1, cfg.scan_interval // 60),
                prompt=cfg.scan_prompt,
            )
            asyncio.create_task(agent.start())
            logger.info("Autonomous scan started (every %ds)", cfg.scan_interval)

        logger.info(
            "AgentHood ready | model=%s dry_run=%s max_order=$%.0f codemode=%s",
            cfg.model, cfg.dry_run, cfg.max_order_value, cfg.codemode,
        )
        return self

    # -- Core inference --------------------------------------------------------

    async def _call_tool_direct(self, tool_name: str, params: dict) -> Any:
        """Call an MCP tool directly by name, bypassing agent.call() permission checks.

        Looks up the tool from the full tool list stored during start(),
        not from the agent's active (deferred) pool. This avoids the
        broken search_tools prewarm path entirely.
        """
        tool = next(
            (t for t in self._all_tools if t.name == tool_name),
            None,
        )
        if tool is None:
            raise ValueError(f"Tool {tool_name!r} not found")

        import inspect as _inspect
        if _inspect.iscoroutinefunction(tool.function):
            return await tool.function(**params)
        else:
            return tool.function(**params)

    async def _get_account_number(self) -> str:
        """Fetch the primary account number (cached after first call).

        Preference order, per AGENTS.md: the agentic account, then the default
        account, then whatever came first.
        """
        if self._account_number:
            return self._account_number
        import json as _j
        raw = await self._call_tool_direct("robinhood_get_accounts", {})
        data = _j.loads(raw) if isinstance(raw, str) else raw
        accounts = (data.get("data") or {}).get("accounts") or data.get("accounts") or []
        if not accounts:
            raise ValueError("No accounts found")
        chosen = None
        for a in accounts:
            if a.get("agentic_allowed"):
                chosen = a
                break
        if chosen is None:
            for a in accounts:
                if a.get("is_default"):
                    chosen = a
                    break
        if chosen is None:
            chosen = accounts[0]
        acct = chosen.get("account_number") or chosen.get("id") or ""
        self._account_number = acct
        self._rhs_account_number = str(chosen.get("rhs_account_number") or acct)
        return acct

    async def _get_rhs_account_number(self) -> str:
        """Fetch the numeric rhs_account_number required by Robinhood crypto tools."""
        if getattr(self, "_rhs_account_number", None):
            return self._rhs_account_number  # type: ignore[return-value]
        await self._get_account_number()
        return getattr(self, "_rhs_account_number", None) or self._account_number or ""

    async def _preview_crypto_order_intent(
        self,
        prompt: str,
        crypto_syms: list[str],
        _progress: Optional[Callable[[str], None]] = None,
    ) -> str | None:
        """Preview a crypto order via robinhood_preview_crypto_order and prompt for confirmation.

        Adheres strictly to AGENTS.md Safety Rules:
          1. Never places an order without a prior preview call.
          2. Always asks explicit confirmation ("Confirm? (yes/no)").
          3. Includes an explicit high-risk crypto warning.
          4. Prefers limit orders unless the user explicitly requested a market order.
          5. Explains dry_run mode when active.
        """
        import json as _j

        sym = crypto_syms[0].upper() if crypto_syms else "DOGE"
        if sym.endswith("-USD"):
            sym = sym[:-4]
        elif len(sym) > 3 and sym.endswith("USD"):
            sym = sym[:-3]

        low = prompt.lower()
        side = "sell" if _re.search(r"\bsell\b", low) else "buy"
        order_type = "market" if _re.search(r"\bmarket\b", low) else "limit"

        dollar_amt: str | None = None
        quantity: str | None = None
        m_dollar = _DOLLAR_AMT_RE.search(prompt)
        m_qty = _QTY_BEFORE_SYM_RE.search(prompt)
        if m_dollar:
            val = float(m_dollar.group(1) or m_dollar.group(2))
            dollar_amt = f"{val:.2f}"
        elif m_qty:
            quantity = m_qty.group(1)
        else:
            dollar_amt = "1.00"

        if _progress:
            _progress(f"fetching {sym}-USD quote & account for preview...")

        rhs_acct = await self._get_rhs_account_number()

        # Fetch live crypto quote to determine current ask/bid price for limit orders
        pair = f"{sym}-USD"
        q_raw = await self._call_tool_direct("robinhood_get_crypto_quotes", {"symbols": [pair]})
        q_data = _j.loads(q_raw) if isinstance(q_raw, str) else q_raw
        q_results = ((q_data.get("data") or q_data).get("results") if isinstance(q_data, dict) else None) or []
        first_q = q_results[0] if q_results and isinstance(q_results[0], dict) else {}

        ask_px = float(first_q.get("ask_price") or first_q.get("mark_price") or 0.0)
        bid_px = float(first_q.get("bid_price") or first_q.get("mark_price") or 0.0)
        ref_px = ask_px if side == "buy" else bid_px

        m_lim = _LIMIT_AT_RE.search(prompt)
        if m_lim:
            ref_px = float(m_lim.group(1))

        if ref_px <= 0:
            order_type = "market"

        preview_params: dict[str, Any] = {
            "rhs_account_number": rhs_acct,
            "symbol": sym,
            "side": side,
            "type": order_type,
        }
        if dollar_amt is not None:
            preview_params["dollar_amount"] = dollar_amt
        elif quantity is not None:
            preview_params["quantity"] = quantity

        if order_type == "limit" and ref_px > 0:
            if ref_px < 0.01:
                lim_str = f"{ref_px:.6f}"
            elif ref_px < 1.0:
                lim_str = f"{ref_px:.4f}"
            else:
                lim_str = f"{ref_px:.2f}"
            preview_params["limit_price"] = lim_str
            preview_params["time_in_force"] = "gtc"

        if _progress:
            _progress(f"previewing {sym} {order_type} {side} order...")

        prev_raw = await self._call_tool_direct("robinhood_preview_crypto_order", preview_params)
        prev_str = prev_raw if isinstance(prev_raw, str) else _j.dumps(prev_raw)
        prev_data = _j.loads(prev_str) if isinstance(prev_str, str) else {}
        inner = (prev_data.get("data") or prev_data) if isinstance(prev_data, dict) else {}
        ord_obj = inner.get("order") if isinstance(inner, dict) else None
        if not isinstance(ord_obj, dict):
            err_msg = prev_data.get("error") or inner.get("error") or prev_str[:400]
            return f"Could not preview {sym} crypto order: {err_msg}"

        est_qty = ord_obj.get("quantity") or quantity or "?"
        px_val = ord_obj.get("price") or preview_params.get("limit_price") or str(ref_px)
        notional_val = ord_obj.get("net_rounded_estimated_notional") or dollar_amt or "?"

        self._pending_crypto_order = {
            "params": preview_params,
            "symbol": sym,
            "side": side,
            "type": order_type,
            "quantity": est_qty,
            "price": px_val,
            "notional": notional_val,
        }

        table_md = _format_tool_result("robinhood_preview_crypto_order", prev_str) or prev_str
        lines = [
            "**Risk Warning:** Cryptocurrency is a high-risk, volatile asset class that trades 24/7 without FDIC or SIPC protection.",
        ]
        if getattr(self.config, "dry_run", True):
            lines.append(
                "\n> **DRY-RUN MODE ACTIVE:** Live order placement is disabled by default. "
                "Confirming below will simulate the order. To place a real order, run `agenthood --live` or type `.live` in the REPL."
            )
        lines.append("")
        lines.append(table_md)
        lines.append(
            "\n*You can switch between Market Maker Routing and Exchange Routing anytime in the Robinhood app under Settings -> Crypto -> Fee Tiers.*"
        )
        try:
            px_f = float(px_val)
            px_disp = f"${px_f:,.4f}" if 0 < abs(px_f) < 1.0 else f"${px_f:,.2f}"
        except (ValueError, TypeError):
            px_disp = f"${px_val}"
        lines.append(
            f"\nI'm about to place a **{order_type.upper()} {side.upper()}** order for **{est_qty} {sym}** "
            f"(~${notional_val} at {px_disp}, GTC). Confirm? (yes/no)"
        )
        return "\n".join(lines)

    async def _execute_pending_crypto_order(
        self,
        pending: dict[str, Any],
        _progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Execute (or simulate in dry_run) a confirmed crypto order."""
        import json as _j
        from ..hooks import HookAction, trading_guard

        params = pending["params"]
        sym = pending["symbol"]
        guard = trading_guard(
            dry_run=getattr(self.config, "dry_run", True),
            max_order_value=getattr(self.config, "max_order_value", 500.0),
            allowed_symbols=getattr(self.config, "allowed_symbols", []),
        )
        hook_res = await guard("robinhood_place_crypto_order", params, {})
        if hook_res.action == HookAction.DENY:
            return (
                f"### Order Blocked by Safety Guard\n\n"
                f"{hook_res.reason}\n\n"
                f"- **Simulated Order:** {pending['type'].upper()} {pending['side'].upper()} "
                f"{pending['quantity']} {sym} (~${pending['notional']})\n"
                f"- **Tip:** Run `agenthood --live` or type `.live` in the REPL to enable live trading."
            )

        if _progress:
            _progress(f"placing live {sym} {pending['type']} {pending['side']} order...")

        raw = await self._call_tool_direct("robinhood_place_crypto_order", params)
        raw_str = raw if isinstance(raw, str) else _j.dumps(raw)
        formatted = _format_tool_result("robinhood_place_crypto_order", raw_str)
        if formatted:
            return f"**Live Crypto Order Submitted**\n\n{formatted}"
        return f"**Live Crypto Order Submitted**\n\n{raw_str[:1000]}"

    async def _direct_dispatch(self, user_input: str) -> dict | None:
        """Check for watch/alert prompts (structural, not intent-based).

        Everything else goes to the LLM via infer().
        """
        from agenthood.watcher import parse_watch_prompt
        watch_spec = parse_watch_prompt(
            user_input.strip(),
            context_ticker=self.last_ticker,
        )
        if watch_spec:
            return {"__watch__": watch_spec}
        return None

    # -- Chat ------------------------------------------------------------------

    async def chat(
        self,
        prompt: str,
        output_schema: Optional[Type] = None,
        _progress: Optional[Callable[[str], None]] = None,
        _on_token: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> str | Any:
        """Send a prompt to the agent. Returns the response text.

        Pipeline:
          1. Check for watch/alert prompts (deterministic, no LLM).
          2. Check for pending order confirmation/cancellation or multi-turn follow-up.
          3. Call agent.infer() -- the LLM discovers tools via search_tools(),
             calls them, and formats the response.
          4. Store conversation history and active ticker context.
        """
        if self._agent is None:
            await self.start()

        if not hasattr(self, "_history"):
            self._history: list[tuple[str, str]] = []
        if not hasattr(self, "_recent_tickers"):
            self._recent_tickers: list[str] = []

        # Step 1: watch/alert detection (structural, bypasses LLM)
        if output_schema is None:
            direct = await self._direct_dispatch(prompt)
            if direct is not None:
                spec_sym = direct.get("__watch__", {}).get("ticker")
                self.record_turn(
                    prompt,
                    str(direct),
                    tickers=[spec_sym] if spec_sym else None,
                )
                return direct

        # Step 1b: multi-turn order confirmation / cancellation handling
        stripped = prompt.strip()
        infer_prompt = prompt
        pending_crypto = getattr(self, "_pending_crypto_order", None)
        if output_schema is None and pending_crypto is not None:
            if _CONFIRM_NO_RE.match(stripped):
                self._pending_crypto_order = None
                msg = "Order cancelled. No order was placed."
                self.record_turn(prompt, msg)
                return msg
            if _CONFIRM_YES_RE.match(stripped):
                self._pending_crypto_order = None
                try:
                    exec_md = await self._execute_pending_crypto_order(pending_crypto, _progress=_progress)
                    self.record_turn(prompt, exec_md, tickers=[pending_crypto["symbol"]])
                    return exec_md
                except Exception as exc:
                    err_md = f"Error placing crypto order: {exc}"
                    self.record_turn(prompt, err_md)
                    return err_md
            # User moved on to a different prompt -- clear stale pending order
            self._pending_crypto_order = None

        if output_schema is None and _is_order_confirm_pending(self._history):
            if _CONFIRM_NO_RE.match(stripped):
                msg = "Order cancelled. No order was placed."
                self.record_turn(prompt, msg)
                return msg
            if _CONFIRM_YES_RE.match(stripped):
                infer_prompt = _build_contextual_prompt(
                    f"{prompt} (User confirmed the reviewed order above. Proceed to place the order.)",
                    self._history,
                    self._recent_tickers,
                )
        elif output_schema is None and _needs_multiturn_context(
            prompt, self._history, self.last_ticker
        ):
            infer_prompt = _build_contextual_prompt(
                prompt, self._history, self._recent_tickers
            )

        prompt_tickers = extract_tickers(prompt, limit=5, context_ticker=self.last_ticker)
        crypto_tickers = [t for t in prompt_tickers if is_crypto_symbol(t)]

        # Step 1b2: Deterministic crypto order preview (Safety Rules 1, 2, 3, 4, 5)
        # Triggers when the user asks to buy/sell a crypto asset ("buy $1 of DOGE",
        # "buy BTC", or "just buy a crypto coin"). Always runs preview_crypto_order
        # first and asks explicit confirmation -- NEVER places an order directly.
        if (
            output_schema is None
            and _CRYPTO_ORDER_VERB_RE.search(stripped)
            and not _DISCOVERY_INTENT_RE.search(stripped)
            and (crypto_tickers or _CRYPTO_WORD_RE.search(stripped))
        ):
            try:
                prev_md = await self._preview_crypto_order_intent(
                    stripped, crypto_tickers, _progress=_progress
                )
                if prev_md:
                    sym_used = (self._pending_crypto_order or {}).get("symbol") or (crypto_tickers[0] if crypto_tickers else "DOGE")
                    self.record_turn(prompt, prev_md, tickers=[sym_used])
                    return prev_md
            except Exception as exc:
                logger.debug("Crypto order preview fast-path failed: %s", exc)

        # Step 1c: Fast-path live market discovery ("best stock to buy", "trending stocks", "daily movers")
        # When the user asks a discovery question with no specific ticker named,
        # fetch Robinhood's live Trending Stocks / Daily Movers watchlists and
        # quotes in ~1s rather than waiting 20-30s on multi-step tool search.
        if output_schema is None and not prompt_tickers and _DISCOVERY_INTENT_RE.search(stripped):
            try:
                disc_md, disc_syms = await self._fetch_discovery_movers(
                    prefer_movers="mover" in stripped.lower(),
                    _progress=_progress,
                )
                if disc_md:
                    self.record_turn(prompt, disc_md, tickers=disc_syms[:8] if disc_syms else None)
                    return disc_md
            except Exception as exc:
                logger.debug("Fast-path discovery movers failed: %s", exc)

        # Step 1d: Fast-path simple price/quote check ("what is NVDA trading at?", "what is it trading at now?", "NVDA price")
        if output_schema is None and prompt_tickers and _SIMPLE_QUOTE_INTENT_RE.match(stripped):
            try:
                if _progress:
                    _progress(f"fetching {', '.join(prompt_tickers)} quote...")
                all_crypto = all(is_crypto_symbol(t) for t in prompt_tickers)
                q_tool = "robinhood_get_crypto_quotes" if all_crypto else "robinhood_get_equity_quotes"
                q_syms = [f"{t}-USD" if all_crypto and not t.endswith("USD") else t for t in prompt_tickers]
                raw_q = await self._call_tool_direct(q_tool, {"symbols": q_syms})
                if isinstance(raw_q, dict) or (isinstance(raw_q, str) and raw_q.strip()[:1] in ("{", "[")):
                    import json as _j
                    raw_str = raw_q if isinstance(raw_q, str) else _j.dumps(raw_q)
                    formatted_q = _format_tool_result(q_tool, raw_str)
                    if formatted_q:
                        self.record_turn(prompt, formatted_q, tickers=prompt_tickers)
                        return formatted_q
            except Exception as exc:
                logger.debug("Fast-path quote check failed: %s", exc)

        # Step 2: call the LLM with full tool access
        if _progress:
            _progress("thinking...")

        # Suppress agentu's noisy error logging -- we handle failures gracefully.
        import logging as _logging
        _agentu_log = _logging.getLogger("agentu")
        _old_level = _agentu_log.level
        _agentu_log.setLevel(_logging.CRITICAL)
        try:
            result = await self._agent.infer(
                infer_prompt, output_type=output_schema, **kwargs
            )
        except Exception as exc:
            logger.debug("infer() failed: %s", exc)
            result = {"error": str(exc)}
        finally:
            _agentu_log.setLevel(_old_level)

        if output_schema is not None:
            return result

        extracted = _extract_response(result)

        # If raw tool JSON leaked through (model didn't format), apply presentation.
        if isinstance(result, dict) and "result" in result and "text" not in result:
            raw_data = extracted
            if raw_data.strip()[:1] in ('{', '['):
                formatted = _format_tool_result(
                    result.get("tool_used", ""), raw_data
                )
                if formatted:
                    extracted = formatted

        # Fallback: if model failed or returned nothing useful, try the tool directly.
        routing_miss = _is_no_tool_sentinel(extracted)
        recovered = False
        prompt_tickers = extract_tickers(prompt, limit=5, context_ticker=self.last_ticker)

        if (
            (isinstance(result, dict) and "error" in result)
            or routing_miss
            or not extracted.strip()
        ):
            # 2a. Portfolio / positions direct fallback on routing miss
            if not recovered and (_PORTFOLIO_INTENT_RE.search(stripped) or _POSITIONS_INTENT_RE.search(stripped)):
                try:
                    from ..tui.format import _format_mcp_response
                    acct = await self._get_account_number()
                    if _POSITIONS_INTENT_RE.search(stripped):
                        raw = await self._call_tool_direct(
                            "robinhood_get_equity_positions",
                            {"account_number": acct},
                        )
                        extracted = _format_mcp_response(raw, "Positions")
                    else:
                        raw = await self._call_tool_direct(
                            "robinhood_get_portfolio",
                            {"account_number": acct},
                        )
                        extracted = _format_mcp_response(raw, "Portfolio")
                    recovered = True
                except Exception as exc:
                    logger.debug("Direct portfolio/positions fallback failed: %s", exc)

            # 2b. Direct fallback for stock/crypto quote lookups when LLM tool selection fails
            if not recovered and prompt_tickers:
                try:
                    all_crypto = all(is_crypto_symbol(t) for t in prompt_tickers)
                    q_tool = "robinhood_get_crypto_quotes" if all_crypto else "robinhood_get_equity_quotes"
                    q_syms = [f"{t}-USD" if all_crypto and not t.endswith("USD") else t for t in prompt_tickers]
                    raw = await self._call_tool_direct(
                        q_tool,
                        {"symbols": q_syms}
                    )
                    ext = _extract_response(raw)
                    formatted = _format_tool_result(q_tool, ext)

                    if formatted:
                        extracted = formatted
                        recovered = True
                except Exception as exc:
                    logger.debug("Direct tool fallback failed: %s", exc)

            # 2c. Live market discovery fallback ("best stock to buy", "trending stocks", "daily movers")
            if not recovered and _DISCOVERY_INTENT_RE.search(stripped):
                try:
                    disc_md, disc_syms = await self._fetch_discovery_movers(
                        prefer_movers="mover" in stripped.lower(),
                        _progress=_progress,
                    )
                    if disc_md:
                        extracted = disc_md
                        if disc_syms:
                            prompt_tickers = disc_syms[:5]
                        recovered = True
                except Exception as exc:
                    logger.debug("Discovery movers fallback failed: %s", exc)

        # If the user asked for stock discovery ("best stock to buy", "trending stocks")
        # and the LLM returned generic prose without fetching live quotes/tables,
        # attach live Trending Stocks / Daily Movers data directly.
        if (
            not recovered
            and not prompt_tickers
            and _DISCOVERY_INTENT_RE.search(stripped)
            and "|" not in extracted
        ):
            try:
                disc_md, disc_syms = await self._fetch_discovery_movers(
                    prefer_movers="mover" in stripped.lower(),
                    _progress=_progress,
                )
                if disc_md:
                    extracted = disc_md
                    if disc_syms:
                        prompt_tickers = disc_syms[:5]
                    recovered = True
            except Exception as exc:
                logger.debug("Discovery movers enrichment failed: %s", exc)

        # A routing miss means the user asked something with no tool behind it --
        # a greeting, a concept question, small talk. That deserves an answer, not
        # a report that routing failed. Ask the LLM directly, no tools offered.
        # Genuine errors (rate limits, auth, network) still surface unchanged.
        if not recovered and (routing_miss or not extracted.strip()):
            reply = await converse(self.config, prompt, history=self._history)
            extracted = reply if reply.strip() else SCOPE_MESSAGE

        # Step 3: store in history and update active tickers
        self.record_turn(prompt, extracted, tickers=prompt_tickers or None)

        return extracted

    async def _fetch_discovery_movers(
        self,
        prefer_movers: bool = False,
        _progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, list[str]]:
        """Fetch live Trending Stocks / Daily Movers from Robinhood MCP and format as a markdown table."""
        import json as _j
        from ..tui.format import _format_mcp_response

        if _progress:
            _progress("scanning trending stocks & market movers...")

        raw_lists = await self._call_tool_direct("robinhood_get_popular_watchlists", {})
        lists_data = _j.loads(raw_lists) if isinstance(raw_lists, str) else raw_lists
        lists = ((lists_data.get("data") or {}).get("lists") if isinstance(lists_data, dict) else None) or []
        if not lists:
            return "", []

        target = None
        preferred_kw = "mover" if prefer_movers else "trending"
        for lst in lists:
            if preferred_kw in (lst.get("display_name") or "").lower():
                target = lst
                break
        if target is None:
            for lst in lists:
                name_low = (lst.get("display_name") or "").lower()
                if any(k in name_low for k in ("trending", "mover", "popular")):
                    target = lst
                    break
        if target is None:
            target = lists[0]

        list_id = target.get("id")
        list_name = target.get("display_name") or "Trending Stocks"
        if not list_id:
            return "", []

        raw_items = await self._call_tool_direct(
            "robinhood_get_watchlist_items", {"list_id": list_id}
        )
        items_data = _j.loads(raw_items) if isinstance(raw_items, str) else raw_items
        items = ((items_data.get("data") or {}).get("items") if isinstance(items_data, dict) else None) or []
        symbols = [it.get("symbol") for it in items if isinstance(it, dict) and it.get("symbol")][:8]
        if not symbols:
            return "", []

        if _progress:
            _progress(f"fetching live quotes for {', '.join(symbols[:5])}...")

        raw_quotes = await self._call_tool_direct(
            "robinhood_get_equity_quotes", {"symbols": symbols}
        )
        table_md = _format_mcp_response(raw_quotes, f"{list_name} Quotes")
        footer = (
            "\n\n*Tip: Run `.a TICKER` (e.g. `.a "
            f"{symbols[0]}`) for full fundamentals, RSI/MACD technicals, and a BUY/SELL/HOLD report.*"
        )
        return table_md + footer, symbols

    def clear_history(self) -> None:
        """Start a genuinely new conversation.

        Our own ``_history`` is only used to give the no-tool conversational
        path some continuity; the context the model actually sees lives on the
        agentu Agent. Clearing just the former left the previous conversation
        fully in play -- ``.new`` looked like it worked and changed nothing.
        """
        self._history = []
        self._recent_tickers = []
        agent = self._agent
        if agent is None:
            return
        try:
            history = getattr(agent, "conversation_history", None)
            if isinstance(history, list):
                history.clear()
        except Exception as exc:            # pragma: no cover - defensive
            logger.debug("could not clear agentu conversation history: %s", exc)
        try:
            agent.clear_short_term_memory()
        except Exception as exc:            # pragma: no cover - defensive
            logger.debug("could not clear agentu short-term memory: %s", exc)

    async def analyze(
        self,
        ticker: str,
        _progress: Optional[Callable[[str], None]] = None,
        _on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Run a structured market analysis for a ticker.

        Fetches data directly via MCP tools (no LLM tool discovery needed),
        then passes the raw data to the LLM for analysis and formatting.
        """
        if self._agent is None:
            await self.start()

        sym = ticker.upper()

        # Step 1: Fetch all data in PARALLEL (not sequential)
        if _progress:
            _progress(f"fetching {sym} data...")

        data_dump = await fetch_market_data(self, sym, _progress=_progress)

        # Step 2: build the analysis prompt from the pre-fetched data
        prompt = analyze_prompt(sym, data_dump)

        # Step 3: Stream or fetch LLM response
        if _on_token:
            # Streaming mode -- yield tokens as they arrive
            if _progress:
                _progress("streaming analysis...")
            try:
                chunks = []
                async for token in self._stream_completion(prompt):
                    chunks.append(token)
                    _on_token(token)
                if chunks:
                    out = "".join(chunks)
                    self.record_turn(f"analyze {sym}", out, tickers=[sym])
                    return out
            except Exception as exc:
                logger.warning("Streaming failed (%s), falling back to non-streaming", exc)

        # Non-streaming fallback
        if _progress:
            _progress("analyzing...")
        out = await self._llm_completion(prompt, _progress=_progress)
        self.record_turn(f"analyze {sym}", out, tickers=[sym])
        return out


    def _analyze_llm_config(self) -> tuple[str, str, str]:
        """Return (model, api_base, api_key) for the analyze LLM path."""
        return analyze_llm_config(self.config)

    async def _llm_completion(
        self,
        prompt: str,
        _progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Call the LLM directly with NO tool definitions.

        Used by analyze() to prevent the model from attempting tool calls
        when we just want it to analyze pre-fetched data.
        """
        return await llm_completion(self.config, prompt, _progress=_progress)

    def _stream_completion(self, prompt: str):
        """Async generator that yields tokens as they arrive via SSE streaming."""
        return stream_completion(self.config, prompt)

    async def stream(self, prompt: str) -> None:
        """Stream response tokens to stdout as they arrive."""
        if self._agent is None:
            await self.start()
        async for token in self._agent.stream(prompt):
            print(token, end="", flush=True)
        print()

    # -- Token refresh loop ----------------------------------------------------

    async def _token_refresh_loop(self) -> None:
        """Background task: hot-swap the Robinhood JWT before it expires."""
        CHECK_INTERVAL = 240
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                new_token = await self._token_manager.get_valid_token_async()
                if new_token and self._mcp_adapter:
                    transport = self._mcp_adapter.transport
                    if hasattr(transport, "config") and transport.config.auth:
                        old = transport.config.auth.headers.get("Authorization", "")
                        if not old.endswith(new_token):
                            transport.config.auth.headers["Authorization"] = (
                                f"Bearer {new_token}"
                            )
                            logger.info("Robinhood token hot-swapped on MCP transport")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Token refresh loop error: %s", e)

    async def stop(self):
        """Stop all background tasks and clean up resources."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        if self._agent:
            try:
                self._agent.stop()
                self._agent.stop_inbox()
            except Exception:
                pass
            try:
                close_fn = getattr(self._agent, "close", None)
                if callable(close_fn):
                    res = close_fn()
                    if asyncio.iscoroutine(res):
                        await res
                sess = getattr(self._agent, "_session", None) or getattr(self._agent, "session", None)
                if sess and hasattr(sess, "close"):
                    res = sess.close()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass
        if self._mcp_adapter:
            try:
                await self._mcp_adapter.close()
                await asyncio.sleep(0.05)
            except Exception:
                pass
        if self._rules_tmp and self._rules_tmp.exists():
            self._rules_tmp.unlink(missing_ok=True)

    async def __aenter__(self) -> "AgentHood":
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()
