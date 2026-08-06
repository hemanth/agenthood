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
from .tickers import extract_tickers

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
)


def _is_no_tool_sentinel(text: str) -> bool:
    """True if *text* is agentu reporting a routing miss rather than an answer."""
    t = text.strip().lower()
    return bool(t) and any(marker in t for marker in _NO_TOOL_MARKERS)


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
        """Fetch the primary account number (cached after first call)."""
        if self._account_number:
            return self._account_number
        import json as _j
        raw = await self._call_tool_direct("robinhood_get_accounts", {})
        data = _j.loads(raw) if isinstance(raw, str) else raw
        accounts = (data.get("data") or {}).get("accounts") or data.get("accounts") or []
        if not accounts:
            raise ValueError("No accounts found")
        acct = accounts[0].get("account_number") or accounts[0].get("id")
        self._account_number = acct
        return acct

    async def _direct_dispatch(self, user_input: str) -> dict | None:
        """Check for watch/alert prompts (structural, not intent-based).

        Everything else goes to the LLM via infer().
        """
        from agenthood.watcher import parse_watch_prompt
        watch_spec = parse_watch_prompt(user_input.strip())
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
          2. Call agent.infer() -- the LLM discovers tools via search_tools(),
             calls them, and formats the response.
          3. Store conversation history.
        """
        if self._agent is None:
            await self.start()

        if not hasattr(self, "_history"):
            self._history: list[tuple[str, str]] = []

        # Step 1: watch/alert detection (structural, bypasses LLM)
        if output_schema is None:
            direct = await self._direct_dispatch(prompt)
            if direct is not None:
                self._history.append(("user", prompt))
                self._history.append(("assistant", str(direct)))
                if len(self._history) > 40:
                    self._history = self._history[-40:]
                return direct

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
                prompt, output_type=output_schema, **kwargs
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
        if (
            (isinstance(result, dict) and "error" in result)
            or routing_miss
            or not extracted.strip()
        ):
            # Direct fallback for stock quote lookups when LLM tool selection fails
            tickers = extract_tickers(prompt, limit=5)
            if tickers:
                try:
                    raw = await self._call_tool_direct(
                        "robinhood_get_equity_quotes",
                        {"symbols": tickers}
                    )
                    raw_str = raw if isinstance(raw, str) else str(raw)
                    ext = _extract_response(raw_str)
                    formatted = _format_tool_result("robinhood_get_equity_quotes", ext)
                    if formatted:
                        extracted = formatted
                        recovered = True
                except Exception as exc:
                    logger.debug("Direct tool fallback failed: %s", exc)

        # A routing miss means the user asked something with no tool behind it --
        # a greeting, a concept question, small talk. That deserves an answer, not
        # a report that routing failed. Ask the LLM directly, no tools offered.
        # Genuine errors (rate limits, auth, network) still surface unchanged.
        if not recovered and (routing_miss or not extracted.strip()):
            reply = await converse(self.config, prompt, history=self._history)
            extracted = reply if reply.strip() else SCOPE_MESSAGE

        # Step 3: store in history
        self._history.append(("user", prompt))
        self._history.append(("assistant", extracted))
        if len(self._history) > 40:
            # Find the nearest safe 'user' turn to avoid splitting assistant tool_call / tool_result pairs
            target_idx = len(self._history) - 40
            safe_idx = target_idx
            for i in range(target_idx, len(self._history)):
                turn = self._history[i]
                if isinstance(turn, (list, tuple)) and turn[0] == "user":
                    safe_idx = i
                    break
            self._history = self._history[safe_idx:]

        return extracted

    def clear_history(self) -> None:
        """Clear conversation history (new session)."""
        self._history = []

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
                    return "".join(chunks)
            except Exception as exc:
                logger.warning("Streaming failed (%s), falling back to non-streaming", exc)

        # Non-streaming fallback
        if _progress:
            _progress("analyzing...")
        return await self._llm_completion(prompt, _progress=_progress)

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
        if self._mcp_adapter:
            try:
                await self._mcp_adapter.close()
            except Exception:
                pass
        if self._rules_tmp and self._rules_tmp.exists():
            self._rules_tmp.unlink(missing_ok=True)

    async def __aenter__(self) -> "AgentHood":
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()
