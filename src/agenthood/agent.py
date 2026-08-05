"""AgentHood -- core agent class.

Wraps agentu.Agent and wires every available agentu feature to a
Robinhood MCP backend.  The LLM discovers tools via search_tools(),
calls them directly, and formats its own output.

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
import httpx
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Type

from agentu import Agent

from .config import AgentHoodConfig, load_config
from .hooks import trading_guard
from .token import TokenManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # only show warnings/errors unless caller enables debug

SYSTEM_PROMPT = """\
You are AgentHood, a Robinhood trading assistant.

RULES:
1. Only answer about stocks, options, crypto, portfolio, and trading.
   Off-topic -> one-liner decline.
2. Ticker input (any case, e.g. "pypl", "PYPL", "aapl") -> uppercase it,
   call robinhood_get_equity_quotes(symbols=["PYPL"]).
3. Portfolio/positions -> call robinhood_get_accounts first for account_number.
4. ALWAYS review_equity_order BEFORE place_equity_order. Confirm with user.
5. Prefer limit orders. Explain risk for options/leveraged plays.
6. Use search_tools() to discover tools not in your active set.
7. After calling any tool, ALWAYS format the data as a clean markdown response.
   Never return raw JSON. Use tables for structured data.
8. For deeper analysis, load skills via get_skill_resource():
   - "technical-analysis" -- RSI, MACD, Bollinger interpretation + workflow
   - "risk-management"    -- position sizing, stop-loss, pre-trade checklist
   - "options-playbook"   -- options strategies, greeks, chain analysis
"""


# -- Response extraction -------------------------------------------------------

def _extract_response(result: Any) -> str:
    """Normalise whatever agentu.infer() returns into a plain string.

    agentu may return:
      - a plain str
      - a dict with 'text', 'content', 'result', or 'response'
      - an object with .text / .content
      - something else entirely (int, list, None)

    This function always returns a string.
    """
    if isinstance(result, str):
        # Check if it's raw Robinhood JSON leaked as a string
        stripped = result.strip()
        if stripped.startswith('{') and '"data"' in stripped and '"guide"' in stripped:
            try:
                import json
                parsed = json.loads(stripped)
                if isinstance(parsed, dict) and "data" in parsed:
                    return json.dumps(parsed["data"], indent=2)
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    if isinstance(result, dict):
        # Raw Robinhood tool response leaked through (model didn't format it)
        if "data" in result and "guide" in result:
            import json
            return json.dumps(result["data"], indent=2)
        for key in ("text", "content", "result", "response", "output"):
            val = result.get(key)
            if val is not None:
                if isinstance(val, (dict, list)):
                    import json
                    return json.dumps(val, indent=2)
                # String value might be raw Robinhood JSON -- recurse
                return _extract_response(val)
        # error key
        err = result.get("error")
        if err:
            return str(err)
        # Unknown dict -- pretty-print rather than repr
        import json
        return json.dumps(result, indent=2, default=str)

    for attr in ("text", "content", "result"):
        val = getattr(result, attr, None)
        if val is not None:
            return str(val)

    return str(result)


# -- Tool result formatting (no LLM call needed) ------------------------------

def _format_tool_result(tool_name: str, raw_json: str) -> str | None:
    """Format raw Robinhood JSON into clean markdown. Returns None if unrecognized."""
    import json as _json
    from datetime import datetime, timezone
    try:
        data = _json.loads(raw_json)
    except (_json.JSONDecodeError, TypeError):
        return None

    results = data.get("results") if isinstance(data, dict) else None
    if results is None and isinstance(data, list):
        results = data
    first = results[0] if (results and isinstance(results, list)) else {}

    def _price(v: str | None) -> str:
        """Clean price: $58.53 not $58.530000."""
        if not v:
            return "-"
        try:
            return f"${float(v):,.2f}"
        except (ValueError, TypeError):
            return str(v)

    def _time(ts: str | None) -> str:
        """Humanize timestamp."""
        if not ts:
            return "-"
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            diff = now - dt
            if diff.total_seconds() < 60:
                return "just now"
            if diff.total_seconds() < 3600:
                return f"{int(diff.total_seconds() / 60)}m ago"
            if diff.total_seconds() < 86400:
                return f"{int(diff.total_seconds() / 3600)}h ago"
            return dt.strftime("%b %d, %H:%M")
        except Exception:
            return str(ts)[:16]

    # -- Quotes --
    if "quote" in first:
        parts = []
        for entry in results:
            q = entry.get("quote", {})
            sym = q.get("symbol", "?")
            cur = float(q.get("last_trade_price", 0) or 0)
            prev = float(q.get("adjusted_previous_close", 0) or 0)
            chg = cur - prev if prev else 0
            pct = (chg / prev * 100) if prev else 0
            arrow = "^" if chg >= 0 else "v"
            sign = "+" if chg >= 0 else ""
            bid = q.get("bid_price")
            ask = q.get("ask_price")
            state = q.get("state", "?")
            trade_time = q.get("venue_last_trade_time")

            # Compute a preliminary signal / recommendation based on momentum
            if pct > 1.5:
                signal_fmt = "[bold green]BUY[/bold green]"
            elif pct < -1.5:
                signal_fmt = "[bold red]SELL[/bold red]"
            else:
                signal_fmt = "[bold yellow]HOLD[/bold yellow]"

            chg_color = "green" if chg >= 0 else "red"
            chg_fmt = f"[{chg_color}]{sign}{chg:.2f} ({sign}{pct:.2f}%)[/{chg_color}]"

            lines = [
                f"Ticker     : [bold cyan]{sym}[/bold cyan]",
                f"Last Price : [bold]{_price(str(cur))}[/bold] ({chg_fmt})",
                f"Verdict    : {signal_fmt}",
                f"Bid / Ask  : {_price(bid)} / {_price(ask)}",
                f"Prev Close : {_price(str(prev))}",
                f"Last Trade : {_time(trade_time)}",
                f"Status     : {state.title()}",
            ]
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # -- Portfolio --
    p_data = data.get("data", {}) if isinstance(data, dict) and "data" in data else data
    if isinstance(p_data, dict) and "total_value" in p_data:
        val = _price(p_data.get("total_value"))
        eq = _price(p_data.get("equity_value"))
        crypto = _price(p_data.get("crypto_value"))
        cash = _price(p_data.get("cash"))
        bp = _price(p_data.get("buying_power", {}).get("buying_power") if isinstance(p_data.get("buying_power"), dict) else p_data.get("buying_power"))
        lines = [
            f"Portfolio Value : [bold cyan]{val}[/bold cyan]",
            f"Equities        : [bold]{eq}[/bold]",
            f"Crypto          : [bold]{crypto}[/bold]",
            f"Cash            : [bold]{cash}[/bold]",
            f"Buying Power    : [bold green]{bp}[/bold green]"
        ]
        return "\n".join(lines)

    # -- Accounts --
    if "account_number" in first:
        lines = ["**Accounts**", "",
                 "| Account | Type | Nickname | Agentic |",
                 "|---------|------|----------|---------|"]
        for a in results:
            num = a.get("account_number", "?")
            masked = f"****{num[-4:]}" if len(num) > 4 else num
            btype = a.get("brokerage_account_type", "?").replace("_", " ").title()
            nick = a.get("nickname") or "-"
            agentic = "Yes" if a.get("agentic_allowed") else "No"
            lines.append(f"| {masked} | {btype} | {nick} | {agentic} |")
        return "\n".join(lines)

    # -- Search --
    if "symbol" in first and "name" in first:
        lines = ["| Symbol | Name |",
                 "|--------|------|"]
        for r in results:
            sym = r.get("symbol", "?")
            name = r.get("simple_name") or r.get("name", "?")
            lines.append(f"| **{sym}** | {name} |")
        return "\n".join(lines)

    return None


# -- AgentHood ----------------------------------------------------------------

class AgentHood:
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

        # Fast-fail: detect missing API key for cloud LLM providers
        _cloud_hosts = ("openai.com", "openrouter.ai", "anthropic.com", "googleapis.com")
        if any(h in (cfg.api_base or "") for h in _cloud_hosts) and not cfg.api_key:
            raise RuntimeError(
                "No API key configured for cloud LLM.\n"
                "  Option 1: run 'agenthood setup' to configure interactively\n"
                "  Option 2: add 'api_key' to ~/.agenthood/config.json\n"
                "  Option 3: export OPENROUTER_API_KEY=sk-or-..."
            )

        # -- Build Agent -------------------------------------------------------
        kwargs: dict[str, Any] = {
            "enable_memory": cfg.enable_memory,
            "auto_discover_rules": False,   # we inject rules ourselves
            "max_turns": cfg.max_turns,
            "codemode": cfg.codemode,
            "enable_rationale_recording": cfg.rationale_recording,
        }
        if cfg.model:
            kwargs["model"] = cfg.model
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base

        agent = Agent("agenthood", **kwargs)

        # -- System prompt -----------------------------------------------------
        # with_rules() needs a file path, so write a temp file
        self._rules_tmp = Path(tempfile.mktemp(suffix=".md"))
        self._rules_tmp.write_text(SYSTEM_PROMPT)
        agent.with_rules(str(self._rules_tmp))

        # -- Safety hooks ------------------------------------------------------
        guard = trading_guard(
            dry_run=cfg.dry_run,
            max_order_value=cfg.max_order_value,
            allowed_symbols=cfg.allowed_symbols,
        )
        agent.with_hooks(pre_tool=guard)

        # -- Permissions -------------------------------------------------------
        # Safety is enforced by the trading_guard pre-tool hook (blocks orders
        # in dry_run, enforces max_order_value, etc.) so we don't need agentu's
        # ask-writes mode -- that mode blocks ALL MCP tools since agentu marks
        # every MCP tool as WRITE, making infer() unusable for read queries.
        # Only explicitly raise the mode if the user chose it themselves.
        if cfg.permissions_mode in ("ask-writes", "plan"):
            agent.with_permissions(mode=cfg.permissions_mode)
        # Otherwise: leave at "auto" -- trading_guard is the safety layer.

        # -- Context compaction ------------------------------------------------
        # Auto-summarise old context when the window fills up
        agent.with_context(
            max_tokens=cfg.context_max_tokens,
            compaction=cfg.context_compaction,
            keep_recent=5,
            max_result_chars=3000,
        )

        # -- Memory consolidation ----------------------------------------------
        # Background LLM task that synthesises important memories every N turns
        if cfg.memory_consolidation:
            agent.with_consolidation(every=cfg.memory_consolidation_every)

        # -- CostTracker middleware --------------------------------------------
        try:
            from agentu.middleware.middleware import CostTracker
            self._cost_tracker = CostTracker()
            agent.use(self._cost_tracker)
        except ImportError:
            pass

        # -- Robinhood MCP connection ------------------------------------------
        # Bypass agent.with_mcp() -- Robinhood needs STREAMABLE_HTTP (MCP v2)
        from agentu.mcp.transport import MCPServerConfig, TransportType, AuthConfig
        from agentu.mcp.tool import MCPToolAdapter

        auth = AuthConfig.bearer_token(cfg.mcp_token) if cfg.mcp_token else None
        mcp_config = MCPServerConfig(
            name="robinhood",
            transport_type=TransportType.STREAMABLE_HTTP,
            url=cfg.mcp_url,
            auth=auth,
            timeout=30,
        )
        adapter = MCPToolAdapter(mcp_config)
        tools = await adapter.load_tools()
        self._all_tools = tools  # keep reference for _call_tool_direct

        # Minimal core tools visible to LLM directly to stay under input guardrails.
        # The rest are deferred behind search_tools() -- the model discovers them on demand.
        # Keep this set TINY: more tools = model picks wrong one for bare ticker queries.
        CORE = {
            "robinhood_get_equity_quotes",   # bare ticker queries -- the most common case
        }
        if cfg.core_tools_only:
            # Only load core tools, skip the rest entirely
            for tool in tools:
                if tool.name in CORE:
                    agent._add_tool_internal(tool, deferred=False)
        else:
            # Core tools always active; rest deferred behind search_tools
            for tool in tools:
                agent._add_tool_internal(tool, deferred=tool.name not in CORE)

        logger.info("Loaded %d Robinhood tools (deferred search enabled)", len(tools))

        # -- Skills ------------------------------------------------------------
        # Look for skills in multiple locations:
        #   1. Package-relative (editable install / repo checkout)
        #   2. Installed package data (pip install)
        _pkg_dir = Path(__file__).parent.parent.parent
        _alt_pkg_dir = Path(__file__).parent.parent  # src/agenthood/../ = src/
        skill_names = ["technical-analysis", "options-playbook", "risk-management"]
        skill_dirs = []
        for name in skill_names:
            for base in [_pkg_dir, _alt_pkg_dir.parent]:
                candidate = base / "skills" / name
                if (candidate / "SKILL.md").exists():
                    skill_dirs.append(str(candidate))
                    break
        skills_to_load = skill_dirs + list(cfg.skills)
        if skills_to_load:
            await agent.with_skills(skills_to_load)
            logger.info("Loaded %d trading skills", len(skills_to_load))

        # -- Sandbox -----------------------------------------------------------
        # Subprocess isolation for codemode -- limits memory, allows network for MCP
        if cfg.codemode and cfg.sandbox:
            agent.with_sandbox(codemode=True, max_memory_mb=256, allow_network=True)

        # -- Cache -------------------------------------------------------------
        if cfg.cache_preset:
            agent.with_cache(preset=cfg.cache_preset)
            logger.info("Cache enabled (preset=%s)", cfg.cache_preset)

        # -- Guardrails --------------------------------------------------------
        if cfg.guardrails:
            try:
                from agentu.middleware.guardrails import ContentFilter, MaxLength
                agent.with_guardrails(
                    input_guardrails=[MaxLength(max_chars=8000)],
                    output_guardrails=[ContentFilter(
                        block=["guaranteed profit", "100% return", "risk-free"]
                    )],
                    max_corrections=2,
                )
            except ImportError:
                pass

        # -- OpenTelemetry -----------------------------------------------------
        if cfg.otel_endpoint:
            try:
                agent.with_otel(endpoint=cfg.otel_endpoint, service_name="agenthood")
                logger.info("OpenTelemetry tracing enabled -> %s", cfg.otel_endpoint)
            except Exception as e:
                logger.warning("OTel setup failed: %s", e)

        # -- Notifications -----------------------------------------------------
        if cfg.notify_targets:
            try:
                agent.with_notifier(targets=cfg.notify_targets, title="AgentHood Alert")
                logger.info("Notifier configured: %s", cfg.notify_targets)
            except Exception as e:
                logger.warning("Notifier setup failed: %s", e)

        # -- Inbox watcher -----------------------------------------------------
        if cfg.inbox_path:
            inbox = Path(cfg.inbox_path)
            inbox.mkdir(parents=True, exist_ok=True)
            agent.with_inbox(str(inbox), poll_interval=cfg.inbox_poll_interval)
            await agent.start_inbox()
            logger.info("Signal inbox watching: %s", inbox)

        self._mcp_adapter = adapter
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

        # Fallback: if model failed or returned error, try calling appropriate tool directly.
        ext_clean = extracted.strip().lower()
        if (
            (isinstance(result, dict) and "error" in result)
            or ext_clean in ("", "no appropriate tool found", "no tool found", "no tool selected")
            or "no appropriate tool" in ext_clean
            or "no tool found" in ext_clean
        ):
            import re
            p_upper = prompt.upper().strip()

            # Direct fallback for stock quote lookups when LLM tool selection fails
            tickers = re.findall(r'\b([A-Z]{1,5})\b', p_upper)
            _STOP = {
                "I", "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO",
                "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON",
                "OR", "SO", "TO", "UP", "US", "WE", "THE", "AND", "FOR",
                "NOT", "YOU", "ALL", "CAN", "HAD", "HER", "WAS", "ONE",
                "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW",
                "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIM",
                "LET", "SAY", "SHE", "TOO", "USE", "BUY", "SELL", "WHAT",
                "SHOW", "PRICE", "STOCK", "QUOTE", "PORTFOLIO", "POSITIONS",
                "SHOULD", "LIKE", "LOOKS", "HOW"
            }
            tickers = [t for t in tickers if t not in _STOP]
            if tickers:
                try:
                    raw = await self._call_tool_direct(
                        "robinhood_get_equity_quotes",
                        {"symbols": tickers[:5]}
                    )
                    raw_str = raw if isinstance(raw, str) else str(raw)
                    ext = _extract_response(raw_str)
                    formatted = _format_tool_result("robinhood_get_equity_quotes", ext)
                    if formatted:
                        extracted = formatted
                except Exception as exc:
                    logger.debug("Direct tool fallback failed: %s", exc)

        # Fallback: if still empty, give scope message
        if not extracted.strip():
            extracted = (
                "I'm a trading assistant. "
                "Ask me about stocks, crypto, options, or your portfolio."
            )

        # Step 3: store in history
        self._history.append(("user", prompt))
        self._history.append(("assistant", extracted))
        if len(self._history) > 40:
            self._history = self._history[-40:]

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
        import json as _json

        if self._agent is None:
            await self.start()

        sym = ticker.upper()

        # Step 1: Fetch all data in PARALLEL (not sequential)
        if _progress:
            _progress(f"fetching {sym} data...")

        async def _fetch_one(label: str, base_name: str, params: dict) -> tuple[str, Any]:
            """Fetch a single data source, trying multiple name patterns."""
            candidates = [f"robinhood_{base_name}", base_name, f"rh_{base_name}"]
            for name in candidates:
                try:
                    raw = await self._call_tool_direct(name, params)
                    if isinstance(raw, str):
                        try:
                            return label, _json.loads(raw)
                        except _json.JSONDecodeError:
                            return label, raw
                    return label, raw
                except (ValueError, KeyError):
                    continue
                except Exception as exc:
                    return label, {"error": str(exc)}
            # Last resort
            try:
                raw = await self._agent.call(f"robinhood_{base_name}", params)
                if isinstance(raw, str):
                    try:
                        return label, _json.loads(raw)
                    except _json.JSONDecodeError:
                        return label, raw
                return label, raw
            except Exception as exc:
                return label, {"error": str(exc)}

        # start_time for 30 days back (ISO 8601)
        from datetime import datetime, timedelta, timezone
        _30d_ago = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

        tasks = [
            _fetch_one("quotes", "get_equity_quotes", {"symbols": [sym]}),
            _fetch_one("fundamentals", "get_equity_fundamentals", {"symbols": [sym]}),
            _fetch_one("technicals", "get_equity_technical_indicators", {
                "symbol": sym, "type": "sma", "interval": "day", "start_time": _30d_ago,
            }),
            _fetch_one("historicals", "get_equity_historicals", {
                "symbols": [sym], "interval": "day", "start_time": _30d_ago,
            }),
        ]
        results = await asyncio.gather(*tasks)
        data_sections = dict(results)

        if _progress:
            ok = sum(1 for _, v in results if not (isinstance(v, dict) and "error" in v))
            _progress(f"got {ok}/4 sources")

        # If ALL sections are errors, surface the problem
        all_errors = all(
            isinstance(v, dict) and "error" in v
            for v in data_sections.values()
        )
        if all_errors:
            tool_count = len(self._all_tools) if self._all_tools else 0
            tool_names = [t.name for t in (self._all_tools or [])[:5]]
            raise RuntimeError(
                f"Could not fetch any market data for {sym}. "
                f"MCP tools loaded: {tool_count}. "
                f"Sample names: {tool_names}. "
                f"Errors: {data_sections}"
            )

        # Step 2: Trim data to essential fields only
        trimmed = {}
        for key, val in data_sections.items():
            if isinstance(val, str) and val.startswith("ERROR"):
                trimmed[key] = val
                continue
            if isinstance(val, dict):
                inner = val.get("data", val)
                if isinstance(inner, dict) and "results" in inner:
                    inner = inner["results"]
                trimmed[key] = inner
            else:
                trimmed[key] = val

        data_dump = _json.dumps(trimmed, indent=1, default=str)
        if len(data_dump) > 4000:
            data_dump = data_dump[:4000] + "\n... (truncated)"

        prompt = f"""\
Analyze {sym}. Data:

{data_dump}

Reply in this format ONLY (no preamble):

## {sym} Analysis
| Metric | Value |
|---|---|
| Price | $X.XX |
| Day Change | +/-X.XX% |
| P/E | X.XX |
| Market Cap | $X.XXB |
| RSI | X.XX |
| 52w High/Low | $X.XX / $X.XX |

**Verdict**: BUY / SELL / HOLD -- one sentence why. Include bias (Bullish/Bearish/Neutral).

Use real numbers from the data. If missing, write N/A."""

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
        """Return (model, api_base, api_key) for the analyze LLM path.

        Prefers dedicated analyze config (e.g. Groq), falls back to main LLM.
        Also checks GROQ_API_KEY env var as a convenience.
        """
        cfg = self.config
        # Dedicated analyze provider configured?
        if cfg.analyze_model and cfg.analyze_api_base:
            key = cfg.analyze_api_key or os.environ.get("GROQ_API_KEY", "") or cfg.api_key
            return cfg.analyze_model, cfg.analyze_api_base, key
        # Fallback: main LLM
        return cfg.model, cfg.api_base, cfg.api_key

    async def _llm_completion(
        self,
        prompt: str,
        _progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Call the LLM directly with NO tool definitions.

        Used by analyze() to prevent the model from attempting tool calls
        when we just want it to analyze pre-fetched data.
        """
        import asyncio as _aio

        model, api_base, api_key = self._analyze_llm_config()
        logger.info("LLM completion using model=%s base=%s", model, api_base[:40])

        last_err = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=60) as http:
                    resp = await http.post(
                        f"{api_base}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "You are a financial analyst. Analyze the provided market data and produce a structured report. Do not refuse or say you cannot help."},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.3,
                            "max_tokens": 800,
                        },
                    )
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning("Rate limited (429), retrying in %ds...", wait)
                        await _aio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    body = resp.json()

                    # Handle missing choices (some providers return errors as 200)
                    choices = body.get("choices")
                    if not choices:
                        err_msg = body.get("error", {}).get("message", "No choices in response")
                        logger.warning("LLM returned no choices: %s", err_msg)
                        last_err = err_msg
                        continue

                    msg = choices[0].get("message", {})
                    content = msg.get("content") or ""

                    # Thinking models (nemotron) put text in reasoning if content is empty
                    if not content.strip() and msg.get("reasoning"):
                        content = msg["reasoning"]

                    if content.strip():
                        return content

                    last_err = "Empty response from LLM"
                    continue

            except Exception as exc:
                last_err = str(exc)
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)

        # All retries exhausted -- fall back to chat()
        logger.warning("All LLM direct call attempts failed (%s), falling back to chat()", last_err)
        return await self.chat(prompt, _progress=_progress)

    async def _stream_completion(self, prompt: str):
        """Async generator that yields tokens as they arrive via SSE streaming."""
        import json as _json

        model, api_base, api_key = self._analyze_llm_config()

        async with httpx.AsyncClient(timeout=120) as http:
            async with http.stream(
                "POST",
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst. Analyze the provided market data and produce a structured report. Do not refuse or say you cannot help."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800,
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(payload)
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token
                    except (_json.JSONDecodeError, IndexError, KeyError):
                        continue

    async def stream(self, prompt: str) -> None:
        """Stream response tokens to stdout as they arrive."""
        if self._agent is None:
            await self.start()
        async for token in self._agent.stream(prompt):
            print(token, end="", flush=True)
        print()

    # -- Memory ----------------------------------------------------------------

    def remember(self, content: str, memory_type: str = "fact", importance: float = 0.7) -> None:
        """Store a fact or observation in the agent's long-term memory."""
        if self._agent:
            self._agent.remember(
                content, memory_type=memory_type,
                importance=importance, store_long_term=True,
            )

    def recall(self, query: str, limit: int = 5) -> list:
        """Retrieve memories matching a query."""
        if self._agent is None:
            return []
        return self._agent.recall(query, limit=limit)

    def memory_stats(self) -> dict:
        """Return memory system statistics."""
        if self._agent is None:
            return {}
        return self._agent.get_memory_stats()

    # -- Direct tool access ----------------------------------------------------

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Directly invoke a Robinhood MCP tool by name (bypasses LLM)."""
        if self._agent is None:
            await self.start()
        return await self._agent.call(tool_name, params)

    # -- Cost tracking ---------------------------------------------------------

    def cost(self) -> dict:
        """Return API usage/cost stats for this session."""
        if self._cost_tracker is None:
            return {"error": "CostTracker not available"}
        return self._cost_tracker.get_usage()

    # -- Loop engineering ------------------------------------------------------

    async def ralph(
        self,
        prompt_file: str,
        max_iterations: int | None = None,
        timeout_minutes: int | None = None,
        on_iteration=None,
    ) -> dict:
        """Run the agent in Ralph autonomous loop mode.

        Ralph continuously reads a PROMPT.md file with checkboxes and
        loops until all are checked off or limits are reached.

        Example PROMPT.md::
            # Morning Scan
            - [ ] Check portfolio for positions down >5%
            - [ ] Flag any RSI > 75 as overbought
            - [ ] Summarise findings
        """
        if self._agent is None:
            await self.start()
        cfg = self.config
        return await self._agent.ralph(
            prompt_file=prompt_file,
            max_iterations=max_iterations or cfg.ralph_max_iterations,
            timeout_minutes=timeout_minutes or cfg.ralph_timeout_minutes,
            on_iteration=on_iteration,
        )

    async def delegate(self, task: str, max_corrections: int = 1, judges: int = 1) -> dict:
        """Delegate a task using the maker-checker pattern.

        A maker sub-agent executes the task; a checker reviews it.
        With judges>1 a panel votes -- use for high-stakes order decisions.
        """
        if self._agent is None:
            await self.start()
        return await self._agent.delegate(task, max_corrections=max_corrections, judges=judges)

    async def best_of(self, n: int, task: str, judge=None) -> dict:
        """Run N parallel agent instances on the same task; judge picks the best.

        Useful for high-confidence analysis:
            result = await agent.best_of(3, "Should I buy NVDA here?")
        """
        if self._agent is None:
            await self.start()
        return await self._agent.best_of(n=n, task=task, judge=judge)

    def findings(self, status: str = "pending", limit: int = 50) -> list[dict]:
        """Get findings from scheduled autonomous scans."""
        if self._agent is None:
            return []
        return self._agent.findings(status=status, limit=limit)

    # -- REST API server -------------------------------------------------------

    async def serve(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Start the AgentHood REST + WebSocket API server.

        Endpoints:
            GET  /            -> agent info
            GET  /health      -> health check
            GET  /tools       -> list tools
            POST /process     -> infer (natural language)
            GET  /process/stream -> SSE streaming
            WS   /ws          -> WebSocket
            POST /tasks       -> async background tasks
            GET  /tasks/{id}  -> poll task status

        Example::
            await agent.serve(host="0.0.0.0", port=8000)
        """
        if self._agent is None:
            await self.start()
        from agentu.runtime.serve import serve as _serve
        await _serve(self._agent, host=host, port=port, **kwargs)

    # -- Evaluation ------------------------------------------------------------

    async def evaluate(self, test_cases: list[dict], use_llm_judge: bool = False) -> dict:
        """Run a batch evaluation against predefined test cases.

        Each test case: {"ask": str, "expect": str, "expect_tool": str (optional)}

        Example::
            results = await agent.evaluate([
                {"ask": "price of AAPL", "expect_tool": "get_equity_quotes"},
                {"ask": "my portfolio value", "expect_tool": "get_portfolio"},
            ])
            print(f"Accuracy: {results['accuracy']:.0%}")
        """
        if self._agent is None:
            await self.start()
        from agentu.eval.eval import evaluate as _eval
        result = await _eval(self._agent, test_cases, use_llm_judge=use_llm_judge)
        return result.to_dict() if hasattr(result, "to_dict") else result

    # -- Token refresh loop ----------------------------------------------------

    async def _token_refresh_loop(self) -> None:
        """Background task: hot-swap the Robinhood JWT before it expires."""
        CHECK_INTERVAL = 240
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                new_token = self._token_manager.get_valid_token()
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
