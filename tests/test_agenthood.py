"""Integration-style tests for AgentHood using a fully mocked agentu agent.

These tests verify the full request flow:
  user input -> agent.infer -> _extract_response -> output

The MCP connection and LLM are replaced with AsyncMock so tests run offline
in milliseconds. Every public method of AgentHood is exercised.
"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# -- Helpers -------------------------------------------------------------------

def _make_agent(response_dict: dict):
    """Return a fake agentu Agent whose infer() returns response_dict."""
    mock = MagicMock()
    mock.infer = AsyncMock(return_value=response_dict)
    mock.stream = AsyncMock(return_value=_aiter(["tok1", " tok2"]))
    mock.remember = MagicMock()
    mock.recall = MagicMock(return_value=[{"content": "AAPL is volatile"}])
    mock.get_memory_stats = MagicMock(return_value={"count": 1})
    mock.call = AsyncMock(return_value={"data": {"price": 42}})
    mock.delegate = AsyncMock(return_value={"response": "delegate done"})
    mock.best_of = AsyncMock(return_value={"response": "best_of done"})
    mock.ralph = AsyncMock(return_value={"iterations": 2, "done": True})
    mock.findings = MagicMock(return_value=[{"id": 1, "text": "RSI overbought"}])
    mock.with_rules = MagicMock()
    mock.with_hooks = MagicMock()
    mock.with_permissions = MagicMock()
    mock.with_context = MagicMock()
    mock.with_consolidation = MagicMock()
    mock.with_tools = MagicMock()
    mock.with_skills = AsyncMock()
    mock.with_sandbox = MagicMock()
    mock.with_cache = MagicMock()
    mock.with_guardrails = MagicMock()
    mock.with_otel = MagicMock()
    mock.with_notifier = MagicMock()
    mock.with_inbox = MagicMock()
    mock.start_inbox = AsyncMock()
    mock.with_schedule = MagicMock()
    mock.start = AsyncMock()
    mock._add_tool_internal = MagicMock()
    mock.tools = []
    mock.search_tools = MagicMock()
    return mock


async def _aiter(items):
    for item in items:
        yield item


async def _build_hood(mock_agent) -> "AgentHood":
    """Instantiate AgentHood with all external calls mocked."""
    from agenthood.config import AgentHoodConfig

    cfg = AgentHoodConfig(
        mcp_token="test-token",
        api_key="test-key",
        api_base="https://openrouter.ai/api/v1",
        model="gpt-test",
        dry_run=True,
        guardrails=False,   # avoid optional import
        enable_memory=False,
        memory_consolidation=False,
        otel_endpoint="",
        notify_targets=[],
        inbox_path="",
        scan_interval=0,
        rationale_recording=False,
        codemode=False,
        sandbox=False,
        cache_preset="",
        core_tools_only=False,
        skills=[],
    )

    # Patch Agent constructor and MCPToolAdapter.
    # NB: these must target the modules where the names are *used*, not
    # agenthood.agent -- that module now only re-exports them.
    with patch("agenthood.core.builder.Agent", return_value=mock_agent), \
         patch("agenthood.core.agent.TokenManager"), \
         patch("agenthood.core.agent.asyncio.create_task"), \
         patch("agentu.mcp.transport.MCPServerConfig"), \
         patch("agentu.mcp.tool.MCPToolAdapter") as MockAdapter:

        mock_adapter = MagicMock()
        mock_adapter.load_tools = AsyncMock(return_value=[])
        MockAdapter.return_value = mock_adapter

        from agenthood.agent import AgentHood
        hood = AgentHood(cfg)
        hood._agent = mock_agent  # inject directly -- skip real start()
        hood._mcp_adapter = mock_adapter
        return hood


# -- chat() --------------------------------------------------------------------

@pytest.mark.asyncio
class TestChat:

    async def test_chat_plain_text(self):
        """Non-watch prompt goes through infer()."""
        mock = _make_agent({"response": "PYPL is trading at $68"})
        hood = await _build_hood(mock)
        result = await hood.chat("what is PYPL trading at?")
        assert "PYPL" in result or "68" in result
        mock.infer.assert_awaited_once()

    async def test_chat_bare_ticker_goes_to_llm(self):
        """Bare ticker goes through infer() now (no more _direct_dispatch)."""
        mock = _make_agent({"response": "PYPL is at $68.50"})
        hood = await _build_hood(mock)
        result = await hood.chat("PYPL")
        mock.infer.assert_awaited_once()
        assert "PYPL" in mock.infer.call_args[0][0]

    async def test_chat_full_sentence_passes_through(self):
        """Full sentence is passed to infer() unmodified."""
        mock = _make_agent({"response": "ok"})
        hood = await _build_hood(mock)
        await hood.chat("buy 10 shares of NVDA")
        call_args = mock.infer.call_args[0][0]
        assert call_args == "buy 10 shares of NVDA"

    async def test_chat_empty_response_converses(self):
        """Empty model response falls through to a conversational reply."""
        mock = _make_agent({"response": ""})
        hood = await _build_hood(mock)
        with patch("agenthood.core.agent.converse",
                   AsyncMock(return_value="Sure -- which ticker?")) as conv:
            result = await hood.chat("tell me a joke")
        conv.assert_awaited_once()
        assert result == "Sure -- which ticker?"

    async def test_chat_routing_miss_answers_instead_of_reporting(self):
        """A routing miss must produce an answer, never agentu's sentinel."""
        mock = _make_agent({"error": "No appropriate tool found"})
        hood = await _build_hood(mock)
        with patch("agenthood.core.agent.converse",
                   AsyncMock(return_value="Hi! What can I look up for you?")):
            result = await hood.chat("hi")
        mock.infer.assert_awaited_once()
        assert "No appropriate tool" not in result
        assert "tool" not in result.lower()
        assert result == "Hi! What can I look up for you?"

    async def test_chat_greeting_does_not_call_a_tool(self):
        """'hi' is not a ticker -- it must not trigger a quote lookup."""
        mock = _make_agent({"error": "No appropriate tool found"})
        hood = await _build_hood(mock)
        hood._call_tool_direct = AsyncMock(
            side_effect=AssertionError("must not call a tool for a greeting")
        )
        with patch("agenthood.core.agent.converse",
                   AsyncMock(return_value="Hey there.")):
            result = await hood.chat("hi")
        assert result == "Hey there."

    async def test_chat_converse_passes_history(self):
        """The conversational path gets prior turns so it can follow context."""
        mock = _make_agent({"error": "No appropriate tool found"})
        hood = await _build_hood(mock)
        hood._history = [("user", "AAPL"), ("assistant", "AAPL is $310")]
        with patch("agenthood.core.agent.converse",
                   AsyncMock(return_value="It closed at $310.")) as conv:
            await hood.chat("what about it")
        assert conv.await_args.kwargs["history"] == hood._history[:2] or \
               conv.await_args.kwargs["history"][:2] == [
                   ("user", "AAPL"), ("assistant", "AAPL is $310")]

    async def test_chat_converse_failure_falls_back_to_scope(self):
        """If the direct LLM call also fails, show a plain line -- no internals."""
        mock = _make_agent({"error": "No appropriate tool found"})
        hood = await _build_hood(mock)
        with patch("agenthood.core.agent.converse", AsyncMock(return_value="")):
            result = await hood.chat("hi")
        assert "trading assistant" in result
        assert "tool" not in result.lower()

    async def test_chat_real_error_still_surfaces(self):
        """Genuine failures are NOT swallowed."""
        mock = _make_agent({"error": "429 rate limit exceeded"})
        hood = await _build_hood(mock)
        result = await hood.chat("what is my buying power")
        assert "429" in result or "rate limit" in result.lower()

    async def test_chat_stores_history(self):
        """History is appended after each chat call."""
        mock = _make_agent({"response": "done"})
        hood = await _build_hood(mock)
        await hood.chat("AAPL price")
        assert len(hood._history) == 2
        assert hood._history[0] == ("user", "AAPL price")

    async def test_chat_history_cap(self):
        """History is capped at 40 entries (20 turns)."""
        mock = _make_agent({"response": "ok"})
        hood = await _build_hood(mock)
        for i in range(25):
            await hood.chat(f"turn {i}")
        assert len(hood._history) <= 40

    async def test_chat_watch_prompt_returns_sentinel(self):
        """Watch prompt returns __watch__ sentinel without calling infer()."""
        mock = _make_agent({"response": "ok"})
        hood = await _build_hood(mock)
        result = await hood.chat("watch AAPL below 150")
        assert isinstance(result, dict)
        assert "__watch__" in result
        assert result["__watch__"]["ticker"] == "AAPL"
        mock.infer.assert_not_awaited()

    async def test_chat_output_schema(self):
        """output_schema passes through to infer() and returns raw result."""
        from pydantic import BaseModel
        class PriceInfo(BaseModel):
            price: float
        mock = _make_agent(PriceInfo(price=42.0))
        hood = await _build_hood(mock)
        result = await hood.chat("AAPL price", output_schema=PriceInfo)
        assert result.price == 42.0

    async def test_chat_progress_callback(self):
        """_progress callback is called during chat."""
        mock = _make_agent({"response": "done"})
        hood = await _build_hood(mock)
        steps = []
        await hood.chat("AAPL", _progress=steps.append)
        assert any("thinking" in s for s in steps)

    async def test_clear_history(self):
        mock = _make_agent({"response": "ok"})
        hood = await _build_hood(mock)
        await hood.chat("test")
        assert len(hood._history) > 0
        hood.clear_history()
        assert len(hood._history) == 0


# -- stream() ------------------------------------------------------------------

@pytest.mark.asyncio
class TestStream:

    async def test_stream_calls_agent(self, capsys):
        mock = _make_agent({})

        async def fake_stream(prompt):
            yield "tok"

        mock.stream = fake_stream
        hood = await _build_hood(mock)
        await hood.stream("AAPL")
        out = capsys.readouterr().out
        assert "tok" in out


# -- memory API ----------------------------------------------------------------

@pytest.mark.asyncio
class TestMemory:

    async def test_remember_calls_agent(self):
        mock = _make_agent({"response": "ok"})
        hood = await _build_hood(mock)
        hood.remember("NVDA RSI above 75", memory_type="signal", importance=0.9)
        mock.remember.assert_called_once()
        args, kwargs = mock.remember.call_args
        assert args[0] == "NVDA RSI above 75"

    async def test_recall_returns_list(self):
        mock = _make_agent({"response": "ok"})
        hood = await _build_hood(mock)
        results = hood.recall("AAPL volatility")
        assert isinstance(results, list)
        assert len(results) > 0

    async def test_recall_before_start_returns_empty(self):
        from agenthood.agent import AgentHood
        from agenthood.config import AgentHoodConfig
        hood = AgentHood(AgentHoodConfig(mcp_token="x"))
        assert hood.recall("anything") == []

    async def test_memory_stats_before_start(self):
        from agenthood.agent import AgentHood
        from agenthood.config import AgentHoodConfig
        hood = AgentHood(AgentHoodConfig(mcp_token="x"))
        assert hood.memory_stats() == {}

    async def test_memory_stats_returns_dict(self):
        mock = _make_agent({"response": "ok"})
        hood = await _build_hood(mock)
        stats = hood.memory_stats()
        assert isinstance(stats, dict)


# -- call_tool() ---------------------------------------------------------------

@pytest.mark.asyncio
class TestCallTool:

    async def test_call_tool_passes_params(self):
        mock = _make_agent({})
        hood = await _build_hood(mock)
        await hood.call_tool("robinhood_get_equity_quotes", {"symbols": ["AAPL"]})
        mock.call.assert_called_once_with(
            "robinhood_get_equity_quotes", {"symbols": ["AAPL"]}
        )

    async def test_call_tool_returns_result(self):
        mock = _make_agent({})
        mock.call = AsyncMock(return_value={"data": {"bid": 200}})
        hood = await _build_hood(mock)
        result = await hood.call_tool("any_tool", {})
        assert result == {"data": {"bid": 200}}


# -- cost() --------------------------------------------------------------------

class TestCost:
    def test_cost_before_cost_tracker(self):
        from agenthood.agent import AgentHood
        from agenthood.config import AgentHoodConfig
        hood = AgentHood(AgentHoodConfig(mcp_token="x"))
        result = hood.cost()
        assert "error" in result


# -- delegate / best_of --------------------------------------------------------

@pytest.mark.asyncio
class TestLoopEngineering:

    async def test_delegate_called_with_task(self):
        mock = _make_agent({})
        hood = await _build_hood(mock)
        await hood.delegate("Analyse NVDA risk", max_corrections=2, judges=3)
        mock.delegate.assert_called_once()
        kwargs = mock.delegate.call_args
        assert "Analyse NVDA risk" in str(kwargs)

    async def test_best_of_called(self):
        mock = _make_agent({})
        hood = await _build_hood(mock)
        await hood.best_of(3, "Should I buy TSLA?")
        mock.best_of.assert_called_once_with(n=3, task="Should I buy TSLA?", judge=None)

    async def test_findings_before_start(self):
        from agenthood.agent import AgentHood
        from agenthood.config import AgentHoodConfig
        hood = AgentHood(AgentHoodConfig(mcp_token="x"))
        assert hood.findings() == []

    async def test_findings_returns_list(self):
        mock = _make_agent({})
        hood = await _build_hood(mock)
        results = hood.findings(status="pending", limit=10)
        assert isinstance(results, list)


# -- System prompt -------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_contains_tool_discovery(self):
        from agenthood.agent import SYSTEM_PROMPT
        assert "search_tools" in SYSTEM_PROMPT

    def test_system_prompt_contains_ticker_example(self):
        from agenthood.agent import SYSTEM_PROMPT
        assert "PYPL" in SYSTEM_PROMPT or "ticker" in SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_safety_rules(self):
        from agenthood.agent import SYSTEM_PROMPT
        assert "review_equity_order" in SYSTEM_PROMPT

    def test_system_prompt_contains_dry_run_note(self):
        from agenthood.agent import SYSTEM_PROMPT
        assert "limit" in SYSTEM_PROMPT.lower() or "market order" in SYSTEM_PROMPT.lower()


# -- _call_tool_direct ---------------------------------------------------------

@pytest.mark.asyncio
class TestCallToolDirect:

    async def test_finds_tool_in_all_tools(self):
        """_call_tool_direct looks up tools from _all_tools, not agent.tools."""
        mock = _make_agent({})
        hood = await _build_hood(mock)

        # Create a fake tool
        fake_tool = MagicMock()
        fake_tool.name = "robinhood_get_equity_quotes"
        fake_tool.function = AsyncMock(return_value={"price": 42})

        hood._all_tools = [fake_tool]

        result = await hood._call_tool_direct(
            "robinhood_get_equity_quotes", {"symbols": ["AAPL"]}
        )
        fake_tool.function.assert_awaited_once_with(symbols=["AAPL"])
        assert result == {"price": 42}

    async def test_raises_for_missing_tool(self):
        """_call_tool_direct raises ValueError for unknown tool name."""
        mock = _make_agent({})
        hood = await _build_hood(mock)
        hood._all_tools = []

        with pytest.raises(ValueError, match="not found"):
            await hood._call_tool_direct("nonexistent_tool", {})

    async def test_handles_sync_tool_function(self):
        """_call_tool_direct handles sync (non-async) tool functions."""
        mock = _make_agent({})
        hood = await _build_hood(mock)

        fake_tool = MagicMock()
        fake_tool.name = "robinhood_get_accounts"
        fake_tool.function = MagicMock(return_value={"accounts": []})

        hood._all_tools = [fake_tool]

        result = await hood._call_tool_direct("robinhood_get_accounts", {})
        fake_tool.function.assert_called_once_with()
        assert result == {"accounts": []}


# -- stop() --------------------------------------------------------------------

@pytest.mark.asyncio
class TestStop:

    async def test_stop_cancels_refresh_task(self):
        mock = _make_agent({})
        hood = await _build_hood(mock)
        # Create a real task that we can cancel and await
        async def _forever():
            await asyncio.sleep(999)
        hood._refresh_task = asyncio.create_task(_forever())
        hood._mcp_adapter = None  # skip adapter close
        await hood.stop()
        assert hood._refresh_task.cancelled()

    async def test_stop_closes_adapter(self):
        mock = _make_agent({})
        hood = await _build_hood(mock)
        hood._refresh_task = None
        mock_adapter = MagicMock()
        mock_adapter.close = AsyncMock()
        hood._mcp_adapter = mock_adapter
        await hood.stop()
        mock_adapter.close.assert_awaited_once()


# -- analyze() ----------------------------------------------------------------

@pytest.mark.asyncio
class TestAnalyze:

    async def test_analyze_fetches_data_then_sends_to_llm(self):
        """analyze() calls 4 MCP tools directly, then sends data to LLM."""
        mock = _make_agent({})
        hood = await _build_hood(mock)

        # Track direct tool calls
        tool_calls = []
        async def _fake_tool(name, params):
            tool_calls.append(name)
            return {"data": f"mock_{name}"}
        hood._call_tool_direct = _fake_tool

        # Mock the direct LLM call
        captured = []
        async def _fake_llm(prompt, **kwargs):
            captured.append(prompt)
            return "## NVDA Market Analysis\n..."
        hood._llm_completion = _fake_llm

        result = await hood.analyze("nvda")

        # Verify all 4 tools were called directly
        assert "robinhood_get_equity_quotes" in tool_calls
        assert "robinhood_get_equity_fundamentals" in tool_calls
        assert "robinhood_get_equity_technical_indicators" in tool_calls
        assert "robinhood_get_equity_historicals" in tool_calls

        # Verify the LLM received the fetched data
        assert len(captured) == 1
        prompt = captured[0]
        assert "NVDA" in prompt
        assert "mock_robinhood_get_equity_quotes" in prompt
        assert "Verdict" in prompt
        assert "NVDA Market Analysis" in result

    async def test_analyze_handles_tool_errors_gracefully(self):
        """If a tool call fails, analyze() includes the error and continues."""
        mock = _make_agent({})
        hood = await _build_hood(mock)

        call_count = 0
        async def _failing_tool(name, params):
            nonlocal call_count
            call_count += 1
            if "fundamentals" in name:
                raise ConnectionError("MCP down")
            return {"data": "ok"}
        hood._call_tool_direct = _failing_tool

        captured = []
        async def _fake_llm(prompt, **kwargs):
            captured.append(prompt)
            return "analysis with partial data"
        hood._llm_completion = _fake_llm

        result = await hood.analyze("AAPL")
        assert call_count == 4  # all 4 attempted
        assert "MCP down" in captured[0]  # error included in prompt
        assert "analysis with partial data" in result

    async def test_analyze_uppercases_ticker(self):
        """analyze() uppercases the ticker."""
        mock = _make_agent({})
        hood = await _build_hood(mock)

        tool_calls = []
        async def _fake_tool(name, params):
            tool_calls.append(params)
            return {}
        hood._call_tool_direct = _fake_tool

        captured = []
        async def _fake_llm(prompt, **kwargs):
            captured.append(prompt)
            return "done"
        hood._llm_completion = _fake_llm

        await hood.analyze("aapl")
        assert "AAPL" in captured[0]
        # All tool calls should use uppercase symbol
        for params in tool_calls:
            if "symbols" in params:
                assert params["symbols"] == ["AAPL"]

# -- New Tests -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_watch_fastpath_caps_history():
    mock = _make_agent({"response": "ok"})
    hood = await _build_hood(mock)
    hood._all_tools = []
    
    # Mock _direct_dispatch if it exists, or just send a watch prompt which chat() handles natively
    for i in range(25):
        await hood.chat(f"watch AAPL below {150+i}")
    
    assert len(hood._history) <= 40

@pytest.mark.asyncio
async def test_get_account_number_caches():
    mock = _make_agent({})
    hood = await _build_hood(mock)
    
    fake_tool = AsyncMock(return_value={"accounts": [{"account_number": "12345"}]})
    hood._call_tool_direct = fake_tool
    
    num1 = await hood._get_account_number()
    num2 = await hood._get_account_number()
    
    assert num1 == "12345"
    assert num2 == "12345"
    fake_tool.assert_awaited_once()



# ── Module decomposition regressions ──────────────────────────────────────────
# These lock in behaviour that the split could silently break: path arithmetic
# relative to __file__, and the back-compat re-export shims.

def test_bundled_skills_are_discoverable():
    """discover_skill_dirs() resolves paths relative to its own __file__.

    Moving it to a different nesting depth silently returns [] -- the agent
    then starts with no trading skills and no error.
    """
    from pathlib import Path
    from agenthood.core.builder import discover_skill_dirs, BUNDLED_SKILLS
    dirs = discover_skill_dirs()
    assert len(dirs) == len(BUNDLED_SKILLS), f"expected all bundled skills, got {dirs}"
    for d in dirs:
        assert Path(d, "SKILL.md").exists()


def test_skills_live_inside_the_package():
    """Regression: the wheel shipped no skills, so pip installs had none.

    They sat at the repo root, outside `packages = ["src/agenthood"]`, so every
    `pip install agenthood` produced an agent with zero trading knowledge and no
    error. Keeping them under the package directory is what puts them in the
    wheel -- assert on location, not just discoverability, because a repo
    checkout would pass the discovery test either way.
    """
    from pathlib import Path
    import agenthood
    from agenthood.core.builder import discover_skill_dirs, BUNDLED_SKILLS

    pkg_dir = Path(agenthood.__file__).parent
    for name in BUNDLED_SKILLS:
        assert (pkg_dir / "skills" / name / "SKILL.md").exists(), (
            f"{name} must live in agenthood/skills/ or it will not ship in the wheel"
        )
    # and discovery must actually resolve to that in-package copy
    assert all(str(pkg_dir) in d for d in discover_skill_dirs())


def test_agent_module_shim_reexports():
    """agenthood.agent kept working after the move to agenthood.core."""
    import agenthood
    from agenthood.agent import (
        AgentHood as ShimAgentHood,
        SYSTEM_PROMPT as shim_prompt,
        _extract_response as shim_extract,
        _format_tool_result as shim_format,
    )
    from agenthood.core.agent import AgentHood as CoreAgentHood
    from agenthood.core.prompts import SYSTEM_PROMPT as core_prompt

    assert ShimAgentHood is CoreAgentHood is agenthood.AgentHood
    assert shim_prompt is core_prompt
    assert callable(shim_extract) and callable(shim_format)


def test_extract_tickers_drops_stopwords():
    """The failure-path ticker heuristic must not fire on ordinary questions."""
    from agenthood.core.tickers import extract_tickers
    assert extract_tickers("WHAT CAN YOU DO") == []
    assert extract_tickers("price of AAPL") == ["AAPL"]
    assert extract_tickers("AAPL NVDA MSFT TSLA GOOG AMZN", limit=5) == [
        "AAPL", "NVDA", "MSFT", "TSLA", "GOOG"
    ]


def test_extract_tickers_is_case_sensitive_in_sentences():
    """Regression: 'tell me a joke' answered with a quote for A (Agilent).

    Upper-casing the whole message first turns ordinary words into symbols. In a
    sentence, only words the user actually typed in caps may count.
    """
    from agenthood.core.tickers import extract_tickers
    # Lowercase prose yields nothing, however ticker-shaped the words look
    assert extract_tickers("tell me a joke") == []
    assert extract_tickers("hey there") == []
    assert extract_tickers("what is an etf") == []
    # A bare word is a ticker in either case
    assert extract_tickers("nvda") == ["NVDA"]
    assert extract_tickers("NVDA") == ["NVDA"]
    # Caps inside a sentence still resolve
    assert extract_tickers("is TSLA overbought?") == ["TSLA"]
    assert extract_tickers("show me AAPL and NVDA") == ["AAPL", "NVDA"]
    # Pronoun "I" and finance jargon are not symbols
    assert extract_tickers("I want to buy AAPL") == ["AAPL"]
    assert extract_tickers("what is an ETF?") == []


def test_system_prompt_forbids_tool_use_for_meta_questions():
    """Regression: 'what can you do?' answered with a stale ticker quote.

    get_equity_quotes is the only non-deferred tool, so the model reaches for it
    and reuses a ticker from context. The prompt must rule that out explicitly.
    """
    from agenthood.core.prompts import SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "what can you do" in lowered
    assert "never reuse a ticker" in lowered
    assert "no tool." in lowered


@pytest.mark.asyncio
async def test_analyze_failure_does_not_route_through_tool_calling():
    """Regression: a 429 on the analyze path produced quote spam.

    llm_completion used to fall back to chat(), which sent the multi-KB market
    data dump through tool selection. The router then quoted whatever symbols
    appeared inside the JSON -- 'agenthood analyze MSFT' returned quotes for
    MSFT, III and others with raw Rich markup.
    """
    from agenthood.core import llm

    mock = _make_agent({"response": "should never be reached"})
    hood = await _build_hood(mock)
    hood.chat = AsyncMock(side_effect=AssertionError(
        "analyze must not fall back to the tool-calling path"
    ))

    with patch.object(llm, "_completion",
                      AsyncMock(return_value=("", "rate limited (429)", False))):
        out = await hood._llm_completion("Analyze MSFT. Data: {...}")

    assert "429" in out
    assert "Could not generate the analysis" in out
    hood.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_completion_uses_model_fallbacks():
    """A rate-limited primary should roll over to a configured fallback model."""
    from agenthood.core import llm

    mock = _make_agent({"response": "x"})
    hood = await _build_hood(mock)
    hood.config.model_fallbacks = ["nvidia/nemotron-3-super-120b-a12b:free"]

    calls = []

    async def fake(model, base, key, messages, **kw):
        calls.append((model, base))
        if len(calls) == 1:
            return "", "rate limited (429)", False
        return "Analysis text", None, False

    with patch.object(llm, "_completion", fake):
        out = await hood._llm_completion("Analyze MSFT.")

    assert out == "Analysis text"
    assert len(calls) == 2
    # the fallback is an OpenRouter id, so it must be routed to OpenRouter
    assert "openrouter.ai" in calls[1][1]


@pytest.mark.asyncio
async def test_reasoning_text_is_last_resort():
    """Chain-of-thought must not win over a real answer from a later model.

    Thinking models leave `content` empty and put CoT in `reasoning`. Accepting
    that immediately printed 'We need to produce a structured report...' instead
    of the report.
    """
    from agenthood.core import llm

    mock = _make_agent({"response": "x"})
    hood = await _build_hood(mock)
    hood.config.model_fallbacks = ["some/thinking-model:free", "other/clean-model:free"]

    seq = [
        ("", "rate limited (429)", False),      # primary: down
        ("thinking out loud...", None, True),   # fallback 1: CoT only
        ("## Real Analysis", None, False),      # fallback 2: actual answer
    ]

    async def fake(model, base, key, messages, **kw):
        return seq.pop(0)

    with patch.object(llm, "_completion", fake):
        out = await hood._llm_completion("Analyze MSFT.")

    assert out == "## Real Analysis"


@pytest.mark.asyncio
async def test_reasoning_text_used_when_nothing_better():
    """If every model returns only reasoning, show it rather than nothing."""
    from agenthood.core import llm

    mock = _make_agent({"response": "x"})
    hood = await _build_hood(mock)
    hood.config.model_fallbacks = []

    with patch.object(llm, "_completion",
                      AsyncMock(return_value=("salvaged text", None, True))):
        out = await hood._llm_completion("Analyze MSFT.")

    assert out == "salvaged text"


@pytest.mark.asyncio
async def test_clear_history_clears_agentu_context():
    """Regression: .new cleared a list nothing read.

    The context the model actually sees lives on the agentu Agent, so clearing
    only AgentHood._history left the previous conversation fully in play.
    """
    mock = _make_agent({"response": "ok"})
    mock.conversation_history = [{"role": "user", "content": "AAPL"}]
    mock.clear_short_term_memory = MagicMock()

    hood = await _build_hood(mock)
    await hood.chat("AAPL")
    assert hood._history

    hood.clear_history()

    assert hood._history == []
    assert mock.conversation_history == []
    mock.clear_short_term_memory.assert_called_once()


# ── Rate-limit backoff ────────────────────────────────────────────────────────

class TestRetryAfterHandling:
    """Regression: 'hi' spent 3.1s asleep on a limit it could never wait out.

    Groq's per-day quota answers 429 with 'retry-after: 202'. Backing off 1s
    then 2s against that burns three seconds and still fails, then falls to a
    slow free model anyway.
    """

    def setup_method(self):
        from agenthood.core.llm import reset_cooldowns
        reset_cooldowns()

    teardown_method = setup_method

    def _resp(self, retry_after):
        m = MagicMock()
        m.status_code = 429
        m.headers = {"retry-after": str(retry_after)} if retry_after else {}
        return m

    @pytest.mark.asyncio
    async def test_long_retry_after_skips_sleeping(self):
        from agenthood.core import llm
        sleeps = []

        async def no_sleep(s):
            sleeps.append(s)

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=self._resp(202))

        with patch.object(llm.httpx, "AsyncClient", return_value=client), \
             patch.object(llm.asyncio, "sleep", no_sleep):
            content, err, _ = await llm._completion("m", "http://b", "k", [], max_retries=3)

        assert content == ""
        assert sleeps == [], "must not sleep against a 202s limit"
        assert client.post.await_count == 1, "must not retry a limit it cannot wait out"

    @pytest.mark.asyncio
    async def test_short_retry_after_still_retries(self):
        from agenthood.core import llm
        sleeps = []

        async def no_sleep(s):
            sleeps.append(s)

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=self._resp(1))

        with patch.object(llm.httpx, "AsyncClient", return_value=client), \
             patch.object(llm.asyncio, "sleep", no_sleep):
            await llm._completion("m2", "http://b", "k", [], max_retries=3)

        assert sleeps, "a 1s limit is worth waiting for"

    @pytest.mark.asyncio
    async def test_cooldown_skips_model_on_later_calls(self):
        """Once a model reports a long limit, later turns must not re-ask it."""
        from agenthood.core import llm

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=self._resp(300))

        with patch.object(llm.httpx, "AsyncClient", return_value=client):
            await llm._completion("cold", "http://b", "k", [])
            first = client.post.await_count
            await llm._completion("cold", "http://b", "k", [])

        assert client.post.await_count == first, "second call must be short-circuited"

    def test_reset_cooldowns_clears_state(self):
        from agenthood.core import llm
        llm._set_cooldown("x", 100)
        assert llm._cooling_down("x") > 0
        llm.reset_cooldowns()
        assert llm._cooling_down("x") == 0


# ── Account priority, Groq key validation, and setup_llm regressions ─────────

@pytest.mark.asyncio
async def test_get_account_number_prefers_agentic_then_default():
    mock = _make_agent({})
    hood = await _build_hood(mock)
    hood._call_tool_direct = AsyncMock(return_value={
        "data": {
            "accounts": [
                {"account_number": "FIRST", "is_default": True, "agentic_allowed": False},
                {"account_number": "AGENTIC", "is_default": False, "agentic_allowed": True},
            ]
        }
    })
    assert await hood._get_account_number() == "AGENTIC"


def test_check_api_key_rejects_groq_without_key():
    from agenthood.config import AgentHoodConfig
    from agenthood.core.builder import _check_api_key

    cfg = AgentHoodConfig(api_base="https://api.groq.com/openai/v1", api_key="")
    with pytest.raises(RuntimeError, match="No API key"):
        _check_api_key(cfg)


def test_setup_llm_openrouter_save_key_does_not_raise_name_error(monkeypatch):
    """Regression: setup_llm assigned to undefined cfg['api_key'] instead of result['api_key']."""
    from agenthood import setup as S

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-123")
    answers = iter(["y", "1", "y"])  # use env -> model #1 -> save key = y
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    out = S.setup_llm()
    assert out["api_key"] == "sk-or-test-123"
    assert out["api_base"] == "https://openrouter.ai/api/v1"
