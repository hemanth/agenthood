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

    # Patch Agent constructor and MCPToolAdapter
    with patch("agenthood.agent.Agent", return_value=mock_agent), \
         patch("agenthood.agent.TokenManager"), \
         patch("agenthood.agent.asyncio.create_task"), \
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

    async def test_chat_empty_response_gives_scope_message(self):
        """Empty model response falls back to scope message."""
        mock = _make_agent({"response": ""})
        hood = await _build_hood(mock)
        result = await hood.chat("tell me a joke")
        assert "trading assistant" in result

    async def test_chat_error_result(self):
        """Error dict is extracted via _extract_response."""
        mock = _make_agent({"error": "No appropriate tool found"})
        hood = await _build_hood(mock)
        result = await hood.chat("show my portfolio")
        mock.infer.assert_awaited_once()
        assert "No appropriate tool found" in result

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

