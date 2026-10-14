import pytest
from unittest.mock import MagicMock
from agenthood.tui import handle_meta, _resolve_model

def test_handle_meta_analyze_returns_ticker():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    res = handle_meta('.analyze NVDA', mock_agent)
    assert res == ['NVDA']

def test_handle_meta_analyze_multi_tickers():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    res = handle_meta('.analyze PYPL, GOOGL, NVDA', mock_agent)
    assert res == ['PYPL', 'GOOGL', 'NVDA']

def test_handle_meta_analyze_resolves_names():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    # "Google" and "Nvidia" get uppercased but stay as-is in handle_meta
    # (resolution happens async in the REPL loop)
    res = handle_meta('.analyze PYPL Google Nvidia', mock_agent)
    assert res == ['PYPL', 'GOOGLE', 'NVIDIA']

def test_handle_meta_analyze_no_ticker():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    mock_agent.last_ticker = None
    res = handle_meta('.analyze', mock_agent)
    assert res is True

def test_handle_meta_analyze_shorthand_and_context():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    mock_agent.last_ticker = 'AAPL'
    assert handle_meta('.a NVDA', mock_agent) == ['NVDA']
    assert handle_meta('.analyze', mock_agent) == ['AAPL']
    assert handle_meta('.a', mock_agent) == ['AAPL']

def test_handle_meta_quote_bare_and_context():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    mock_agent.last_ticker = None
    # Bare .q without last_ticker prints usage and returns True (never leaks to LLM)
    assert handle_meta('.q', mock_agent) is True
    # Bare .q with last_ticker quotes that ticker
    mock_agent.last_ticker = 'NVDA'
    assert handle_meta('.q', mock_agent) == ('__quote__', 'NVDA')
    assert handle_meta('.quote', mock_agent) == ('__quote__', 'NVDA')

def test_handle_meta_watches_history_status():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    mock_agent._history = [('user', 'NVDA'), ('assistant', 'NVDA is $219')]
    mock_agent.last_ticker = 'NVDA'
    assert handle_meta('.watches', mock_agent) == '__watches__'
    assert handle_meta('.history', mock_agent) is True
    assert handle_meta('.status', mock_agent) is True

def test_handle_meta_unknown_dot_command_caught_locally():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    # Typos starting with '.' are handled locally (return True), never sent to LLM
    assert handle_meta('.portfolo', mock_agent) is True
    assert handle_meta('.nonexistent', mock_agent) is True

def test_handle_meta_cost():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    mock_agent.cost.return_value = {}
    res = handle_meta('.cost', mock_agent)
    assert res is True
    mock_agent.cost.assert_called_once()

def test_handle_meta_memory():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    mock_agent.memory_stats.return_value = {}
    res = handle_meta('.memory', mock_agent)
    assert res is True
    mock_agent.memory_stats.assert_called_once()

def test_handle_meta_clear():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    res = handle_meta('.clear', mock_agent)
    assert res is True
    mock_agent.clear_history.assert_called_once()

def test_handle_meta_unknown():
    mock_agent = MagicMock()
    mock_agent.config.model = 'test'
    res = handle_meta('not a command', mock_agent)
    assert res is False

def test_resolve_model_by_number():
    from agenthood.tui import MODELS
    res = _resolve_model('1')
    assert res == MODELS[0][0]

def test_resolve_model_by_name_prefix():
    res = _resolve_model('gpt-4o')
    assert res == 'openai/gpt-4o'



# ── Readline prompt width ─────────────────────────────────────────────────────

def test_prompts_mark_ansi_as_invisible():
    """Regression: up-arrow recall scrambled the line and ate the prompt.

    Readline counts every byte of the prompt when tracking the cursor, so raw
    ANSI made it think the prompt was ~15 columns wider than it renders.
    Recalling a history entry then redrew over the prompt:

        before:  't now and should I buyg at righ'
        after :  '> what is TSLA trading at right now and shoul' / 'd I buy'

    Every escape sequence must sit inside \001...\002 so readline skips it.
    """
    import re
    from agenthood.tui.theme import PROMPT_MAIN, PROMPT_CONT

    for name, prompt in (("PROMPT_MAIN", PROMPT_MAIN), ("PROMPT_CONT", PROMPT_CONT)):
        assert "\x1b" in prompt, f"{name} should still be coloured"
        # strip every \001...\002 span; no escape may survive outside them
        outside = re.sub(r"\001.*?\002", "", prompt, flags=re.S)
        assert "\x1b" not in outside, (
            f"{name} has ANSI outside \\001/\\002 -- readline will miscount its width"
        )
        # a newline inside the prompt breaks the same arithmetic
        assert "\n" not in prompt, f"{name} must not embed a newline"


def test_prompt_visible_width_is_small():
    """The measurable prompt is just the chevron and padding, not the colour codes."""
    import re
    from agenthood.tui.theme import PROMPT_MAIN

    visible = re.sub(r"\001.*?\002", "", PROMPT_MAIN, flags=re.S)
    assert len(visible) <= 4, f"visible prompt should be tiny, got {visible!r}"
    assert "❯" in visible


# ── Quote price extraction ────────────────────────────────────────────────────

class TestExtractQuotePrice:
    """Regression: the watcher read the wrong level of the quote payload.

    Robinhood returns {"data":{"results":[{"quote":{...},"close":{...}}]}}.
    Looking for last_trade_price at the top level never matched, so every poll
    fell through to a regex scrape and then a full LLM round trip -- and could
    latch onto a bid, an ask, or an unrelated number.
    """

    PAYLOAD = {
        "data": {"results": [{
            "quote": {
                "symbol": "NVDA",
                "last_trade_price": "219.210000",
                "bid_price": "219.340000",
                "ask_price": "219.560000",
                "adjusted_previous_close": "219.220000",
            },
            "close": {"price": "219.22"},
        }]},
        "guide": "...",
    }

    def test_reads_last_trade_price(self):
        from agenthood.tui.watch import extract_quote_price
        assert extract_quote_price(self.PAYLOAD) == 219.21

    def test_accepts_json_string(self):
        import json
        from agenthood.tui.watch import extract_quote_price
        assert extract_quote_price(json.dumps(self.PAYLOAD)) == 219.21

    def test_prefers_trade_price_over_bid_and_ask(self):
        from agenthood.tui.watch import extract_quote_price
        price = extract_quote_price(self.PAYLOAD)
        assert price != 219.34 and price != 219.56

    def test_falls_back_to_close_when_no_quote_prices(self):
        from agenthood.tui.watch import extract_quote_price
        payload = {"data": {"results": [{"quote": {"symbol": "X"},
                                         "close": {"price": "12.50"}}]}}
        assert extract_quote_price(payload) == 12.50

    def test_returns_none_on_junk(self):
        from agenthood.tui.watch import extract_quote_price
        assert extract_quote_price("not json") is None
        assert extract_quote_price({"data": {"results": []}}) is None
        assert extract_quote_price(None) is None


# ── Tool argument coercion ────────────────────────────────────────────────────

class TestCoerceToolArgs:
    """Regression: MCP rejected calls when the model stringified an array arg.

        MCP error: invalid params: validating /properties/symbols:
        type: ["NVDA"] has type "string", want one of "null, array"
    """

    SPEC = {"symbols": "['null', 'array']: One or more stock symbols. (required)"}

    def test_json_encoded_array_is_decoded(self):
        from agenthood.core.toolargs import coerce_args
        assert coerce_args(self.SPEC, {"symbols": '["NVDA"]'}) == {"symbols": ["NVDA"]}

    def test_bare_scalar_is_wrapped(self):
        from agenthood.core.toolargs import coerce_args
        assert coerce_args(self.SPEC, {"symbols": "NVDA"}) == {"symbols": ["NVDA"]}

    def test_correct_array_untouched(self):
        from agenthood.core.toolargs import coerce_args
        assert coerce_args(self.SPEC, {"symbols": ["NVDA"]}) == {"symbols": ["NVDA"]}

    def test_string_typed_param_left_alone(self):
        from agenthood.core.toolargs import coerce_args
        spec = {"account_number": "['string']: the account"}
        assert coerce_args(spec, {"account_number": "123"}) == {"account_number": "123"}

    def test_numeric_and_boolean_coercion(self):
        from agenthood.core.toolargs import coerce_args
        spec = {"qty": "['number']: how many", "live": "['boolean']: real order"}
        out = coerce_args(spec, {"qty": "5", "live": "false"})
        assert out == {"qty": 5.0, "live": False}

    def test_unknown_param_passes_through(self):
        from agenthood.core.toolargs import coerce_args
        assert coerce_args(self.SPEC, {"other": "x"}) == {"other": "x"}


# ── .update ───────────────────────────────────────────────────────────────────

class TestUpdateCommand:

    def test_handle_meta_returns_update_sentinel(self):
        from unittest.mock import MagicMock
        from agenthood.tui import handle_meta
        agent = MagicMock()
        agent.config.model = "test"
        assert handle_meta(".update", agent) == "__update__"
        assert handle_meta(".upgrade", agent) == "__update__"

    def test_update_is_tab_completable(self):
        from agenthood.tui.input import _DOT_COMMANDS
        assert ".update" in _DOT_COMMANDS

    def test_common_commands_are_tab_completable(self):
        """Regression: Tab did nothing for the six most-used commands."""
        from agenthood.tui.input import _DOT_COMMANDS
        for c in (".analyze", ".portfolio", ".positions", ".quote", ".tools"):
            assert c in _DOT_COMMANDS, f"{c} missing from tab completion"

    def test_editable_install_is_not_upgraded(self):
        """Upgrading a source checkout would overwrite the working copy."""
        import asyncio
        from unittest.mock import patch
        from agenthood.tui import update as U
        with patch.object(U, "_is_editable", return_value=True), \
             patch.object(U, "_run_upgrade") as run, \
             patch.object(U, "_latest_version") as latest:
            asyncio.run(U.run_update())
        run.assert_not_called()
        latest.assert_not_called()

    def test_no_upgrade_when_current(self):
        import asyncio
        from unittest.mock import patch
        from agenthood.tui import update as U
        with patch.object(U, "_is_editable", return_value=False), \
             patch.object(U, "_installed_version", return_value="0.3.1"), \
             patch.object(U, "_latest_version", return_value="0.3.1"), \
             patch.object(U, "_run_upgrade") as run:
            asyncio.run(U.run_update())
        run.assert_not_called()

    def test_upgrade_runs_when_newer_exists(self):
        import asyncio
        from unittest.mock import patch
        from agenthood.tui import update as U
        with patch.object(U, "_is_editable", return_value=False), \
             patch.object(U, "_installed_version", return_value="0.3.1"), \
             patch.object(U, "_latest_version", return_value="0.4.0"), \
             patch.object(U, "_run_upgrade", return_value=(True, "ok")) as run:
            asyncio.run(U.run_update())
        run.assert_called_once()

    def test_version_ordering(self):
        from agenthood.tui.update import _parse
        assert _parse("0.3.1") < _parse("0.4.0") < _parse("0.10.0")
        assert _parse("1.0.0") > _parse("0.99.9")

    def test_no_break_system_packages_inside_venv(self):
        """A virtualenv is never externally managed.

        sysconfig reports the *base* interpreter's stdlib, so the PEP 668 marker
        shows up even inside a venv -- checking it naively added the override
        where it is neither needed nor wanted.
        """
        from unittest.mock import patch
        from agenthood.tui import update as U
        with patch.object(U, "_in_virtualenv", return_value=True):
            assert U._externally_managed() is False
            assert "--break-system-packages" not in U._pip_command()

    def test_break_system_packages_when_externally_managed(self):
        from unittest.mock import patch
        from agenthood.tui import update as U
        with patch.object(U, "_externally_managed", return_value=True):
            assert "--break-system-packages" in U._pip_command()

    def test_pip_command_targets_this_interpreter(self):
        import sys
        from agenthood.tui.update import _pip_command
        cmd = _pip_command()
        assert cmd[:5] == [sys.executable, "-m", "pip", "install", "--upgrade"]
        assert "agenthood" in cmd


def test_apply_model_switch_resets_cooldowns():
    from agenthood.core import llm
    from agenthood.tui.models import apply_model_switch

    llm._set_cooldown("meta-llama/llama-3.3-70b-instruct:free", 300)
    assert llm._cooling_down("meta-llama/llama-3.3-70b-instruct:free") > 0

    mock_agent = MagicMock()
    apply_model_switch(mock_agent, "openai/gpt-4o")
    assert llm._cooling_down("meta-llama/llama-3.3-70b-instruct:free") == 0


def test_session_save_and_load_preserves_history(tmp_path, monkeypatch):
    from agenthood.tui import sessions as S
    monkeypatch.setattr(S, "_SESS_DIR", tmp_path)

    mock_agent = MagicMock()
    mock_agent._history = [("user", "NVDA"), ("assistant", "NVDA is $219")]
    mock_agent._recent_tickers = ["NVDA"]
    mock_agent.config.model = "gpt-4o"

    sess_file = S.session_path("alpha")
    S.save_session_state(sess_file, "alpha", 1, mock_agent)

    fresh_agent = MagicMock()
    fresh_agent._history = []
    fresh_agent._recent_tickers = []
    restored_turns = S.load_session_state(sess_file, fresh_agent)

    assert restored_turns == 1
    assert fresh_agent._history == [("user", "NVDA"), ("assistant", "NVDA is $219")]
    assert fresh_agent._recent_tickers == ["NVDA"]


@pytest.mark.asyncio
async def test_run_quote_resolves_name_and_records_turn():
    from unittest.mock import AsyncMock
    from agenthood.tui.actions import run_quote

    mock_agent = MagicMock()
    mock_agent._call_tool_direct = AsyncMock(return_value={
        "data": {"results": [{"quote": {"symbol": "NVDA", "last_trade_price": "219.50", "previous_close": "215.00"}}]}
    })
    mock_agent.record_turn = MagicMock()

    await run_quote(mock_agent, "NVIDIA")
    mock_agent._call_tool_direct.assert_awaited_once_with(
        "robinhood_get_equity_quotes", {"symbols": ["NVDA"]}
    )
    mock_agent.record_turn.assert_called_once()
    args, kwargs = mock_agent.record_turn.call_args
    assert "NVDA" in args[1]
    assert kwargs.get("tickers") == ["NVDA"]


# ── Slice 3: UI/UX & Markdown Rendering TDD tests ─────────────────────────────

class TestUIUXAndFormatting:

    def test_render_with_code_blocks_detects_tables_lists_bold_and_headers(self):
        from rich.markdown import Markdown
        from agenthood.tui.render import _render_with_code_blocks

        # Table with wide separator (like core/formatting.py produces)
        t1 = "| Symbol | Name |\n|--------|------|\n| **AAPL** | Apple |"
        assert isinstance(_render_with_code_blocks(t1), Markdown)

        # Bullet list with bold text
        t2 = "- **Price**: $219.50\n- **Verdict**: BUY"
        assert isinstance(_render_with_code_blocks(t2), Markdown)

        # H3 heading
        t3 = "### Key Risks\n Valuation is elevated."
        assert isinstance(_render_with_code_blocks(t3), Markdown)

    def test_format_mcp_response_multi_ticker_quote_renders_all_symbols(self):
        from agenthood.tui.format import _format_mcp_response

        payload = {
            "data": {
                "results": [
                    {
                        "quote": {
                            "symbol": "AAPL",
                            "last_trade_price": "225.00",
                            "adjusted_previous_close": "220.00",
                            "bid_price": "224.90",
                            "ask_price": "225.10",
                        }
                    },
                    {
                        "quote": {
                            "symbol": "NVDA",
                            "last_trade_price": "140.00",
                            "adjusted_previous_close": "135.00",
                            "bid_price": "139.90",
                            "ask_price": "140.10",
                        }
                    },
                ]
            }
        }
        out = _format_mcp_response(payload, "AAPL, NVDA Quote")
        assert "AAPL" in out
        assert "NVDA" in out
        assert "$225.00" in out
        assert "$140.00" in out
        assert "Verdict" in out or "Signal" in out

    def test_format_mcp_response_single_quote_includes_day_change_and_verdict(self):
        from agenthood.tui.format import _format_mcp_response

        payload = {
            "data": {
                "results": [
                    {
                        "quote": {
                            "symbol": "NVDA",
                            "last_trade_price": "220.00",
                            "adjusted_previous_close": "200.00",
                            "bid_price": "219.90",
                            "ask_price": "220.10",
                        }
                    }
                ]
            }
        }
        out = _format_mcp_response(payload, "NVDA Quote")
        assert "+10.00%" in out
        assert "BUY" in out

    def test_no_emojis_or_em_dashes_in_ui_modules(self):
        """AGENTS.md rule: No emojis. No em-dashes."""
        from pathlib import Path
        import agenthood

        pkg = Path(agenthood.__file__).parent
        ui_files = [
            pkg / "cli.py",
            pkg / "tui" / "render.py",
            pkg / "tui" / "watch.py",
            pkg / "tui" / "actions.py",
            pkg / "tui" / "commands.py",
        ]
        for f in ui_files:
            src = f.read_text()
            assert "⚠️" not in src, f"Emoji found in {f.name}"
            assert "⚡" not in src, f"Emoji found in {f.name}"
            # Check string literals don't output em-dashes
            for line in src.splitlines():
                stripped = line.lstrip()
                assert "—" not in stripped, f"Em-dash found in {f.name}: {line}"


# ── Slice 1: Tab Completion, Dynamic Context, & Numeric Quick-Select ──────────

class TestTabCompletionAndQuickSelect:

    def test_completer_completes_ticker_and_company_names_after_dot_quote_or_analyze(self):
        from agenthood.tui.input import _make_completer, set_completion_context

        set_completion_context(["PLTR", "NVDA"])
        comp = _make_completer()

        with pytest.MonkeyPatch.context() as mp:
            import readline
            mp.setattr(readline, "get_line_buffer", lambda: ".q nv")
            matches = []
            i = 0
            while (m := comp("nv", i)) is not None:
                matches.append(m)
                i += 1
            assert "NVDA" in matches or "nvidia" in [x.lower() for x in matches]

            mp.setattr(readline, "get_line_buffer", lambda: ".a pl")
            matches_a = []
            i = 0
            while (m := comp("pl", i)) is not None:
                matches_a.append(m)
                i += 1
            assert "PLTR" in matches_a

    def test_completer_completes_watch_direction_after_ticker(self):
        from agenthood.tui.input import _make_completer

        comp = _make_completer()
        with pytest.MonkeyPatch.context() as mp:
            import readline
            mp.setattr(readline, "get_line_buffer", lambda: ".watch NVDA b")
            assert comp("b", 0) == "below"

            mp.setattr(readline, "get_line_buffer", lambda: "watch AAPL a")
            assert comp("a", 0) == "above"

    def test_completer_completes_lowercase_ticker_suffixes(self):
        from agenthood.tui.input import _make_completer

        comp = _make_completer()
        with pytest.MonkeyPatch.context() as mp:
            import readline
            mp.setattr(readline, "get_line_buffer", lambda: "nvda op")
            assert comp("op", 0) == "options"

    def test_handle_meta_numeric_quick_select_quotes_recent_ticker(self):
        """Typing '1' or '.1' selects the #1 ticker from the most recent table."""
        mock_agent = MagicMock()
        mock_agent.config.model = "test"
        mock_agent._recent_tickers = ["BMBL", "GS", "BBAI"]

        # '1' -> 1st ticker (BMBL), '2' -> 2nd ticker (GS)
        assert handle_meta("1", mock_agent) == ("__quote__", "BMBL")
        assert handle_meta("2", mock_agent) == ("__quote__", "GS")
        assert handle_meta(".1", mock_agent) == ("__quote__", "BMBL")
        assert handle_meta(".a 2", mock_agent) == ["GS"]

    def test_format_mcp_response_includes_row_indices_and_clickable_links(self):
        """Multi-ticker quotes and positions tables include 1-based # indices and Robinhood links."""
        from agenthood.tui.format import _format_mcp_response

        raw_quotes = {
            "results": [
                {"quote": {"symbol": "NVDA", "last_trade_price": "220.00", "adjusted_previous_close": "210.00", "bid_price": "219.90", "ask_price": "220.10"}},
                {"quote": {"symbol": "AAPL", "last_trade_price": "195.00", "adjusted_previous_close": "194.00", "bid_price": "194.90", "ask_price": "195.10"}},
            ]
        }
        table = _format_mcp_response(raw_quotes, "Top Movers Quote")
        assert "| # |" in table
        assert "| 1 |" in table
        assert "| 2 |" in table
        assert "https://robinhood.com/stocks/NVDA" in table
        assert "https://robinhood.com/stocks/AAPL" in table

        raw_pos = {
            "results": [
                {"symbol": "TSLA", "quantity": "5", "average_buy_price": "200.00", "current_price": "250.00"},
            ]
        }
        pos_table = _format_mcp_response(raw_pos, "Positions")
        assert "| # |" in pos_table
        assert "| 1 |" in pos_table
        assert "https://robinhood.com/stocks/TSLA" in pos_table

    def test_handle_meta_trending_movers_and_risk_dot_commands(self):
        """Zero-LLM dot-commands .trending/.t, .movers, and .risk return dedicated action sentinels."""
        mock_agent = MagicMock()
        mock_agent.config.model = "test"
        mock_agent.last_ticker = "NVDA"
        mock_agent._recent_tickers = ["NVDA", "AAPL"]

        assert handle_meta(".trending", mock_agent) == "__trending__"
        assert handle_meta(".t", mock_agent) == "__trending__"
        assert handle_meta(".movers", mock_agent) == "__movers__"
        assert handle_meta(".risk NVDA 180", mock_agent) == ("__risk__", "NVDA", 180.0)
        assert handle_meta(".risk", mock_agent) == ("__risk__", "NVDA", None)
        assert handle_meta(".risk 2 185.5", mock_agent) == ("__risk__", "AAPL", 185.5)

    @pytest.mark.asyncio
    async def test_run_discovery_and_run_risk_actions(self):
        """run_discovery and run_risk render structured markdown tables without LLM overhead."""
        from unittest.mock import AsyncMock
        from agenthood.tui import actions

        mock_agent = MagicMock()
        mock_agent.config.max_order_value = 5000.0
        mock_agent._get_account_number = AsyncMock(return_value="ACC-1")
        mock_agent._fetch_discovery_movers = AsyncMock(
            return_value=("## Trending Stocks Quote\n| # | Symbol |\n|---|---|", ["NVDA", "AAPL"])
        )

        bars = [{"close_price": str(190.0 + i), "high_price": str(192.0 + i), "low_price": str(189.0 + i)} for i in range(20)]

        async def fake_direct(name, params):
            if name == "robinhood_get_portfolio":
                return {"data": {"total_value": "25000.00"}}
            if name == "robinhood_get_equity_quotes":
                return {"results": [{"quote": {"symbol": "NVDA", "last_trade_price": "200.00"}}]}
            if name == "robinhood_get_equity_historicals":
                return {"results": [{"historicals": bars}]}
            raise ValueError(name)

        mock_agent._call_tool_direct = AsyncMock(side_effect=fake_direct)

        rendered = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(actions, "render_response", lambda text, elapsed: rendered.append(text))
            await actions.run_discovery(mock_agent, "__trending__")
            await actions.run_risk(mock_agent, "NVDA", 190.0)

        assert len(rendered) == 2
        assert "Trending Stocks Quote" in rendered[0]
        assert "NVDA Position Sizing & Risk Plan" in rendered[1]
        assert "25 shares" in rendered[1]
        assert "14-Day Wilder RSI" in rendered[1]





