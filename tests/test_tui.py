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
    res = handle_meta('.analyze', mock_agent)
    assert res is True

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
