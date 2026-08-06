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
