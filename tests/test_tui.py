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

