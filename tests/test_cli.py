import sys
import pytest
from unittest.mock import patch
from agenthood.cli import parse_args

def test_parse_args_default():
    with patch.object(sys, 'argv', ['agenthood']):
        args = parse_args()
        assert args.command is None

def test_parse_args_analyze():
    with patch.object(sys, 'argv', ['agenthood', 'analyze', 'AAPL', 'NVDA']):
        args = parse_args()
        assert args.command == 'analyze'
        assert args.tickers == ['AAPL', 'NVDA']

def test_parse_args_watch():
    with patch.object(sys, 'argv', ['agenthood', 'watch', 'AAPL', 'below', '180']):
        args = parse_args()
        assert args.command == 'watch'
        assert args.ticker == 'AAPL'
        assert args.direction == 'below'
        assert args.price == 180.0

def test_parse_args_daemon():
    with patch.object(sys, 'argv', ['agenthood', 'daemon', 'start']):
        args = parse_args()
        assert args.command == 'daemon'
        assert args.action == 'start'

def test_parse_args_live_flag():
    with patch.object(sys, 'argv', ['agenthood', '--live']):
        args = parse_args()
        assert args.live is True


@pytest.mark.asyncio
async def test_main_analyze_splits_commas_and_resolves_company_names():
    """`agenthood analyze AAPL, nvidia` should resolve 'nvidia' -> 'NVDA' and analyze both."""
    from unittest.mock import MagicMock, AsyncMock
    from agenthood.cli import main

    mock_agent = MagicMock()
    mock_agent.analyze = AsyncMock(side_effect=["## AAPL Analysis", "## NVDA Analysis"])
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_agent)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    cfg = MagicMock(mcp_token="tok", model=None, dry_run=True, codemode=False, stream=False)

    with patch.object(sys, "argv", ["agenthood", "analyze", "AAPL,", "nvidia"]), \
         patch("agenthood.cli.load_config", return_value=cfg), \
         patch("agenthood.cli.AgentHood", return_value=mock_ctx), \
         patch("agenthood.cli._print_rich") as mock_print:
        await main()

    assert mock_agent.analyze.await_count == 2
    called_tickers = [c.args[0] for c in mock_agent.analyze.await_args_list]
    assert called_tickers == ["AAPL", "NVDA"]
    assert mock_print.call_count == 2


@pytest.mark.asyncio
async def test_main_oneshot_dot_portfolio_calls_run_account_view():
    """`agenthood .p` should dispatch through handle_meta -> run_account_view without LLM chat."""
    from unittest.mock import MagicMock, AsyncMock
    from agenthood.cli import main

    mock_agent = MagicMock()
    mock_agent.chat = AsyncMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_agent)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    cfg = MagicMock(mcp_token="tok", model=None, dry_run=True, codemode=False, stream=False)

    with patch.object(sys, "argv", ["agenthood", ".p"]), \
         patch("agenthood.cli.load_config", return_value=cfg), \
         patch("agenthood.cli.AgentHood", return_value=mock_ctx), \
         patch("agenthood.tui.actions.run_account_view", new_callable=AsyncMock) as mock_acct:
        await main()

    mock_acct.assert_awaited_once_with(mock_agent, "portfolio")
    mock_agent.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_oneshot_dot_quote_calls_run_quote():
    """`agenthood .q NVDA` should dispatch directly to run_quote without LLM chat."""
    from unittest.mock import MagicMock, AsyncMock
    from agenthood.cli import main

    mock_agent = MagicMock()
    mock_agent.chat = AsyncMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_agent)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    cfg = MagicMock(mcp_token="tok", model=None, dry_run=True, codemode=False, stream=False)

    with patch.object(sys, "argv", ["agenthood", ".q", "NVDA"]), \
         patch("agenthood.cli.load_config", return_value=cfg), \
         patch("agenthood.cli.AgentHood", return_value=mock_ctx), \
         patch("agenthood.tui.actions.run_quote", new_callable=AsyncMock) as mock_quote:
        await main()

    mock_quote.assert_awaited_once_with(mock_agent, "NVDA")
    mock_agent.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_main_oneshot_natural_watch_registers_daemon_watch(capsys):
    """`agenthood 'watch AAPL below 180'` should register watch via add_watch and print clean message."""
    from unittest.mock import MagicMock, AsyncMock
    from agenthood.cli import main

    mock_agent = MagicMock()
    mock_agent.chat = AsyncMock(return_value={
        "__watch__": True,
        "spec": {
            "ticker": "AAPL",
            "direction": "below",
            "threshold": 180.0,
            "action": "alert",
            "shares": 0,
            "interval": 30,
        },
    })
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_agent)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    cfg = MagicMock(mcp_token="tok", model=None, dry_run=True, codemode=False, stream=False)

    with patch.object(sys, "argv", ["agenthood", "watch AAPL below 180"]), \
         patch("agenthood.cli.load_config", return_value=cfg), \
         patch("agenthood.cli.AgentHood", return_value=mock_ctx), \
         patch("agenthood.daemon.add_watch", return_value="w123") as mock_add, \
         patch("agenthood.daemon.start_daemon") as mock_start:
        await main()

    mock_add.assert_called_once()
    mock_start.assert_called_once()
    out = capsys.readouterr().out
    assert "AAPL" in out
    assert "180.00" in out
    assert "__watch__" not in out


@pytest.mark.asyncio
async def test_plain_repl_supports_dot_commands():
    """`plain_repl` should support dot-commands via handle_meta instead of sending `.p` to LLM."""
    from unittest.mock import MagicMock, AsyncMock
    from agenthood.cli import plain_repl

    mock_agent = MagicMock()
    mock_agent.config = MagicMock(model="test-model", dry_run=True, max_order_value=500.0)
    mock_agent.chat = AsyncMock(return_value="should not be called")

    inputs = iter([".p", "exit"])
    with patch("builtins.input", side_effect=lambda *_: next(inputs)), \
         patch("agenthood.tui.actions.run_account_view", new_callable=AsyncMock) as mock_acct:
        await plain_repl(mock_agent)

    mock_acct.assert_awaited_once_with(mock_agent, "portfolio")
    mock_agent.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_completions_subcommand_outputs_zsh_and_bash_scripts(capsys):
    """`agenthood completions zsh` and `agenthood completions bash` output valid completion scripts."""
    from agenthood.cli import main

    with patch.object(sys, "argv", ["agenthood", "completions", "zsh"]):
        await main()
    zsh_out = capsys.readouterr().out
    assert "#compdef agenthood" in zsh_out or "compdef" in zsh_out
    assert "analyze" in zsh_out and "portfolio" in zsh_out

    with patch.object(sys, "argv", ["agenthood", "completions", "bash"]):
        await main()
    bash_out = capsys.readouterr().out
    assert "complete -F" in bash_out
    assert "agenthood" in bash_out


