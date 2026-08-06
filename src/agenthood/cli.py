#!/usr/bin/env python3
"""agenthood CLI -- interactive Robinhood trading agent."""
from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# ── Silence all loggers immediately, before any agenthood/agentu imports ------
# We must do this BEFORE imports -- agentu creates loggers at module load time.
_VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

_root = logging.getLogger()
_root.setLevel(logging.DEBUG if _VERBOSE else logging.WARNING)
if not _VERBOSE:
    _null = logging.NullHandler()
    for _ns in ("agentu", "agentu.middleware.observe", "agentu._core.agent",
                "agentu.mcp.tool", "agentu.mcp.transport",
                "agenthood", "asyncio", "aiohttp"):
        _lg = logging.getLogger(_ns)
        _lg.handlers = [_null]
        _lg.propagate = False
        _lg.setLevel(logging.CRITICAL)

from agenthood import AgentHood, load_config
from agenthood.setup import run_setup


KNOWN_SUBCOMMANDS = {"setup", "serve", "eval", "watch", "watches", "analyze", "daemon"}


def _add_common_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Path to config file (default: ~/.agenthood/config.json)")
    p.add_argument("--model",  default=None, help="LLM model override (e.g. gpt-4o)")
    p.add_argument("--live",   action="store_true", help="Enable live trading (disables dry_run)")
    p.add_argument("--codemode", action="store_true", help="LLM writes Python to call tools")
    p.add_argument("--stream", action="store_true", help="Stream response tokens as they arrive")
    p.add_argument("--ralph",  metavar="PROMPT.md", help="Run autonomous loop until all checkpoints done")
    p.add_argument("--inbox",  metavar="DIR",       help="Watch directory for signal files")
    p.add_argument("--permissions", choices=["auto", "ask-writes", "plan"],
                   default=None, help="Tool permission mode")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    p.add_argument("--plain",   action="store_true", help="Disable TUI -- use plain text output")
    p.add_argument("--session", metavar="NAME",       help="Continue named session (preserves context)")
    p.add_argument("--new-session", metavar="NAME",   help="Start a fresh named session")
    p.add_argument("--sessions", action="store_true", help="List saved sessions and exit")


def _build_command_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agenthood",
        description="AgentHood -- Robinhood trading agent powered by agentu",
    )
    _add_common_options(p)
    sub = p.add_subparsers(dest="command", required=True)

    # setup subcommand
    sub.add_parser("setup", help="Interactive first-run setup wizard")

    # serve subcommand -- REST + WebSocket API
    srv = sub.add_parser("serve", help="Start REST + WebSocket API server")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)

    # eval subcommand -- batch evaluation
    ev = sub.add_parser("eval", help="Run batch evaluation test suite")
    ev.add_argument("test_file", nargs="?", default=None,
                    help="JSON file with test cases (default: built-in smoke tests)")
    ev.add_argument("--llm-judge", action="store_true", help="Use LLM as judge")

    # watch subcommand
    w_cmd = sub.add_parser("watch", help="Watch TICKER price and act when it crosses PRICE.")
    w_cmd.add_argument("ticker")
    w_cmd.add_argument("direction", choices=["below", "above"])
    w_cmd.add_argument("price", type=float)
    w_cmd.add_argument("--buy", dest="shares_buy", type=int, default=0, help="Shares to buy on trigger")
    w_cmd.add_argument("--sell", dest="shares_sell", type=int, default=0, help="Shares to sell on trigger")
    w_cmd.add_argument("--every", dest="interval_str", default="30s", help="Poll interval: 30s, 5m, 1h")
    w_cmd.add_argument("--notify", dest="notify_specs", action="append", default=[], help="Notification channels")
    w_cmd.add_argument("--no-daemon", dest="daemon", action="store_false", default=True, help="Don't run as background daemon")

    # watches subcommand
    sub.add_parser("watches", help="List active price watches.")

    # analyze subcommand -- structured market analysis
    a_cmd = sub.add_parser("analyze", help="Run structured market analysis for ticker(s).")
    a_cmd.add_argument("tickers", nargs="+", help="One or more ticker symbols (e.g. AAPL NVDA MSFT)")

    # daemon subcommand
    d_cmd = sub.add_parser("daemon", help="Manage the AgentHood watcher daemon.")
    d_cmd.add_argument("action", choices=["start", "stop", "status", "logs"])

    return p


def _build_prompt_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agenthood",
        description="AgentHood -- Robinhood trading agent powered by agentu",
    )
    _add_common_options(p)
    p.add_argument(
        "prompt",
        nargs="*",
        help="One-shot prompt. If omitted, starts interactive REPL.",
    )
    return p


def parse_args() -> argparse.Namespace:
    # argparse can't cleanly mix an optional subparsers action with a trailing
    # positional -- it eagerly validates the first positional token against the
    # subcommand choices and errors out even when no subcommand was intended.
    # Work around it by pre-scanning argv and picking the right parser tree.
    argv = sys.argv[1:]
    first_positional = next((a for a in argv if not a.startswith("-")), None)

    if first_positional in KNOWN_SUBCOMMANDS:
        args = _build_command_parser().parse_args(argv)
        args.prompt = []
    else:
        args = _build_prompt_parser().parse_args(argv)
        args.command = None
    return args


def _print_rich(text) -> None:
    """Print agent output with Rich markup and markdown actually rendered.

    Agent output carries Rich tags ("[bold cyan]NVDA[/bold cyan]") and markdown
    tables. Plain print() dumps the tags as literal text, so route through the
    same renderer the TUI uses. Falls back to print() if Rich is unavailable.
    """
    txt = str(text)
    try:
        from rich.console import Console
        from agenthood.tui.render import _render_with_code_blocks
        # Force terminal colors if stdout supports color or environment requests it
        Console(force_terminal=True).print(
            _render_with_code_blocks(txt), highlight=False
        )
    except Exception:
        print(txt)


# ── Plain fallback REPL (--plain or non-TTY) ─────────────────────────────────

async def plain_repl(agent: AgentHood) -> None:
    """Minimal REPL for non-TTY or --plain mode."""
    cfg = agent.config
    trade = "DRY-RUN" if cfg.dry_run else "LIVE"
    print(f"\nagenthood  model={cfg.model}  trade={trade}  max=${cfg.max_order_value:.0f}")
    print("type 'exit' to quit\n")
    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nsession ended")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("session ended")
            break
        try:
            response = await agent.chat(user_input)
            print(f"agenthood > {response}\n")
        except Exception as exc:
            print(f"! Error: {exc}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def _ensure_auth(cfg) -> bool:
    """Prompt to authenticate with Robinhood if no token is cached.

    Without this, AgentHood.start() fails deep inside MCP tool-loading with a
    raw 401 traceback -- catch it here instead and offer to run the OAuth flow.
    Returns False if the caller should abort (no token, user declined).
    """
    if cfg.mcp_token:
        return True
    print("\n⚠️  No Robinhood auth token found -- trading tools can't load.")
    try:
        choice = input("Authenticate with Robinhood now? (y/n) [y]: ").strip().lower() or "y"
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if choice != "y":
        print("Skipped. Run 'agenthood setup' anytime to authenticate.")
        return False
    from .setup import setup_robinhood_auth
    token = setup_robinhood_auth()
    if not token:
        print("Auth did not complete.")
        return False
    cfg.mcp_token = token
    return True


async def main() -> None:
    args = parse_args()

    # Setup wizard
    if args.command == "setup":
        run_setup(args.config)
        return

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.model:
        cfg.model = args.model
    if args.live:
        cfg.dry_run = False
    if args.codemode:
        cfg.codemode = True
    if args.stream:
        cfg.stream = True
    if getattr(args, "inbox", None):
        cfg.inbox_path = args.inbox
    if getattr(args, "permissions", None):
        cfg.permissions_mode = args.permissions

    # ── serve subcommand ───────────────────────────────────────────────────────
    if args.command == "serve":
        if not _ensure_auth(cfg):
            return
        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", 8000)
        async with AgentHood(cfg) as agent:
            print(f"AgentHood API server starting on http://{host}:{port}")
            print(f"Docs: http://{host}:{port}/docs")
            await agent.serve(host=host, port=port)
        return

    # ── eval subcommand ────────────────────────────────────────────────────────
    if args.command == "eval":
        import json as _json
        test_file = getattr(args, "test_file", None)
        llm_judge = getattr(args, "llm_judge", False)

        if test_file:
            test_cases = _json.loads(Path(test_file).read_text())
        else:
            test_cases = [
                {"ask": "What is AAPL trading at?", "expect_tool": "robinhood_get_equity_quotes"},
                {"ask": "Show my portfolio",         "expect_tool": "robinhood_get_portfolio"},
                {"ask": "Search for Tesla",          "expect_tool": "robinhood_search"},
            ]

        print(f"Running {len(test_cases)} evaluation cases...")
        if not _ensure_auth(cfg):
            return
        async with AgentHood(cfg) as agent:
            results = await agent.evaluate(test_cases, use_llm_judge=llm_judge)
        print(f"Passed   : {results.get('passed','?')}/{results.get('total','?')}")
        print(f"Accuracy : {results.get('accuracy',0):.0%}")
        if results.get("failures"):
            print("\nFailures:")
            for f in results["failures"]:
                print(f"  x {f.get('query','?')[:60]}")
        return

    # ── sessions list (no agent needed) ─────────────────────────────────────
    if getattr(args, "sessions", False):
        from agenthood.tui import print_sessions
        print_sessions()
        return

    # ── analyze subcommand ─────────────────────────────────────────────────────
    if args.command == "analyze":
        tickers = [t.upper() for t in args.tickers]
        print(f"Analyzing {', '.join(tickers)}...\n")
        if not _ensure_auth(cfg):
            return
        async with AgentHood(cfg) as agent:
            for ticker in tickers:
                result = await agent.analyze(ticker)
                _print_rich(result)
                print()
        return

    # ── daemon / watches / watch ────────────────────────────────────────────────
    if args.command == "daemon":
        from agenthood.daemon import start_daemon, stop_daemon, is_daemon_running, DATA_DIR
        if args.action == "start":
            start_daemon()
            print("Daemon started.")
        elif args.action == "stop":
            stop_daemon()
            print("Daemon stopped.")
        elif args.action == "status":
            if is_daemon_running():
                print("Daemon is running.")
            else:
                print("Daemon is stopped.")
        elif args.action == "logs":
            log_file = DATA_DIR / 'daemon.log'
            if log_file.exists():
                print(log_file.read_text())
            else:
                print("No logs found.")
        return

    if args.command == "watches":
        from agenthood.daemon import list_watches
        watches = list_watches()
        if not watches:
            print("No active watches.")
        else:
            for w in watches:
                print(f"[{w.get('id')}] {w['action']} {w.get('shares',0)} {w['ticker']} when {w['direction']} ${w['threshold']:,.2f}")
        return

    if args.command == "watch":
        from agenthood.daemon import add_watch, start_daemon
        iv_str = args.interval_str
        interval = 30
        if iv_str.endswith("h"): interval = int(iv_str[:-1]) * 3600
        elif iv_str.endswith("m"): interval = int(iv_str[:-1]) * 60
        elif iv_str.endswith("s"): interval = int(iv_str[:-1])
        else: interval = int(iv_str)

        action = "alert"
        shares = 0
        if args.shares_buy > 0:
            action = "buy"
            shares = args.shares_buy
        elif args.shares_sell > 0:
            action = "sell"
            shares = args.shares_sell

        spec = {
            "ticker": args.ticker,
            "direction": args.direction,
            "threshold": args.price,
            "action": action,
            "shares": shares,
            "interval": interval,
            "notify": args.notify_specs
        }
        add_watch(spec)
        print(f"Added watch for {args.ticker}.")
        if args.daemon:
            start_daemon()
            print("Daemon started.")
        return

    # ── Interactive / one-shot ─────────────────────────────────────────────────
    if not _ensure_auth(cfg):
        return
    async with AgentHood(cfg) as agent:

        # Ralph autonomous loop
        if args.ralph:
            print(f"Ralph loop starting -- reading {args.ralph}")
            result = await agent.ralph(args.ralph)
            print(f"Ralph complete: {result.get('status')} "
                  f"after {result.get('iterations',0)} iterations")
            return

        # One-shot prompt (non-interactive)
        if args.prompt:
            if cfg.stream:
                await agent.stream(" ".join(args.prompt))
            else:
                response = await agent.chat(" ".join(args.prompt))
                if isinstance(response, dict):
                    print(response.get("error", str(response)))
                else:
                    _print_rich(response)
            return

        # Determine session name
        session_name: str | None = None
        if getattr(args, "session", None):
            session_name = args.session
        elif getattr(args, "new_session", None):
            session_name = args.new_session
            # Delete any existing session file to start fresh
            from agenthood.tui import session_path
            sp = session_path(session_name)
            if sp.exists():
                sp.unlink()
                print(f"Starting fresh session {session_name!r}")

        # Interactive REPL -- TUI unless --plain or non-TTY
        use_tui = not getattr(args, "plain", False) and sys.stdin.isatty()
        if use_tui:
            try:
                from agenthood.tui import tui_repl
                await tui_repl(agent, session_name=session_name)
                return
            except ImportError:
                pass  # rich not available, fall through to plain

        await plain_repl(agent)


def cli() -> None:
    import signal, os
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        os._exit(0)


if __name__ == "__main__":
    cli()
