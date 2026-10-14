"""The main TUI REPL loop: read, dispatch, render, repeat."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from rich.markup import escape

from .actions import run_account_view, run_analyze, run_discovery, run_quote, run_risk
from .commands import handle_meta
from .input import read_input, save_readline, setup_readline
from .render import (
    _print_status, _strip_emoji, print_banner, print_help, render_response,
)
from .sessions import load_session_state, save_session_state, session_path
from .theme import C_DIM, C_DIMMER, C_ERR, C_LABEL, C_WARN, console
from .thinking import ThinkingDisplay
from .update import run_update
from .watch import WatchController

DOUBLE_INTERRUPT_SECS = 2.0
HISTORY_HELP_WORDS = ("?", "help", ".help", "?help")
EXIT_WORDS = ("exit", "quit", "q")


async def _read_input_async() -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_input)


def _save_session(sess_file, session_name: str, turn: int, agent) -> None:
    save_session_state(sess_file, session_name, turn, agent)
    console.print(f"  [{C_DIMMER}]session saved: {session_name}[/{C_DIMMER}]")


def _report_daemon_triggers() -> None:
    from ..daemon import get_pending_triggers
    for trigger in get_pending_triggers():
        spec = trigger['spec']
        price = trigger['price']
        console.print()
        console.print(
            f"  [{C_WARN}]  DAEMON WATCHER TRIGGERED  "
            f"{spec['ticker']} hit ${price:,.2f} "
            f"(threshold {spec['direction']} ${spec['threshold']:,.2f})[/{C_WARN}]"
        )


async def tui_repl(agent, session_name: Optional[str] = None) -> None:
    """Main TUI REPL.  Call with optional session_name to persist context."""
    cfg = agent.config

    # Load session memory if continuing
    turn = 0
    sess_file = session_path(session_name) if session_name else None
    if sess_file and sess_file.exists():
        turn = load_session_state(sess_file, agent)
        console.print(
            f"  [{C_DIM}]continuing session {session_name!r} "
            f"({turn} prior turn{'s' if turn != 1 else ''})[/{C_DIM}]"
        )

    setup_readline()
    print_banner(
        model=cfg.model or "auto",
        dry_run=cfg.dry_run,
        max_order=cfg.max_order_value,
        session_name=session_name,
    )

    verbose        = False
    total_cost_usd = 0.0
    _last_interrupt = 0.0   # timestamp of most recent Ctrl+C

    watches = WatchController(agent)

    try:
        while True:
            _report_daemon_triggers()

            # ── Input ─────────────────────────────────────────────────────────
            try:
                input_task = asyncio.create_task(_read_input_async())
                drain_task = asyncio.create_task(watches.drain_while_waiting(input_task))
                await asyncio.wait([input_task, drain_task], return_when=asyncio.FIRST_COMPLETED)
                drain_task.cancel()
                try:
                    user_input = await input_task
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - _last_interrupt < DOUBLE_INTERRUPT_SECS:
                    # Second Ctrl+C within 2 s -> exit
                    console.print(f"\n  [{C_DIM}]session ended  {turn} turns[/{C_DIM}]")
                    break
                _last_interrupt = now
                console.print(f"\n  [{C_DIM}]Ctrl+C again to exit  (or type exit)[/{C_DIM}]")
                continue
            except EOFError:
                console.print(f"\n  [{C_DIM}]session ended  {turn} turns[/{C_DIM}]")
                break

            if not user_input:
                continue

            await watches.drain_triggered()

            # ── Built-in words ────────────────────────────────────────────────
            if user_input in HISTORY_HELP_WORDS:
                print_help()
                continue
            if user_input.lower() in EXIT_WORDS:
                watches.cancel_all()
                console.print(f"  [{C_DIM}]session ended  {turn} turns[/{C_DIM}]")
                break
            if user_input.lower() == ".verbose":
                verbose = not verbose
                console.print(f"  [{C_DIM}]verbose {'on' if verbose else 'off'}[/{C_DIM}]")
                continue

            # ── Dot-commands ──────────────────────────────────────────────────
            meta_result = handle_meta(user_input, agent)
            if meta_result is True:
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # .watches
            if meta_result == "__watches__":
                watches.show_watches()
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # .watch TICKER below|above PRICE
            if isinstance(meta_result, dict) and "__watch__" in meta_result:
                watches.start(meta_result["__watch__"])
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # .update -- self-upgrade from PyPI
            if meta_result == "__update__":
                await run_update()
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # .trending / .movers -- direct watchlist + quotes discovery
            if meta_result in ("__trending__", "__movers__"):
                turn += 1
                await run_discovery(agent, meta_result)
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # .portfolio / .positions -- direct MCP calls
            if isinstance(meta_result, str) and meta_result.startswith("__"):
                turn += 1
                await run_account_view(agent, meta_result)
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # .risk TICKER [STOP] -- direct position sizing & risk calculator
            if isinstance(meta_result, tuple) and meta_result[0] == "__risk__":
                turn += 1
                await run_risk(agent, meta_result[1], meta_result[2])
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # .quote TICKER -- direct MCP call
            if isinstance(meta_result, tuple) and meta_result[0] == "__quote__":
                turn += 1
                await run_quote(agent, meta_result[1])
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # .analyze TICKER [, TICKER2, ...]
            if isinstance(meta_result, list):
                turn += 1
                await run_analyze(agent, meta_result)
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            # ── Thinking + inference ──────────────────────────────────────────
            turn += 1
            t0       = time.monotonic()
            response = None
            error    = None

            if cfg.stream:
                console.print()
                console.print(f"  [{C_LABEL}]agenthood >[/{C_LABEL}]  ", end="")
                try:
                    await agent.stream(user_input)
                except Exception as exc:
                    console.print(f"\n  [{C_ERR}]! {escape(str(exc))}[/{C_ERR}]")
                _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)
                continue

            async with ThinkingDisplay(console) as td:
                td.step("routing query...")
                try:
                    response = await agent.chat(
                        user_input,
                        _progress=td.step,
                    )
                    td.step("done")
                except KeyboardInterrupt:
                    td.step("cancelled")
                    console.print(f"\n  [{C_DIM}]cancelled[/{C_DIM}]")
                    _last_interrupt = time.monotonic()
                    continue
                except Exception as exc:
                    error = str(exc)
                    td.step(f"error: {error[:80]}")

            elapsed = time.monotonic() - t0

            # Cost tracking
            try:
                c = agent.cost()
                if isinstance(c, dict) and "total_cost" in c:
                    total_cost_usd = float(c["total_cost"])
            except Exception:
                pass

            # ── Output ─────────────────────────────────────────────────────────
            console.print()
            if error:
                console.print(f"  [{C_ERR}]! Error: {escape(error)}[/{C_ERR}]")
            elif response is not None:
                if isinstance(response, dict) and "__watch__" in response:
                    # Watcher sentinel -- start background task
                    watches.start(response["__watch__"])
                    turn -= 1  # don't count as a regular turn
                elif isinstance(response, dict):
                    msg = response.get("error", str(response))
                    console.print(f"  [{C_WARN}]! {escape(_strip_emoji(msg))}[/{C_WARN}]")
                elif not str(response).strip():
                    # Empty response -- model couldn't handle this input
                    render_response(
                        "I'm a trading assistant. "
                        "Ask me about stocks, crypto, options, or your portfolio.",
                        elapsed,
                    )
                else:
                    render_response(_strip_emoji(str(response)), elapsed)

            _print_status(agent, turn, total_cost_usd, active_watches=watches.active_count)

    finally:
        # Always save history and session
        save_readline()
        if sess_file and session_name:
            _save_session(sess_file, session_name, turn, agent)

