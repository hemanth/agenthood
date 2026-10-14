"""agenthood TUI -- terminal interface built with Rich.

Design principles (distilled from agy, codex, claude CLIs):
  agy    -- compact status bar, tool trace shown inline, always feels fast
  codex  -- numbered model picker, minimal chrome, task-first layout
  claude -- live thinking panel, beautiful panels, context-aware footer

Features:
  - Persistent readline history  (~/.agenthood/history)
  - Arrow keys + Ctrl+R reverse search  (readline / libedit)
  - Session management  (--session NAME to continue, --new-session for fresh)
  - Numbered model picker  (.model command, hot-swap, no restart needed)
  - Live thinking panel  (transient spinner + step trace)
  - Syntax-highlighted response panels  (fenced code blocks detected)
  - Elapsed time shown per response

No emojis. No em-dashes. ASCII via Rich only.

Layout:
  theme.py     -- console + colour tokens
  input.py     -- readline setup, completion, line reading
  models.py    -- model catalogue, picker, hot-swap
  sessions.py  -- saved session files
  thinking.py  -- transient spinner panel
  render.py    -- banner, response panels, help, status bar
  format.py    -- MCP JSON -> markdown tables
  tickers.py   -- company name -> ticker resolution
  commands.py  -- dot-command dispatch (handle_meta)
  actions.py   -- execution bodies for .portfolio/.positions/.quote/.analyze
  watch.py     -- in-REPL price watchers
  repl.py      -- the loop
"""
from .commands import handle_meta
from .format import _format_mcp_response
from .input import save_readline, set_completion_context, setup_readline
from .models import MODELS, _resolve_model, _short_model, apply_model_switch, show_model_picker
from .render import _strip_emoji, print_banner, print_help, render_response
from .repl import tui_repl
from .sessions import list_sessions, print_sessions, session_path
from .theme import console
from .thinking import ThinkingDisplay

__all__ = [
    "MODELS",
    "ThinkingDisplay",
    "apply_model_switch",
    "console",
    "handle_meta",
    "list_sessions",
    "print_banner",
    "print_help",
    "print_sessions",
    "render_response",
    "save_readline",
    "session_path",
    "set_completion_context",
    "setup_readline",
    "show_model_picker",
    "tui_repl",
    "_format_mcp_response",
    "_resolve_model",
    "_short_model",
    "_strip_emoji",
]
