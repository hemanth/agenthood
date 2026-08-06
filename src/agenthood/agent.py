"""Backwards-compatibility shim for the pre-split ``agenthood.agent`` module.

The implementation now lives in :mod:`agenthood.core`:

  agenthood.core.agent       -- the AgentHood class
  agenthood.core.builder     -- build_agent(), agentu wiring
  agenthood.core.llm         -- direct OpenAI-compatible HTTP client
  agenthood.core.prompts     -- SYSTEM_PROMPT, analyze prompt template
  agenthood.core.response    -- _extract_response
  agenthood.core.formatting  -- _format_tool_result

Importing from here still works.  Note that ``mock.patch`` targets do NOT: patch
the module where a name is *used* (e.g. ``agenthood.core.builder.Agent``), not
this shim.
"""
from __future__ import annotations

from .core.agent import AgentHood, TokenManager
from .core.formatting import _format_tool_result
from .core.prompts import SYSTEM_PROMPT, analyze_prompt
from .core.response import _extract_response

__all__ = [
    "AgentHood",
    "SYSTEM_PROMPT",
    "analyze_prompt",
    "_extract_response",
    "_format_tool_result",
]
