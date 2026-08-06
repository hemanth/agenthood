"""Build and wire the agentu Agent from an AgentHoodConfig.

Everything that turns a config into a ready-to-use ``agentu.Agent`` lives here:
MCP connection, tool deferral, skills, hooks, permissions, middleware, and the
optional subsystems (sandbox, cache, guardrails, OTel, notifier, inbox).

``AgentHood.start()`` calls :func:`build_agent` and keeps only the state that is
genuinely its own -- the token-refresh task and the autonomous scan schedule.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentu import Agent

from ..hooks import trading_guard
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Minimal core tools visible to LLM directly to stay under input guardrails.
# The rest are deferred behind search_tools() -- the model discovers them on demand.
# Keep this set TINY: more tools = model picks wrong one for bare ticker queries.
CORE_TOOLS = {
    "robinhood_get_equity_quotes",   # bare ticker queries -- the most common case
}

BUNDLED_SKILLS = ["technical-analysis", "options-playbook", "risk-management"]


@dataclass
class BuiltAgent:
    """Everything ``build_agent`` hands back to AgentHood."""

    agent: Agent
    tools: list = field(default_factory=list)
    adapter: Any = None
    cost_tracker: Any = None
    rules_tmp: Path | None = None


def discover_skill_dirs() -> list[str]:
    """Find the bundled trading skill directories.

    The skills live inside the package (``agenthood/skills/``) so they ship in
    the wheel. They previously sat at the repo root, which meant a wheel install
    silently loaded no skills at all -- ``pip install agenthood`` gave you an
    agent with none of its trading knowledge.

    The old repo-root location is still checked so an existing checkout that has
    not been rebuilt keeps working.
    """
    pkg_dir = Path(__file__).parent.parent          # .../agenthood
    bases = [
        pkg_dir / "skills",                          # installed + editable (canonical)
        pkg_dir.parent.parent / "skills",            # legacy: repo root
    ]
    dirs = []
    for name in BUNDLED_SKILLS:
        for base in bases:
            candidate = base / name
            if (candidate / "SKILL.md").exists():
                dirs.append(str(candidate))
                break
    return dirs


def _check_api_key(cfg: Any) -> None:
    """Fast-fail: detect missing API key for cloud LLM providers."""
    _cloud_hosts = ("openai.com", "openrouter.ai", "anthropic.com", "googleapis.com")
    if any(h in (cfg.api_base or "") for h in _cloud_hosts) and not cfg.api_key:
        raise RuntimeError(
            "No API key configured for cloud LLM.\n"
            "  Option 1: run 'agenthood setup' to configure interactively\n"
            "  Option 2: add 'api_key' to ~/.agenthood/config.json\n"
            "  Option 3: export OPENROUTER_API_KEY=sk-or-..."
        )


def _write_rules(agent: Agent) -> Path:
    """with_rules() needs a file path, so write the system prompt to a temp file."""
    rules_tmp = Path(tempfile.mktemp(suffix=".md"))
    rules_tmp.write_text(SYSTEM_PROMPT)
    agent.with_rules(str(rules_tmp))
    return rules_tmp


async def _connect_mcp(cfg: Any) -> tuple[Any, list]:
    """Connect to the Robinhood MCP server and load its tools.

    Bypasses agent.with_mcp() -- Robinhood needs STREAMABLE_HTTP (MCP v2).
    """
    from agentu.mcp.transport import MCPServerConfig, TransportType, AuthConfig
    from agentu.mcp.tool import MCPToolAdapter

    auth = AuthConfig.bearer_token(cfg.mcp_token) if cfg.mcp_token else None
    mcp_config = MCPServerConfig(
        name="robinhood",
        transport_type=TransportType.STREAMABLE_HTTP,
        url=cfg.mcp_url,
        auth=auth,
        timeout=30,
    )
    adapter = MCPToolAdapter(mcp_config)
    tools = await adapter.load_tools()
    return adapter, tools


def _register_tools(agent: Agent, tools: list, core_tools_only: bool) -> None:
    """Register MCP tools, deferring everything outside CORE_TOOLS."""
    if core_tools_only:
        # Only load core tools, skip the rest entirely
        for tool in tools:
            if tool.name in CORE_TOOLS:
                agent._add_tool_internal(tool, deferred=False)
    else:
        # Core tools always active; rest deferred behind search_tools
        for tool in tools:
            agent._add_tool_internal(tool, deferred=tool.name not in CORE_TOOLS)

    logger.info("Loaded %d Robinhood tools (deferred search enabled)", len(tools))


def _apply_optional_subsystems(agent: Agent, cfg: Any) -> None:
    """Wire the subsystems that are only enabled when configured."""
    # -- Sandbox -----------------------------------------------------------
    # Subprocess isolation for codemode -- limits memory, allows network for MCP
    if cfg.codemode and cfg.sandbox:
        agent.with_sandbox(codemode=True, max_memory_mb=256, allow_network=True)

    # -- Cache -------------------------------------------------------------
    if cfg.cache_preset:
        agent.with_cache(preset=cfg.cache_preset)
        logger.info("Cache enabled (preset=%s)", cfg.cache_preset)

    # -- Guardrails --------------------------------------------------------
    if cfg.guardrails:
        try:
            from agentu.middleware.guardrails import ContentFilter, MaxLength
            agent.with_guardrails(
                input_guardrails=[MaxLength(max_chars=8000)],
                output_guardrails=[ContentFilter(
                    block=["guaranteed profit", "100% return", "risk-free"]
                )],
                max_corrections=2,
            )
        except ImportError:
            pass

    # -- OpenTelemetry -----------------------------------------------------
    if cfg.otel_endpoint:
        try:
            agent.with_otel(endpoint=cfg.otel_endpoint, service_name="agenthood")
            logger.info("OpenTelemetry tracing enabled -> %s", cfg.otel_endpoint)
        except Exception as e:
            logger.warning("OTel setup failed: %s", e)

    # -- Notifications -----------------------------------------------------
    if cfg.notify_targets:
        try:
            agent.with_notifier(targets=cfg.notify_targets, title="AgentHood Alert")
            logger.info("Notifier configured: %s", cfg.notify_targets)
        except Exception as e:
            logger.warning("Notifier setup failed: %s", e)


async def build_agent(cfg: Any) -> BuiltAgent:
    """Construct and fully wire an agentu Agent from *cfg*."""
    _check_api_key(cfg)

    # -- Build Agent -------------------------------------------------------
    kwargs: dict[str, Any] = {
        "enable_memory": cfg.enable_memory,
        "auto_discover_rules": False,   # we inject rules ourselves
        "max_turns": cfg.max_turns,
        "codemode": cfg.codemode,
        "enable_rationale_recording": cfg.rationale_recording,
    }
    if cfg.model:
        kwargs["model"] = cfg.model
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base

    agent = Agent("agenthood", **kwargs)

    # -- System prompt -----------------------------------------------------
    rules_tmp = _write_rules(agent)

    # -- Safety hooks ------------------------------------------------------
    guard = trading_guard(
        dry_run=cfg.dry_run,
        max_order_value=cfg.max_order_value,
        allowed_symbols=cfg.allowed_symbols,
    )
    agent.with_hooks(pre_tool=guard)

    # -- Permissions -------------------------------------------------------
    # Safety is enforced by the trading_guard pre-tool hook (blocks orders
    # in dry_run, enforces max_order_value, etc.) so we don't need agentu's
    # ask-writes mode -- that mode blocks ALL MCP tools since agentu marks
    # every MCP tool as WRITE, making infer() unusable for read queries.
    # Only explicitly raise the mode if the user chose it themselves.
    if cfg.permissions_mode in ("ask-writes", "plan"):
        agent.with_permissions(mode=cfg.permissions_mode)
    # Otherwise: leave at "auto" -- trading_guard is the safety layer.

    # -- Context compaction ------------------------------------------------
    # Auto-summarise old context when the window fills up
    agent.with_context(
        max_tokens=cfg.context_max_tokens,
        compaction=cfg.context_compaction,
        keep_recent=5,
        max_result_chars=3000,
    )

    # -- Memory consolidation ----------------------------------------------
    # Background LLM task that synthesises important memories every N turns
    if cfg.memory_consolidation:
        agent.with_consolidation(every=cfg.memory_consolidation_every)

    # -- CostTracker middleware --------------------------------------------
    cost_tracker = None
    try:
        from agentu.middleware.middleware import CostTracker
        cost_tracker = CostTracker()
        agent.use(cost_tracker)
    except ImportError:
        pass

    # -- Robinhood MCP connection ------------------------------------------
    adapter, tools = await _connect_mcp(cfg)
    _register_tools(agent, tools, cfg.core_tools_only)

    # -- Skills ------------------------------------------------------------
    skills_to_load = discover_skill_dirs() + list(cfg.skills)
    if skills_to_load:
        await agent.with_skills(skills_to_load)
        logger.info("Loaded %d trading skills", len(skills_to_load))

    _apply_optional_subsystems(agent, cfg)

    # -- Inbox watcher -----------------------------------------------------
    if cfg.inbox_path:
        inbox = Path(cfg.inbox_path)
        inbox.mkdir(parents=True, exist_ok=True)
        agent.with_inbox(str(inbox), poll_interval=cfg.inbox_poll_interval)
        await agent.start_inbox()
        logger.info("Signal inbox watching: %s", inbox)

    return BuiltAgent(
        agent=agent,
        tools=tools,
        adapter=adapter,
        cost_tracker=cost_tracker,
        rules_tmp=rules_tmp,
    )
