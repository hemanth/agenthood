"""agenthood configuration."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .token import get_token, token_status

# ── Canonical data directory ──────────────────────────────────────────────────
DATA_DIR = Path.home() / ".agenthood"
CONFIG_FILE = DATA_DIR / "config.json"
LOG_DIR = DATA_DIR / "logs"
SESS_DIR = DATA_DIR / "sessions"


def ensure_dirs() -> None:
    """Create ~/.agenthood and subdirectories if they don't exist."""
    for d in (DATA_DIR, LOG_DIR, SESS_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentHoodConfig:
    # Robinhood MCP
    mcp_url: str = "https://agent.robinhood.com/mcp/trading"
    mcp_token: str = ""

    # LLM
    model: str = "qwen3"          # default: local Ollama
    model_fallbacks: list[str] = field(default_factory=list)  # fallback models
    api_key: str = ""             # set for OpenAI / Gemini
    api_base: str = "http://localhost:11434/v1"            # custom OpenAI-compat base URL

    # Analyze LLM (optional: faster provider for .analyze, falls back to main LLM)
    analyze_model: str = ""       # e.g. "llama-3.3-70b-versatile"
    analyze_api_key: str = ""     # e.g. Groq API key
    analyze_api_base: str = ""    # e.g. "https://api.groq.com/openai/v1"

    # Safety
    dry_run: bool = True          # if True, block place_* tools
    max_order_value: float = 500.0  # max notional per single order (USD)
    allowed_symbols: list[str] = field(default_factory=list)  # empty = all allowed

    # Memory
    enable_memory: bool = True


    # Agent loop
    max_turns: int = 20           # bumped from agentu default (10) to handle multi-step tasks
    codemode: bool = False        # LLM writes Python code instead of JSON tool calls
    core_tools_only: bool = False # load only 15 core tools (for very small models)
    skills: list = field(default_factory=list)  # extra skill paths or GitHub URLs

    # Permissions & safety
    permissions_mode: str = "auto"   # "auto" | "ask-writes" | "plan"

    # Context window management
    context_max_tokens: int = 128_000
    context_compaction: str = "auto"  # "none" | "auto" | "truncate" | "summarize"

    # Memory consolidation (background LLM synthesis)
    memory_consolidation: bool = True
    memory_consolidation_every: int = 30  # turns between consolidations

    # Rationale recording (audit trail for trade decisions)
    rationale_recording: bool = False

    # Sandbox (subprocess isolation for codemode)
    sandbox: bool = False         # enable subprocess sandbox for codemode

    # Inbox watcher (drop signal files → auto-ingested)
    inbox_path: str = ""          # e.g. "./signals" — empty = disabled
    inbox_poll_interval: int = 5  # seconds between polls

    # Loop engineering (Ralph autonomous mode)
    ralph_prompt_file: str = ""   # path to PROMPT.md; enables Ralph loop when set
    ralph_max_iterations: int = 50
    ralph_timeout_minutes: int = 30

    # Scheduling (autonomous scans)
    scan_interval: int = 0        # seconds between autonomous scans, 0 = disabled
    scan_prompt: str = (
        "Run a market scan: check the portfolio for any positions down >5%, "
        "look for momentum breakouts using RSI < 30 or > 70, and summarise findings."
    )

    # Caching
    cache_preset: str = ""        # "basic" | "smart" | "offline" | "distributed" | ""

    # Guardrails
    guardrails: bool = True       # enable input/output guardrails (finance content filter)

    # Streaming
    stream: bool = False          # stream token-by-token to stdout

    # Observability
    otel_endpoint: str = ""       # OpenTelemetry collector endpoint

    # Notifications
    notify_targets: list = field(default_factory=list)  # Apprise URLs


def load_config(path: str | None = None) -> AgentHoodConfig:
    """Load config from ~/.agenthood/config.json, falling back to env vars then defaults."""
    cfg = AgentHoodConfig()
    ensure_dirs()

    # 1. Load from file (explicit --config path or ~/.agenthood/config.json)
    candidates = [path] if path else [str(CONFIG_FILE)]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            with open(candidate) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    expected = getattr(cfg, k)
                    # Coerce types to match the dataclass field defaults
                    if isinstance(expected, bool) and not isinstance(v, bool):
                        # "false", "0", "no" -> False; everything else -> True
                        if isinstance(v, str):
                            v = v.lower() in ("true", "1", "yes", "on")
                        else:
                            v = bool(v)
                    elif isinstance(expected, (int, float)) and isinstance(v, str):
                        try:
                            v = type(expected)(v)
                        except (ValueError, TypeError):
                            pass
                    setattr(cfg, k, v)
            break

    # 2. Override with env vars
    if os.environ.get("ROBINHOOD_TOKEN"):
        cfg.mcp_token = os.environ["ROBINHOOD_TOKEN"]
    if os.environ.get("AGENTHOOD_MODEL"):
        cfg.model = os.environ["AGENTHOOD_MODEL"]
    if os.environ.get("GROQ_API_KEY"):
        cfg.api_key = os.environ["GROQ_API_KEY"]
        cfg.api_base = "https://api.groq.com/openai/v1"
    elif os.environ.get("OPENAI_API_KEY"):
        cfg.api_key = os.environ["OPENAI_API_KEY"]
    elif os.environ.get("OPENROUTER_API_KEY") and not cfg.api_key:
        cfg.api_key = os.environ["OPENROUTER_API_KEY"]
        cfg.api_base = "https://openrouter.ai/api/v1"
    if os.environ.get("AGENTHOOD_DRY_RUN", "").lower() in ("0", "false", "no"):
        cfg.dry_run = False

    # 3. Get a valid Robinhood token (auto-refreshes if near-expiry)
    if not cfg.mcp_token:
        cfg.mcp_token = get_token()

    return cfg


def _discover_mcp_remote_token(url: str) -> str:
    """Look for a cached Bearer token left by mcp-remote for this URL."""
    import hashlib
    mcp_auth_dir = Path.home() / ".mcp-auth"
    if not mcp_auth_dir.exists():
        return ""

    # mcp-remote names token files with an MD5 of the server URL
    url_hash = hashlib.md5(url.encode()).hexdigest()

    for version_dir in mcp_auth_dir.iterdir():
        if not version_dir.is_dir():
            continue
        tokens_file = version_dir / f"{url_hash}_tokens.json"
        # Also try scanning all token files for a Robinhood token
        if not tokens_file.exists():
            for f in version_dir.glob("*_tokens.json"):
                try:
                    data = json.loads(f.read_text())
                    token = data.get("access_token", "")
                    # Robinhood JWTs contain "robinhood" in the payload
                    if token and "robinhood" in _jwt_payload_str(token):
                        return token
                except Exception:
                    continue
        else:
            try:
                data = json.loads(tokens_file.read_text())
                return data.get("access_token", "")
            except Exception:
                pass

    return ""


def _jwt_payload_str(token: str) -> str:
    """Decode JWT payload (no signature verification) for inspection."""
    import base64
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return ""
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        return base64.urlsafe_b64decode(payload).decode("utf-8", errors="replace")
    except Exception:
        return ""
