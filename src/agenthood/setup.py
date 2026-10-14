"""agenthood setup wizard — interactive first-run configuration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


# ── helpers ────────────────────────────────────────────────────────────────

def _print(msg: str = "", color: str = "") -> None:
    colors = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
               "bold": "\033[1m", "dim": "\033[2m", "reset": "\033[0m", "cyan": "\033[96m"}
    prefix = colors.get(color, "")
    print(f"{prefix}{msg}{colors['reset']}")


def _ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"  {prompt}{hint}: ").strip()
        return val or default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _ask_choice(prompt: str, choices: list[tuple[str, str]], default: int = 0) -> int:
    """Print numbered choices, return selected index."""
    print(f"\n  {prompt}")
    for i, (key, desc) in enumerate(choices):
        marker = "→" if i == default else " "
        _print(f"  {marker} [{i+1}] {key:<28} {desc}", "dim" if i != default else "")
    while True:
        raw = _ask(f"Choice (1-{len(choices)})", str(default + 1))
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return idx
        except ValueError:
            pass
        _print("  Please enter a number.", "yellow")


# ── step 1: Robinhood auth ─────────────────────────────────────────────────

def _check_existing_token() -> str:
    """Return cached Robinhood token if available."""
    from .config import _discover_mcp_remote_token
    return _discover_mcp_remote_token("https://agent.robinhood.com/mcp/trading")


def _run_oauth_flow() -> str:
    """Launch mcp-remote to do the Robinhood OAuth browser flow."""
    _print("\n  Opening browser for Robinhood login...", "cyan")
    _print("  (A browser window will open. Log in and approve access.)", "dim")

    # mcp-remote blocks until auth is complete, then exits
    proc = subprocess.Popen(
        ["npx", "-y", "mcp-remote", "https://agent.robinhood.com/mcp/trading",
         "--once"],           # --once: auth then exit (don't start the proxy)
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # mcp-remote prints the token path on success; wait up to 3 minutes
    try:
        _, stderr = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        _print("  ⚠️  Auth timed out. Try running manually:", "red")
        _print("     npx mcp-remote https://agent.robinhood.com/mcp/trading", "dim")
        return ""

    token = _check_existing_token()
    if token:
        _print("  ✅ Robinhood auth successful!", "green")
    else:
        _print("  ⚠️  Could not find token after auth. Try:", "yellow")
        _print("     npx mcp-remote https://agent.robinhood.com/mcp/trading", "dim")
    return token


def setup_robinhood_auth() -> str:
    """Ensure a Robinhood token exists. Returns the token."""
    _print("\n── Robinhood Authentication ──────────────────────────", "bold")

    token = _check_existing_token()
    if token:
        _print("  ✅ Found existing Robinhood token (from mcp-remote cache)", "green")
        return token

    _print("  No Robinhood token found.", "yellow")
    _print("  agenthood uses Robinhood's official OAuth flow (via mcp-remote).", "dim")
    _print("  Your credentials never touch agenthood — only Robinhood's servers.", "dim")

    choice = _ask("  Authenticate now? (y/n)", "y").lower()
    if choice != "y":
        _print("  ⚠️  Skipping auth. You can run 'agenthood setup' later.", "yellow")
        return ""

    return _run_oauth_flow()


# ── step 2: LLM model setup ────────────────────────────────────────────────

LLM_PROVIDERS = [
    ("openrouter",  "Free & paid models (OpenAI-compat). Recommended."),
    ("openai",      "OpenAI API (GPT-4o, GPT-4.1, etc.)"),
    ("ollama",      "Local Ollama (no API key needed, runs on your machine)"),
    ("other",       "Other OpenAI-compatible endpoint"),
]

OPENROUTER_FREE_MODELS = [
    ("nvidia/nemotron-3-super-120b-a12b:free", "(Recommended) 120B • fast, reliable tool-calls & analysis"),
    ("google/gemma-4-26b-a4b-it:free",         "Gemma 4 26B • fast MoE instruct"),
    ("google/gemma-4-31b-it:free",             "Gemma 4 31B • strong reasoning & formatting"),
    ("nvidia/nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra 550B • deep analysis"),
    ("nvidia/nemotron-3.5-lightning:free",     "Nemotron 3.5 Lightning • low latency"),
    ("cohere/north-mini-code:free",            "Cohere North Mini • lightweight & fast"),
]


def setup_llm() -> dict:
    """Interactively configure the LLM. Returns partial config dict."""
    _print("\n── LLM Model Setup ───────────────────────────────────", "bold")

    # Check env vars first
    if os.environ.get("OPENROUTER_API_KEY"):
        _print("  ✅ Found OPENROUTER_API_KEY in environment", "green")
        use_env = _ask("  Use it? (y/n)", "y").lower()
        if use_env == "y":
            idx = _ask_choice("  Pick a free model:", OPENROUTER_FREE_MODELS)
            result = {
                "model": OPENROUTER_FREE_MODELS[idx][0],
                "api_base": "https://openrouter.ai/api/v1",
            }
            save_key = _ask("  Save API key to config? (y/n)", "n").lower()
            if save_key == "y":
                result["api_key"] = os.environ["OPENROUTER_API_KEY"]
                _print("  Key will be saved to ~/.agenthood/config.json.", "dim")
            else:
                _print("  Key stays in env only (export OPENROUTER_API_KEY=...).", "dim")
            return result

    if os.environ.get("OPENAI_API_KEY"):
        _print("  ✅ Found OPENAI_API_KEY in environment", "green")
        use_env = _ask("  Use it with gpt-4o-mini? (y/n)", "y").lower()
        if use_env == "y":
            result = {"model": "gpt-4o-mini"}
            save_key = _ask("  Save API key to config? (y/n)", "n").lower()
            if save_key == "y":
                result["api_key"] = os.environ["OPENAI_API_KEY"]
            return result

    # Check Ollama
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        if models:
            _print(f"  ✅ Ollama is running with {len(models)} model(s)", "green")
            use_ollama = _ask("  Use local Ollama? (y/n)", "y").lower()
            if use_ollama == "y":
                _print(f"  Available: {', '.join(models[:5])}", "dim")
                model = _ask("  Model name", models[0])
                return {"model": model, "api_base": "http://localhost:11434/v1"}
    except Exception:
        pass

    # Manual provider selection
    idx = _ask_choice("  Choose an LLM provider:", LLM_PROVIDERS)
    provider = LLM_PROVIDERS[idx][0]

    if provider == "openrouter":
        _print("\n  Get a free key at: https://openrouter.ai/keys", "cyan")
        key = _ask("  OpenRouter API key (sk-or-...)")
        if not key:
            _print("  Skipping LLM setup.", "yellow")
            return {}
        idx2 = _ask_choice("  Pick a free model:", OPENROUTER_FREE_MODELS)
        return {
            "model": OPENROUTER_FREE_MODELS[idx2][0],
            "api_key": key,
            "api_base": "https://openrouter.ai/api/v1",
        }

    elif provider == "openai":
        key = _ask("  OpenAI API key (sk-...)")
        model = _ask("  Model", "gpt-4o-mini")
        return {"model": model, "api_key": key}

    elif provider == "ollama":
        _print("  Install Ollama at: https://ollama.com — then run: ollama pull qwen3", "dim")
        model = _ask("  Model name", "qwen3")
        return {"model": model, "api_base": "http://localhost:11434/v1"}

    elif provider == "other":
        base = _ask("  API base URL")
        key = _ask("  API key (leave blank if not needed)", "")
        model = _ask("  Model name")
        cfg = {"model": model, "api_base": base}
        if key:
            cfg["api_key"] = key
        return cfg

    return {}


# ── step 3: safety settings ────────────────────────────────────────────────

def setup_safety() -> dict:
    _print("\n── Safety & Agent Settings ──────────────────────────", "bold")

    dry_run_ans = _ask("  Start in dry_run mode (blocks real orders)? (y/n)", "y").lower()
    dry_run = dry_run_ans != "n"
    if not dry_run:
        _print("  ⚠️  LIVE mode enabled — real orders will be placed!", "red")

    max_val = _ask("  Max order value in USD", "500")
    try:
        max_val = float(max_val)
    except ValueError:
        max_val = 500.0

    _print("\n  Tool strategy:", "cyan")
    _print("  [1] 🔍 tool-search  — agent searches for tools dynamically (recommended)", "dim")
    _print("  [2] ⚡ codemode     — LLM writes Python to batch tool calls (faster for analysis)", "dim")
    mode_choice = _ask("  Choose (1/2)", "1")
    codemode = mode_choice.strip() == "2"

    return {
        "dry_run": dry_run,
        "max_order_value": max_val,
        "codemode": codemode,
    }


# ── main setup flow ────────────────────────────────────────────────────────

def run_setup(config_path: str | None = None) -> None:
    from .config import CONFIG_FILE, ensure_dirs
    ensure_dirs()
    if config_path is None:
        config_path = str(CONFIG_FILE)
    _print("\n╔══════════════════════════════════════════════════════╗", "bold")
    _print("║        agenthood setup wizard                        ║", "bold")
    _print("╚══════════════════════════════════════════════════════╝", "bold")
    _print(f"  Config will be saved to {config_path}", "dim")
    _print("  Press Ctrl+C at any time to cancel.\n", "dim")

    # Load existing config if present
    existing: dict = {}
    if Path(config_path).exists():
        try:
            existing = json.loads(Path(config_path).read_text())
            _print(f"  Found existing {config_path} — will update it.\n", "dim")
        except Exception:
            pass

    cfg = {
        "mcp_url": "https://agent.robinhood.com/mcp/trading",
        "enable_memory": True,
        "scan_interval": 0,
        **existing,
    }

    # Step 1: Robinhood auth
    token = setup_robinhood_auth()
    # Don't store the token in the file — keep it in mcp-remote cache only
    # agenthood auto-discovers it at runtime

    # Step 2: LLM
    llm_cfg = setup_llm()
    cfg.update(llm_cfg)

    # Step 3: Safety
    safety_cfg = setup_safety()
    cfg.update(safety_cfg)

    # Write config (strip api_key if it came from env — don't persist secrets)
    # Only persist api_key if user explicitly typed it in (not from env)
    save_cfg = {k: v for k, v in cfg.items()}

    Path(config_path).write_text(json.dumps(save_cfg, indent=2))

    _print(f"\n  ✅ Saved {config_path}", "green")

    if not token:
        _print("\n── Not ready yet ─────────────────────────────────────", "bold")
        _print("  ⚠️  No Robinhood auth token -- trading tools won't load.", "yellow")
        _print("  Run 'agenthood setup' again, or 'agenthood' will offer to", "dim")
        _print("  authenticate the next time you start it.", "dim")
        print()
        return

    _print("\n── You're ready! ─────────────────────────────────────", "bold")
    _print("  Start the agent:", "dim")
    _print("    agenthood", "cyan")
    _print("  Or one-shot:", "dim")
    _print('    agenthood "What are my open positions?"', "cyan")
    if cfg.get("dry_run", True):
        _print("\n  🟡 DRY RUN is ON — no real orders will be placed.", "yellow")
    else:
        _print("\n  🔴 LIVE mode — the agent can place real orders!", "red")
    print()
