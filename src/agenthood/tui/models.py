"""Model catalogue, numbered picker, and live model hot-swap."""
from __future__ import annotations

import os
from typing import Optional

from rich import box
from rich.panel import Panel
from rich.table import Table

from .theme import (
    C_ACCENT, C_DIM, C_DIMMER, C_FG, C_GREEN, C_LABEL, C_WARN,
    PROMPT_MAIN, _PROVIDER_COL, _SPEED_COL, console,
)

# ── Curated model catalogue ───────────────────────────────────────────────────
MODELS: list[tuple[str, str, str, str, str]] = [
    # (model_id, display_label, provider, speed, note)
    # --- OpenRouter free (ranked by tool-calling benchmark) ------------------
    ("meta-llama/llama-3.3-70b-instruct:free","Llama 3.3 70B",     "openrouter", "fast",   "#1 free: best JSON tool-call accuracy"),
    ("openai/gpt-oss-120b:free",              "GPT-OSS 120B",      "openrouter", "fast",   "#2 free: built for agentic tool-use"),
    ("openai/gpt-oss-20b:free",               "GPT-OSS 20B",       "openrouter", "fast",   "#3 free: fast, reliable tool-calls"),
    ("qwen/qwen3-coder:free",                 "Qwen3 Coder",       "openrouter", "fast",   "#4 free: strong agent/function-call"),
    ("deepseek/deepseek-r1-0528:free",        "DeepSeek R1-0528",  "openrouter", "medium", "#5 free: best reasoning + tool calls"),
    ("google/gemini-flash-1.5:free",          "Gemini 1.5 Flash",  "openrouter", "fast",   "#6 free: long context, fast"),
    ("mistralai/mistral-7b-instruct:free",    "Mistral 7B",        "openrouter", "fast",   "#7 free: lightweight"),
    # --- OpenRouter paid ------------------------------------------------------
    ("openai/gpt-4o",                         "GPT-4o",            "openrouter", "medium", "best all-round"),
    ("openai/o3-mini",                        "o3-mini",           "openrouter", "medium", "strong reasoning"),
    ("anthropic/claude-sonnet-4-5",           "Claude Sonnet 4.5", "openrouter", "medium", "excellent prose"),
    ("anthropic/claude-opus-4-5",             "Claude Opus 4.5",   "openrouter", "slow",   "most capable"),
    ("google/gemini-2.0-flash-001",           "Gemini 2.0 Flash",  "openrouter", "fast",   "long context"),
    ("google/gemini-2.5-pro",                 "Gemini 2.5 Pro",    "openrouter", "medium", "deep reasoning"),
    # --- Ollama local ---------------------------------------------------------
    ("qwen3:latest",                          "Qwen3",             "ollama",     "fast",   "local, strong tool-calls"),
    ("llama3.3:latest",                       "Llama 3.3",         "ollama",     "fast",   "local, general"),
    ("deepseek-r1:latest",                    "DeepSeek R1",       "ollama",     "slow",   "local, reasoning"),
    ("mistral:latest",                        "Mistral 7B",        "ollama",     "fast",   "local, lightweight"),
    ("codestral:latest",                      "Codestral",         "ollama",     "medium", "local, code-optimised"),
]


def _short_model(model: str) -> str:
    return model.split("/")[-1].replace(":free", "").replace(":latest", "")


def show_model_picker(current: str) -> Optional[str]:
    """Interactive numbered model table.  Returns new model id or None."""
    console.print()
    t = Table(box=box.SIMPLE, show_header=True,
              header_style=f"bold {C_DIM}", padding=(0, 2), expand=False)
    t.add_column("#",        style=C_DIMMER, width=4,  no_wrap=True)
    t.add_column("Model",    style=C_FG,     min_width=20, no_wrap=True)
    t.add_column("Provider", width=11,        no_wrap=True)
    t.add_column("Speed",    width=7,         no_wrap=True)
    t.add_column("Note",     style=C_DIMMER,  no_wrap=True)

    last_prov = None
    for idx, (mid, label, prov, speed, note) in enumerate(MODELS, 1):
        if prov != last_prov:
            t.add_row("", "", "", "", "")
            last_prov = prov
        mark = f"[{C_ACCENT}]*[/{C_ACCENT}] " if mid == current else "  "
        t.add_row(
            str(idx),
            f"{mark}{label}",
            f"[{_PROVIDER_COL.get(prov, C_DIM)}]{prov}[/{_PROVIDER_COL.get(prov, C_DIM)}]",
            f"[{_SPEED_COL.get(speed, C_DIM)}]{speed}[/{_SPEED_COL.get(speed, C_DIM)}]",
            note,
        )

    console.print(Panel(
        t,
        border_style=C_DIM,
        title=f"[{C_LABEL}]model picker[/{C_LABEL}]",
        title_align="left",
        subtitle=f"[{C_DIMMER}]number / name prefix / full id  --  empty to cancel[/{C_DIMMER}]",
        subtitle_align="right",
    ))

    try:
        raw = input(PROMPT_MAIN).strip()
    except (EOFError, KeyboardInterrupt):
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]")
        return None

    if not raw:
        console.print(f"  [{C_DIM}]cancelled[/{C_DIM}]")
        return None

    if raw.isdigit():
        n = int(raw)
        if 1 <= n <= len(MODELS):
            mid, label = MODELS[n - 1][0], MODELS[n - 1][1]
            console.print(f"  [{C_GREEN}]switched to {label}[/{C_GREEN}]")
            return mid
        console.print(f"  [{C_WARN}]! invalid number[/{C_WARN}]")
        return None

    raw_low = raw.lower()
    for mid, label, *_ in MODELS:
        if mid.lower().startswith(raw_low) or label.lower().startswith(raw_low):
            console.print(f"  [{C_GREEN}]switched to {label}[/{C_GREEN}]")
            return mid

    # Accept arbitrary model id  (has / or : or multiple -)
    if "/" in raw or ":" in raw or raw.count("-") >= 2:
        console.print(f"  [{C_GREEN}]switched to {raw}[/{C_GREEN}]")
        return raw

    console.print(f"  [{C_WARN}]! no match for {raw!r}[/{C_WARN}]")
    return None


def _resolve_model(arg: str) -> Optional[str]:
    if arg.isdigit():
        n = int(arg)
        if 1 <= n <= len(MODELS):
            label = MODELS[n - 1][1]
            console.print(f"  [{C_GREEN}]switched to {label}[/{C_GREEN}]")
            return MODELS[n - 1][0]
        console.print(f"  [{C_WARN}]! invalid number[/{C_WARN}]")
        return None
    raw_low = arg.lower()
    for mid, label, *_ in MODELS:
        if mid.lower().startswith(raw_low) or label.lower().startswith(raw_low):
            return mid
    if "/" in arg or ":" in arg or arg.count("-") >= 2:
        return arg
    console.print(f"  [{C_WARN}]! no match for {arg!r}[/{C_WARN}]")
    return None


def apply_model_switch(agent, new_model: str) -> None:
    """Hot-swap LLM model on running agent (no restart needed)."""
    try:
        if agent._agent is not None:
            agent._agent.model = new_model
        agent.config.model = new_model

        # Auto-configure API base & key depending on provider
        if ":latest" in new_model or "ollama" in new_model.lower():
            agent.config.api_base = "http://localhost:11434/v1"
            if agent._agent is not None:
                agent._agent.api_base = "http://localhost:11434/v1"
                agent._agent.api_key = "ollama"
        elif ":free" in new_model or "/" in new_model:
            openrouter_key = os.environ.get("OPENROUTER_API_KEY", "") or agent.config.api_key
            agent.config.api_base = "https://openrouter.ai/api/v1"
            if agent._agent is not None:
                agent._agent.api_base = "https://openrouter.ai/api/v1"
                if openrouter_key:
                    agent._agent.api_key = openrouter_key
        elif "groq" in new_model.lower() or "llama-3" in new_model.lower():
            groq_key = os.environ.get("GROQ_API_KEY", "") or agent.config.analyze_api_key or agent.config.api_key
            agent.config.api_base = "https://api.groq.com/openai/v1"
            if agent._agent is not None:
                agent._agent.api_base = "https://api.groq.com/openai/v1"
                if groq_key:
                    agent._agent.api_key = groq_key
    except Exception as exc:
        console.print(f"  [{C_WARN}]! model switch error: {exc}[/{C_WARN}]")
