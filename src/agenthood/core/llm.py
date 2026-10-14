"""Direct OpenAI-compatible HTTP client -- bypasses agentu entirely.

Used by ``AgentHood.analyze()``: the data is pre-fetched via MCP, so the model
must analyze it without being offered any tool definitions.  Sending tools here
makes small models attempt tool calls instead of writing the report.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

from .prompts import ANALYST_SYSTEM_PROMPT, CONVERSE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TEMPERATURE = 0.3
MAX_TOKENS = 2500

# An interactive prompt cannot honour a multi-minute retry-after, so cap backoff
# and degrade instead of hanging.
MAX_BACKOFF_SECS = 4

# Providers report how long the limit lasts. A per-day quota says "try again in
# 202s" -- sleeping 1s then 2s against that just burns three seconds and still
# fails, so give up immediately and let the caller fall to another model.
_cooldowns: dict[str, float] = {}


def _retry_after_secs(resp) -> float | None:
    """Seconds the provider asked us to wait, if it said."""
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _cooling_down(model: str) -> float:
    """Seconds left on a recorded cooldown for *model*, 0 if it is usable."""
    until = _cooldowns.get(model)
    if until is None:
        return 0.0
    left = until - time.monotonic()
    if left <= 0:
        _cooldowns.pop(model, None)
        return 0.0
    return left


def _set_cooldown(model: str, secs: float) -> None:
    _cooldowns[model] = time.monotonic() + secs


def reset_cooldowns() -> None:
    """Forget every recorded cooldown (used by tests and .model switches)."""
    _cooldowns.clear()

CONVERSE_TEMPERATURE = 0.4
CONVERSE_MAX_TOKENS = 1200

OLLAMA_BASE = "http://localhost:11434/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _base_for_model(cfg: Any, model: str, cur_base: str, cur_key: str) -> tuple[str, str]:
    """Route a fallback model id to the right provider base and key.

    ``model_fallbacks`` may name models from a different provider than the
    primary (OpenRouter ids while the primary is Groq, say), so the base URL has
    to follow the model rather than stay pinned.
    """
    if ":latest" in model or "ollama" in model.lower():
        return OLLAMA_BASE, "ollama"
    if "/" in model:
        key = os.environ.get("OPENROUTER_API_KEY", "") or cfg.api_key or cur_key
        return OPENROUTER_BASE, key
    return cur_base, cur_key


def analyze_llm_config(cfg: Any) -> tuple[str, str, str]:
    """Return (model, api_base, api_key) for the analyze LLM path.

    Prefers dedicated analyze config (e.g. Groq), falls back to main LLM.
    Also checks GROQ_API_KEY env var as a convenience.
    """
    # Dedicated analyze provider configured?
    if cfg.analyze_model and cfg.analyze_api_base:
        key = cfg.analyze_api_key or os.environ.get("GROQ_API_KEY", "") or cfg.api_key
        return cfg.analyze_model, cfg.analyze_api_base, key
    # Fallback: main LLM
    return cfg.model, cfg.api_base, cfg.api_key


def chat_llm_config(cfg: Any) -> tuple[str, str, str]:
    """Return (model, api_base, api_key) for the main conversational path."""
    return cfg.model, cfg.api_base, cfg.api_key


def _messages(
    prompt: str,
    system: str = ANALYST_SYSTEM_PROMPT,
    history: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system}]
    for role, content in (history or []):
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _content_of(body: dict) -> tuple[str, str | None, bool]:
    """Pull the assistant text out of an OpenAI-shaped response body.

    Returns (content, error, from_reasoning). Some providers return errors with
    HTTP 200. Thinking models (nemotron) leave `content` empty and put text in
    `reasoning` -- that is usually chain-of-thought rather than the answer, so
    it is flagged and treated as a last resort by the caller.
    """
    choices = body.get("choices")
    if not choices:
        return "", body.get("error", {}).get("message", "No choices in response"), False
    msg = choices[0].get("message", {})
    content = msg.get("content") or ""
    if content.strip():
        return content, None, False
    reasoning = msg.get("reasoning") or ""
    if reasoning.strip():
        return reasoning, None, True
    return "", "Empty response from LLM", False


async def _completion(
    model: str,
    api_base: str,
    api_key: str,
    messages: list[dict[str, str]],
    *,
    timeout: int = 60,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_retries: int = MAX_RETRIES,
    max_backoff: int = MAX_BACKOFF_SECS,
) -> tuple[str, str | None, bool]:
    """POST a chat completion with retry/backoff.

    Returns (content, error, from_reasoning).

    Backoff is capped: a provider asking us to wait minutes is not something an
    interactive prompt can honour, so we give up and let the caller degrade.
    """
    left = _cooling_down(model)
    if left:
        return "", f"rate limited (429), {int(left)}s left on {model}", False

    last_err = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                resp = await http.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                if resp.status_code == 429:
                    last_err = "rate limited (429)"
                    retry_after = _retry_after_secs(resp)
                    # Longer than we can plausibly wait -- stop now, remember it,
                    # and let the caller try a different model instead of
                    # sleeping through retries that cannot succeed.
                    if retry_after is not None and retry_after > max_backoff:
                        _set_cooldown(model, retry_after)
                        logger.warning(
                            "%s rate limited for %ss -- skipping to fallback",
                            model, int(retry_after),
                        )
                        break
                    wait = min(retry_after or (2 ** attempt), max_backoff)
                    if attempt == max_retries - 1:
                        break
                    logger.warning("Rate limited (429), retrying in %gs...", wait)
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                content, err, from_reasoning = _content_of(resp.json())
                if content:
                    return content, None, from_reasoning
                last_err = err
                continue

        except Exception as exc:
            last_err = str(exc)
            logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)

    return "", last_err, False


async def _completion_with_fallbacks(
    cfg: Any,
    model: str,
    api_base: str,
    api_key: str,
    messages: list[dict[str, str]],
    **kw: Any,
) -> tuple[str, str | None]:
    """Try the primary model, then each configured fallback model.

    On a shared free tier the primary is exactly the one that gets rate limited,
    so a second provider is usually the difference between an answer and none.
    """
    # A reasoning-only reply is chain-of-thought, not an answer. Hold on to the
    # first one and keep trying; only surface it if nothing better turns up.
    salvage = ""

    content, err, from_reasoning = await _completion(
        model, api_base, api_key, messages, **kw
    )
    if content and not from_reasoning:
        return content, None
    salvage = salvage or content

    for alt in getattr(cfg, "model_fallbacks", None) or []:
        alt_base, alt_key = _base_for_model(cfg, alt, api_base, api_key)
        content, err, from_reasoning = await _completion(
            alt, alt_base, alt_key, messages, **{**kw, "max_retries": 1}
        )
        if content and not from_reasoning:
            logger.info("used fallback model %s", alt)
            return content, None
        salvage = salvage or content

    if salvage:
        logger.info("no model returned clean content; using reasoning text")
        return salvage, None
    return "", err


async def llm_completion(
    cfg: Any,
    prompt: str,
    _progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Call the LLM directly with NO tool definitions.

    Retries with backoff on 429, then tries each configured fallback model.

    Deliberately does NOT fall back to the agentu tool-calling path: the caller
    (analyze) passes a multi-KB market-data dump, and routing that through tool
    selection produces quote spam for whatever symbols happen to appear inside
    the JSON rather than an analysis.
    """
    model, api_base, api_key = analyze_llm_config(cfg)
    logger.info("LLM completion using model=%s base=%s", model, api_base[:40])

    content, err = await _completion_with_fallbacks(
        cfg, model, api_base, api_key, _messages(prompt)
    )
    if content:
        return content

    logger.warning("All LLM attempts failed: %s", err)
    return (
        f"Could not generate the analysis: {err}. "
        "The data was fetched fine, so this is the LLM provider. "
        "Try again shortly, or switch model with .model."
    )


async def converse(
    cfg: Any,
    prompt: str,
    history: list[tuple[str, str]] | None = None,
    max_history: int = 10,
) -> str:
    """Answer *prompt* conversationally, with no tools offered.

    Used when agentu's tool router finds no match. The user asked something --
    "hi", "what can you do", "what is an ETF" -- that simply has no tool behind
    it, and the right response is an answer, not a report that routing failed.

    Returns "" if the call fails, so the caller can decide what to do.
    """
    model, api_base, api_key = chat_llm_config(cfg)
    messages = _messages(
        prompt, system=CONVERSE_SYSTEM_PROMPT, history=(history or [])[-max_history:]
    )

    content, err = await _completion_with_fallbacks(
        cfg, model, api_base, api_key, messages,
        timeout=30, temperature=CONVERSE_TEMPERATURE, max_tokens=CONVERSE_MAX_TOKENS,
    )
    if content:
        return content.strip()

    logger.debug("converse() failed: %s", err)
    return ""


async def stream_completion(cfg: Any, prompt: str):
    """Async generator that yields tokens as they arrive via SSE streaming."""
    import json as _json

    model, api_base, api_key = analyze_llm_config(cfg)

    async with httpx.AsyncClient(timeout=120) as http:
        async with http.stream(
            "POST",
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": _messages(prompt),
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(payload)
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        yield token
                except (_json.JSONDecodeError, IndexError, KeyError):
                    continue
