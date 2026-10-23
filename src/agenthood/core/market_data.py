"""Fetch and trim the market data that feeds ``AgentHood.analyze()``.

Pulls quotes / fundamentals / technicals / historicals straight off the MCP
tools in parallel -- no LLM tool discovery involved -- then trims the payload
down to something that fits in a prompt.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Prompt budget for the serialised data dump.
MAX_DUMP_CHARS = 4000
LOOKBACK_DAYS = 45


def _lookback_start() -> str:
    """ISO 8601 timestamp for LOOKBACK_DAYS ago."""
    ago = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    return ago.strftime("%Y-%m-%dT%H:%M:%SZ")


def _sources(sym: str) -> list[tuple[str, str, dict]]:
    """The (label, tool base name, params) triples analyze() pulls."""
    from .tickers import is_crypto_symbol
    start = _lookback_start()
    if is_crypto_symbol(sym):
        pair = sym if sym.endswith("USD") else f"{sym}-USD"
        return [
            ("quotes", "get_crypto_quotes", {"symbols": [pair]}),
        ]
    return [
        ("quotes", "get_equity_quotes", {"symbols": [sym]}),
        ("fundamentals", "get_equity_fundamentals", {"symbols": [sym]}),
        ("analyst_ratings", "get_equity_analyst_ratings", {"symbols": [sym]}),
        ("technicals", "get_equity_technical_indicators", {
            "symbol": sym, "type": "sma", "interval": "day", "start_time": start,
        }),
        ("historicals", "get_equity_historicals", {
            "symbols": [sym], "interval": "day", "start_time": start,
        }),
    ]


def _parse(raw: Any) -> Any:
    """Decode a tool result to JSON when possible, else pass it through."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


async def _fetch_one(hood, label: str, base_name: str, params: dict) -> tuple[str, Any]:
    """Fetch a single data source, trying multiple name patterns."""
    candidates = [f"robinhood_{base_name}", base_name, f"rh_{base_name}"]
    for name in candidates:
        try:
            return label, _parse(await hood._call_tool_direct(name, params))
        except (ValueError, KeyError):
            continue
        except Exception as exc:
            return label, {"error": str(exc)}
    # Last resort
    try:
        return label, _parse(await hood._agent.call(f"robinhood_{base_name}", params))
    except Exception as exc:
        return label, {"error": str(exc)}


def _trim(data_sections: dict) -> dict:
    """Strip the MCP envelope down to the payload each section carries."""
    from .quant import summarize_historicals_quant

    trimmed = {}
    quant = summarize_historicals_quant(data_sections.get("historicals"))
    if quant:
        trimmed["quant_metrics"] = quant

    for key, val in data_sections.items():
        if isinstance(val, str) and val.startswith("ERROR"):
            trimmed[key] = val
            continue
        if isinstance(val, dict):
            inner = val.get("data", val)
            if isinstance(inner, dict) and "results" in inner:
                inner = inner["results"]
            trimmed[key] = inner
        else:
            trimmed[key] = val
    return trimmed


async def fetch_market_data(
    hood,
    sym: str,
    _progress: Optional[Callable[[str], None]] = None,
) -> str:
    """Fetch every source for *sym* in parallel and return a trimmed JSON dump.

    Raises RuntimeError if every source failed -- that means the MCP connection
    or token is broken, and silently analyzing nothing would be worse.
    """
    results = await asyncio.gather(
        *(_fetch_one(hood, label, base, params) for label, base, params in _sources(sym))
    )
    data_sections = dict(results)

    if _progress:
        ok = sum(1 for _, v in results if not (isinstance(v, dict) and "error" in v))
        _progress(f"got {ok}/{len(results)} sources")

    # If ALL sections are errors, surface the problem
    if all(isinstance(v, dict) and "error" in v for v in data_sections.values()):
        tools = hood._all_tools or []
        raise RuntimeError(
            f"Could not fetch any market data for {sym}. "
            f"MCP tools loaded: {len(tools)}. "
            f"Sample names: {[t.name for t in tools[:5]]}. "
            f"Errors: {data_sections}"
        )

    data_dump = json.dumps(_trim(data_sections), indent=1, default=str)
    if len(data_dump) > MAX_DUMP_CHARS:
        data_dump = data_dump[:MAX_DUMP_CHARS] + "\n... (truncated)"
    return data_dump
