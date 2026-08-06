"""Repair tool arguments the model got slightly wrong.

Small models routinely hand back a JSON-encoded string where the schema wants an
array, or a bare scalar where it wants a list. MCP validates strictly and
rejects the call:

    MCP error: invalid params: validating /properties/symbols:
    type: ["NVDA"] has type "string", want one of "null, array"

The user sees a crash for a request that was perfectly clear. Coercing the
obvious cases costs nothing and removes a whole class of flakiness.
"""
from __future__ import annotations

import functools
import inspect
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# agentu renders each parameter spec as "['null', 'array']: description"
_TYPES_RE = re.compile(r"^\s*\[([^\]]*)\]")

_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}


def declared_types(spec: Any) -> set[str]:
    """Extract the declared JSON-schema types from a parameter spec string."""
    if not isinstance(spec, str):
        return set()
    m = _TYPES_RE.match(spec)
    if not m:
        return set()
    return {t.strip().strip("'\"").lower() for t in m.group(1).split(",") if t.strip()}


def coerce_value(value: Any, types: set[str]) -> Any:
    """Nudge *value* toward one of *types*. Returns it unchanged if unsure."""
    if not types or not isinstance(value, str):
        return value

    # A declared string type means the string is probably intended -- leave it.
    if "string" in types:
        return value

    s = value.strip()
    if not s:
        return value

    # '["NVDA"]' or '{"a":1}' sent as text
    if s[0] in "[{" and ("array" in types or "object" in types):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return value

    # bare "NVDA" where a list was wanted
    if "array" in types:
        return [s]

    if "boolean" in types:
        low = s.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False

    if "integer" in types:
        try:
            return int(s)
        except ValueError:
            pass

    if "number" in types:
        try:
            return float(s)
        except ValueError:
            pass

    return value


def coerce_args(parameters: dict | None, kwargs: dict) -> dict:
    """Return *kwargs* with values coerced to the types *parameters* declares."""
    if not parameters:
        return kwargs
    fixed = {}
    for name, value in kwargs.items():
        new = coerce_value(value, declared_types(parameters.get(name)))
        if new is not value and new != value:
            logger.debug("coerced arg %s: %r -> %r", name, value, new)
        fixed[name] = new
    return fixed


def wrap_tool(tool) -> None:
    """Wrap ``tool.function`` in place so its arguments are repaired on call.

    Applied at registration so it covers every caller -- agentu's tool loop and
    our own direct invocations alike.
    """
    original = tool.function
    params = getattr(tool, "parameters", None)

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapper(**kwargs):
            return await original(**coerce_args(params, kwargs))
    else:
        @functools.wraps(original)
        def wrapper(**kwargs):
            return original(**coerce_args(params, kwargs))

    tool.function = wrapper
