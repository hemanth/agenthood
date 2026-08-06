"""Normalise whatever agentu.infer() returns into a plain string."""
from __future__ import annotations

from typing import Any


def _extract_response(result: Any) -> str:
    """Normalise whatever agentu.infer() returns into a plain string.

    agentu may return:
      - a plain str
      - a dict with 'text', 'content', 'result', or 'response'
      - an object with .text / .content
      - something else entirely (int, list, None)

    This function always returns a string.
    """
    if isinstance(result, str):
        # Check if it's raw Robinhood JSON leaked as a string
        stripped = result.strip()
        if stripped.startswith('{') and '"data"' in stripped and '"guide"' in stripped:
            try:
                import json
                parsed = json.loads(stripped)
                if isinstance(parsed, dict) and "data" in parsed:
                    return json.dumps(parsed["data"], indent=2)
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    if isinstance(result, dict):
        # Raw Robinhood tool response leaked through (model didn't format it)
        if "data" in result and "guide" in result:
            import json
            return json.dumps(result["data"], indent=2)
        for key in ("text", "content", "result", "response", "output"):
            val = result.get(key)
            if val is not None:
                if isinstance(val, (dict, list)):
                    import json
                    return json.dumps(val, indent=2)
                # String value might be raw Robinhood JSON -- recurse
                return _extract_response(val)
        # error key
        err = result.get("error")
        if err:
            return str(err)
        # Unknown dict -- pretty-print rather than repr
        import json
        return json.dumps(result, indent=2, default=str)

    for attr in ("text", "content", "result"):
        val = getattr(result, attr, None)
        if val is not None:
            return str(val)

    return str(result)
