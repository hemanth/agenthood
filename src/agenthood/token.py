"""Robinhood OAuth token management — auto-refresh before expiry."""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TOKEN_ENDPOINT = "https://api.robinhood.com/oauth2/token/"
MCP_URL = "https://agent.robinhood.com/mcp/trading"
# Refresh when less than this many seconds remain
REFRESH_BUFFER_SECS = 300  # 5 minutes


def _find_token_file() -> Optional[Path]:
    """Locate the mcp-remote tokens.json for the Robinhood MCP."""
    mcp_auth_dir = Path.home() / ".mcp-auth"
    if not mcp_auth_dir.exists():
        return None
    for version_dir in sorted(mcp_auth_dir.iterdir(), reverse=True):
        if not version_dir.is_dir():
            continue
        for f in version_dir.glob("*_tokens.json"):
            try:
                data = json.loads(f.read_text())
                token = data.get("access_token", "")
                if token and "robinhood" in _jwt_payload_str(token):
                    return f
            except Exception:
                continue
    return None


def _jwt_payload_str(token: str) -> str:
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


def _token_exp(token: str) -> float:
    """Extract expiry timestamp from JWT access token."""
    try:
        import base64
        parts = token.split(".")
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return float(data.get("exp", 0))
    except Exception:
        return 0.0


def _refresh_token(refresh_tok: str, client_id: str) -> Optional[dict]:
    """Call Robinhood's token endpoint with the refresh_token grant."""
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_tok,
        "client_id": client_id,
    }).encode()

    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.error("Token refresh failed: %s", e)
        return None


def _load_client_id(token_file: Path) -> str:
    """Read client_id from the sibling client_info.json."""
    prefix = token_file.stem.replace("_tokens", "")
    client_info = token_file.parent / f"{prefix}_client_info.json"
    try:
        data = json.loads(client_info.read_text())
        return data.get("client_id", "")
    except Exception:
        return ""


class TokenManager:
    """Manages the Robinhood OAuth token: loads, checks expiry, auto-refreshes."""

    def __init__(self):
        self._token_file: Optional[Path] = None
        self._tokens: dict = {}
        self._client_id: str = ""

    def load(self) -> str:
        """Load token from mcp-remote cache. Returns access_token or ''."""
        self._token_file = _find_token_file()
        if not self._token_file:
            return ""
        try:
            self._tokens = json.loads(self._token_file.read_text())
            self._client_id = _load_client_id(self._token_file)
            return self._tokens.get("access_token", "")
        except Exception as e:
            logger.warning("Could not load token file: %s", e)
            return ""

    def get_valid_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        if not self._tokens:
            return self.load()

        access_token = self._tokens.get("access_token", "")
        if not access_token:
            return ""

        exp = _token_exp(access_token)
        remaining = exp - time.time()

        if remaining > REFRESH_BUFFER_SECS:
            logger.debug("Token valid for %.0f more seconds", remaining)
            return access_token

        # Token expired or near-expiry — try refresh
        logger.info(
            "Token expires in %.0f seconds — refreshing...", max(remaining, 0)
        )
        return self._do_refresh() or access_token  # fall back to old token on error

    def _do_refresh(self) -> str:
        """Perform the refresh_token grant and persist new tokens."""
        refresh_tok = self._tokens.get("refresh_token", "")
        if not refresh_tok:
            logger.warning("No refresh_token available — cannot refresh")
            return ""
        if not self._client_id:
            logger.warning("No client_id found — cannot refresh")
            return ""

        new_tokens = _refresh_token(refresh_tok, self._client_id)
        if not new_tokens or "access_token" not in new_tokens:
            logger.error("Token refresh returned no access_token")
            return ""

        # Persist to the same tokens.json so mcp-remote and agenthood stay in sync
        self._tokens = new_tokens
        if self._token_file:
            try:
                self._token_file.write_text(json.dumps(new_tokens, indent=2))
                logger.info("✅ Token refreshed and saved to %s", self._token_file)
            except Exception as e:
                logger.warning("Could not save refreshed token: %s", e)

        return new_tokens["access_token"]

    def seconds_until_expiry(self) -> float:
        token = self._tokens.get("access_token", "")
        if not token:
            return 0.0
        return max(0.0, _token_exp(token) - time.time())

    def is_expired(self) -> bool:
        return self.seconds_until_expiry() <= 0

    def status(self) -> str:
        remaining = self.seconds_until_expiry()
        if remaining <= 0:
            return "❌ expired"
        h = remaining / 3600
        if h > 24:
            return f"✅ valid ({h/24:.1f} days remaining)"
        return f"✅ valid ({h:.1f} hours remaining)"


# Module-level singleton used by config.py
_manager = TokenManager()


def get_token() -> str:
    """Get a valid Robinhood access token, refreshing if needed."""
    return _manager.get_valid_token()


def token_status() -> str:
    """Human-readable token validity status."""
    _manager.load()
    return _manager.status()
