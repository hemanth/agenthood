import json
import base64
import pytest
from unittest.mock import patch, MagicMock
from agenthood.token import _token_exp, token_status, _find_token_file

def test_token_exp_valid_jwt():
    payload = json.dumps({"exp": 1700000000}).encode('utf-8')
    encoded_payload = base64.urlsafe_b64encode(payload).decode('utf-8').rstrip('=')
    token = f"header.{encoded_payload}.signature"
    assert _token_exp(token) == 1700000000.0

def test_token_exp_invalid():
    assert _token_exp("") == 0.0
    assert _token_exp("invalid.token") == 0.0

@patch('agenthood.token._manager.load')
@patch('agenthood.token._manager.seconds_until_expiry')
def test_token_status_no_token(mock_seconds, mock_load):
    mock_load.return_value = ""
    mock_seconds.return_value = 0.0
    assert "expired" in token_status()

@patch('agenthood.token.Path')
def test_find_token_file(mock_path):
    mock_home = MagicMock()
    mock_path.home.return_value = mock_home
    mock_mcp_auth = mock_home / ".mcp-auth"
    mock_mcp_auth.exists.return_value = True
    
    mock_version = MagicMock()
    mock_version.is_dir.return_value = True
    
    mock_token_file = MagicMock()
    
    # A payload that has 'robinhood' string inside
    payload = json.dumps({"aud": "robinhood"}).encode('utf-8')
    encoded_payload = base64.urlsafe_b64encode(payload).decode('utf-8').rstrip('=')
    jwt = f"header.{encoded_payload}.signature"
    
    mock_token_file.read_text.return_value = json.dumps({"access_token": jwt})
    mock_version.glob.return_value = [mock_token_file]
    mock_mcp_auth.iterdir.return_value = [mock_version]
    
    res = _find_token_file()
    assert res == mock_token_file


def test_do_refresh_returns_new_access_token(tmp_path):
    """Regression: _do_refresh fell off the end returning None, so get_valid_token
    returned the old expired token via `self._do_refresh() or access_token`."""
    from agenthood.token import TokenManager

    token_file = tmp_path / "rh_tokens.json"
    expired_payload = base64.urlsafe_b64encode(json.dumps({"exp": 1000}).encode()).decode().rstrip("=")
    expired_jwt = f"h.{expired_payload}.s"

    tm = TokenManager()
    tm._token_file = token_file
    tm._tokens = {"access_token": expired_jwt, "refresh_token": "ref-123"}
    tm._client_id = "client-abc"

    with patch("agenthood.token._refresh_token", return_value={"access_token": "fresh-jwt-token"}):
        tok = tm.get_valid_token()

    assert tok == "fresh-jwt-token"
    assert json.loads(token_file.read_text())["access_token"] == "fresh-jwt-token"
