"""Tests for AgentHoodConfig — loading, env overrides, defaults, validation.

No network, no MCP, no LLM required.
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from agenthood.config import AgentHoodConfig, load_config, _jwt_payload_str


# ── Default values ────────────────────────────────────────────────────────────

class TestDefaults:
    def test_dry_run_default_true(self):
        cfg = AgentHoodConfig()
        assert cfg.dry_run is True

    def test_max_order_value_default(self):
        cfg = AgentHoodConfig()
        assert cfg.max_order_value == 500

    def test_allowed_symbols_empty(self):
        cfg = AgentHoodConfig()
        assert cfg.allowed_symbols == []

    def test_max_turns_default(self):
        cfg = AgentHoodConfig()
        assert cfg.max_turns == 20

    def test_scan_interval_default_zero(self):
        cfg = AgentHoodConfig()
        assert cfg.scan_interval == 0

    def test_context_max_tokens(self):
        cfg = AgentHoodConfig()
        assert cfg.context_max_tokens > 0

    def test_memory_enabled_by_default(self):
        cfg = AgentHoodConfig()
        assert cfg.enable_memory is True

    def test_guardrails_enabled_by_default(self):
        cfg = AgentHoodConfig()
        assert cfg.guardrails is True

    def test_rationale_recording_enabled(self):
        cfg = AgentHoodConfig()
        assert cfg.rationale_recording is True


# ── JSON file loading ─────────────────────────────────────────────────────────

class TestJsonLoading:
    def _write_cfg(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.flush()
        return f.name

    def test_loads_dry_run_false(self):
        path = self._write_cfg({"dry_run": False, "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.dry_run is False

    def test_loads_max_order_value(self):
        path = self._write_cfg({"max_order_value": 1234.5, "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.max_order_value == 1234.5

    def test_loads_model(self):
        path = self._write_cfg({"model": "gpt-4o", "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.model == "gpt-4o"

    def test_loads_allowed_symbols(self):
        path = self._write_cfg({"allowed_symbols": ["AAPL", "MSFT"], "mcp_token": "tok"})
        cfg = load_config(path)
        assert "AAPL" in cfg.allowed_symbols

    def test_ignores_unknown_keys(self):
        path = self._write_cfg({"unknown_field": "ignored", "mcp_token": "tok"})
        cfg = load_config(path)
        assert not hasattr(cfg, "unknown_field")

    def test_missing_file_uses_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.json"))
        assert isinstance(cfg, AgentHoodConfig)

    def test_scan_prompt_loadable(self):
        p = "custom scan prompt"
        path = self._write_cfg({"scan_prompt": p, "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.scan_prompt == p

    def test_string_bool_false_coerced(self):
        path = self._write_cfg({"dry_run": "false", "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.dry_run is False

    def test_string_bool_true_coerced(self):
        path = self._write_cfg({"dry_run": "true", "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.dry_run is True

    def test_string_bool_zero_coerced(self):
        path = self._write_cfg({"dry_run": "0", "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.dry_run is False

    def test_string_int_coerced(self):
        path = self._write_cfg({"max_turns": "30", "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.max_turns == 30
        assert isinstance(cfg.max_turns, int)

    def test_string_float_coerced(self):
        path = self._write_cfg({"max_order_value": "1250.50", "mcp_token": "tok"})
        cfg = load_config(path)
        assert cfg.max_order_value == 1250.5
        assert isinstance(cfg.max_order_value, float)



# ── Env var overrides ─────────────────────────────────────────────────────────

class TestEnvOverrides:
    def test_openrouter_key_sets_base(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = load_config(str(tmp_path / "x.json"))
        assert "openrouter" in cfg.api_base
        assert cfg.api_key == "sk-or-test"

    def test_openai_key_preferred_over_openrouter(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        cfg = load_config(str(tmp_path / "x.json"))
        assert cfg.api_key == "sk-openai"

    def test_dry_run_env_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AGENTHOOD_DRY_RUN", "false")
        cfg = load_config(str(tmp_path / "x.json"))
        assert cfg.dry_run is False

    def test_dry_run_env_0(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AGENTHOOD_DRY_RUN", "0")
        cfg = load_config(str(tmp_path / "x.json"))
        assert cfg.dry_run is False

    def test_dry_run_env_no(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AGENTHOOD_DRY_RUN", "no")
        cfg = load_config(str(tmp_path / "x.json"))
        assert cfg.dry_run is False

    def test_robinhood_token_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ROBINHOOD_TOKEN", "rh-jwt-test")
        cfg = load_config(str(tmp_path / "x.json"))
        assert cfg.mcp_token == "rh-jwt-test"

    def test_model_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AGENTHOOD_MODEL", "ollama/llama3")
        cfg = load_config(str(tmp_path / "x.json"))
        assert cfg.model == "ollama/llama3"


# ── JWT helper ────────────────────────────────────────────────────────────────

class TestJwtPayload:
    def test_valid_jwt(self):
        import base64
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(b'{"sub":"robinhood-user"}').rstrip(b"=").decode()
        token = f"{header}.{payload}.fakesig"
        assert "robinhood" in _jwt_payload_str(token)

    def test_invalid_jwt_empty(self):
        assert _jwt_payload_str("not.a.jwt.at.all.extra") == ""

    def test_empty_string(self):
        assert _jwt_payload_str("") == ""

    def test_two_parts_only(self):
        assert _jwt_payload_str("a.b") == ""
