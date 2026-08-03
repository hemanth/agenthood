"""Tests for _extract_response() — the result-normalization layer.

Covers every branch of the dict-picking logic without needing a live agent.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pytest
from agenthood.agent import _extract_response


class TestExtractResponse:

    # ── String passthrough ────────────────────────────────────────────────

    def test_plain_string_returned_as_is(self):
        assert _extract_response("hello") == "hello"

    def test_non_dict_non_str_coerced(self):
        assert _extract_response(42) == "42"
        assert _extract_response(None) == "None"
        assert _extract_response([1, 2]) == "[1, 2]"

    # ── Dict: response key ─────────────────────────────────────────────────

    def test_response_key_preferred(self):
        assert _extract_response({"response": "The price is $42"}) == "The price is $42"

    def test_response_key_takes_priority_over_reasoning(self):
        r = {"response": "final answer", "reasoning": "some chain"}
        assert _extract_response(r) == "final answer"

    # ── Dict: error key ────────────────────────────────────────────────────

    def test_error_key_returned_as_string(self):
        r = {"error": "No appropriate tool found"}
        out = _extract_response(r)
        assert "No appropriate tool found" in out

    def test_error_takes_precedence(self):
        r = {"error": "tool failed", "reasoning": "I tried search_tools but..."}
        out = _extract_response(r)
        assert "tool failed" in out

    # ── Dict: reasoning key ────────────────────────────────────────────────

    def test_unknown_key_stringified(self):
        r = {"reasoning": "Based on the RSI data..."}
        out = _extract_response(r)
        assert isinstance(out, str)
        assert "RSI" in out

    # ── Dict: result key ──────────────────────────────────────────────────

    def test_result_key_json_string(self):
        payload = json.dumps({"data": {"price": 42.5}})
        r = {"result": payload}
        out = _extract_response(r)
        assert "42.5" in out

    def test_result_key_json_with_data_wrapper(self):
        payload = json.dumps({"data": {"symbol": "AAPL", "price": 195}})
        r = {"result": payload}
        out = _extract_response(r)
        assert "AAPL" in out

    def test_result_key_dict_direct(self):
        r = {"result": {"symbol": "TSLA", "bid": 200}}
        out = _extract_response(r)
        assert "TSLA" in out

    def test_result_key_invalid_json_fallback(self):
        r = {"result": "not json {{{ bad"}
        out = _extract_response(r)
        assert out == "not json {{{ bad"

    # ── Empty / unknown dict ──────────────────────────────────────────────

    def test_empty_dict_stringified(self):
        out = _extract_response({})
        assert isinstance(out, str)

    def test_unknown_keys_stringified(self):
        out = _extract_response({"foo": "bar", "baz": 99})
        assert isinstance(out, str)

    def test_nested_dict_uses_json_dumps(self):
        out = _extract_response({"result": {"data": {"price": 42}}})
        assert '"price": 42' in out

    def test_list_value_uses_json_dumps(self):
        out = _extract_response({"output": [1, 2, 3]})
        assert out == '[1, 2, 3]'

    def test_nested_dict_is_valid_json(self):
        out = _extract_response({"result": {"data": {"price": 42}}})
        assert json.loads(out)

