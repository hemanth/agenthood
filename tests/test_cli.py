import sys
import pytest
from unittest.mock import patch
from agenthood.cli import parse_args

def test_parse_args_default():
    with patch.object(sys, 'argv', ['agenthood']):
        args = parse_args()
        assert args.command is None

def test_parse_args_analyze():
    with patch.object(sys, 'argv', ['agenthood', 'analyze', 'AAPL', 'NVDA']):
        args = parse_args()
        assert args.command == 'analyze'
        assert args.tickers == ['AAPL', 'NVDA']

def test_parse_args_watch():
    with patch.object(sys, 'argv', ['agenthood', 'watch', 'AAPL', 'below', '180']):
        args = parse_args()
        assert args.command == 'watch'
        assert args.ticker == 'AAPL'
        assert args.direction == 'below'
        assert args.price == 180.0

def test_parse_args_daemon():
    with patch.object(sys, 'argv', ['agenthood', 'daemon', 'start']):
        args = parse_args()
        assert args.command == 'daemon'
        assert args.action == 'start'

def test_parse_args_live_flag():
    with patch.object(sys, 'argv', ['agenthood', '--live']):
        args = parse_args()
        assert args.live is True
