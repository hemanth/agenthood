import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from agenthood import daemon

@pytest.fixture
def tmp_data_dir(tmp_path):
    with patch('agenthood.daemon.DATA_DIR', tmp_path), \
         patch('agenthood.daemon.WATCHES_FILE', tmp_path / 'watches.json'), \
         patch('agenthood.daemon.TRIGGERED_DB', tmp_path / 'triggered.db'), \
         patch('agenthood.daemon.PID_FILE', tmp_path / 'daemon.pid'), \
         patch('agenthood.daemon._LOCK_FILE', tmp_path / '.watches.lock'):
        yield tmp_path

def test_add_list_remove_watch(tmp_data_dir):
    assert daemon.list_watches() == []
    
    daemon.add_watch({'ticker': 'AAPL', 'threshold': 150})
    watches = daemon.list_watches()
    assert len(watches) == 1
    assert watches[0]['ticker'] == 'AAPL'
    assert 'id' in watches[0]
    
    wid = watches[0]['id']
    daemon.add_watch({'ticker': 'TSLA', 'threshold': 200})
    
    assert len(daemon.list_watches()) == 2
    
    daemon.remove_watch(wid)
    
    watches = daemon.list_watches()
    assert len(watches) == 1
    assert watches[0]['ticker'] == 'TSLA'
    
    daemon.remove_watch(watches[0]['id'])
    assert daemon.list_watches() == []
    assert not (tmp_data_dir / 'watches.json').exists()

def test_triggered_db_roundtrip(tmp_data_dir):
    assert daemon.get_pending_triggers() == []
    
    spec1 = {'id': '123', 'ticker': 'AAPL'}
    daemon.write_triggered(spec1, 150.5)
    
    triggers = daemon.get_pending_triggers()
    assert len(triggers) == 1
    assert triggers[0]['spec'] == spec1
    assert triggers[0]['price'] == 150.5
    
    assert daemon.get_pending_triggers() == []

def test_is_daemon_running_no_pid(tmp_data_dir):
    assert not daemon.is_daemon_running()

@patch('os.kill')
def test_is_daemon_running_active(mock_kill, tmp_data_dir):
    mock_kill.return_value = None
    (tmp_data_dir / 'daemon.pid').write_text("12345\n")
    assert daemon.is_daemon_running()
    mock_kill.assert_called_once_with(12345, 0)

@patch('os.kill')
def test_is_daemon_running_stale(mock_kill, tmp_data_dir):
    mock_kill.side_effect = OSError("No such process")
    (tmp_data_dir / 'daemon.pid').write_text("12345\n")
    assert not daemon.is_daemon_running()
