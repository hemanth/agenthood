from __future__ import annotations
import asyncio
import fcntl
import sqlite3
import json
import logging
import time
import uuid
import sys
import os
import subprocess
from contextlib import contextmanager

logger = logging.getLogger(__name__)

__all__ = ["start_daemon", "stop_daemon", "is_daemon_running", "add_watch",
           "list_watches", "remove_watch", "get_pending_triggers",
           "write_triggered", "DATA_DIR"]

from .config import DATA_DIR

WATCHES_FILE = DATA_DIR / 'watches.json'
TRIGGERED_DB = DATA_DIR / 'triggered.db'
PID_FILE = DATA_DIR / 'daemon.pid'

# ── File locking for watches.json ──────────────────────────────────────────────

_LOCK_FILE = DATA_DIR / '.watches.lock'


@contextmanager
def _file_lock():
    """Exclusive file lock around watches.json read-modify-write operations."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd = open(_LOCK_FILE, 'w')
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


# ── SQLite connection management ───────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """Open (or create) the triggered database and return a connection.

    Each caller is responsible for closing the connection. For the daemon
    long-running loop, a single connection should be held for the process
    lifetime (see ``main()``).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRIGGERED_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute('''CREATE TABLE IF NOT EXISTS triggers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  watch_id TEXT,
                  spec TEXT,
                  price REAL,
                  timestamp REAL,
                  delivered INTEGER DEFAULT 0)''')
    conn.commit()
    return conn


# ── Daemon lifecycle ───────────────────────────────────────────────────────────

def is_daemon_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, OSError):
        # Stale PID file — clean up
        try:
            PID_FILE.unlink()
        except OSError:
            pass
        return False


def start_daemon() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if is_daemon_running():
        return

    log_fd = open(DATA_DIR / 'daemon.log', 'a')
    try:
        subprocess.Popen(
            [sys.executable, '-m', 'agenthood.daemon'],
            start_new_session=True,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    except Exception:
        log_fd.close()
        raise


def stop_daemon() -> None:
    if not PID_FILE.exists():
        return
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 15)
    except (ValueError, OSError):
        pass
    finally:
        try:
            PID_FILE.unlink()
        except OSError:
            pass


# ── Watch CRUD (file-locked) ──────────────────────────────────────────────────

def add_watch(spec: dict) -> None:
    with _file_lock():
        if 'id' not in spec:
            spec['id'] = str(uuid.uuid4())

        watches = _read_watches_unlocked()
        watches.append(spec)
        _write_watches_unlocked(watches)


def list_watches() -> list[dict]:
    with _file_lock():
        return _read_watches_unlocked()


def remove_watch(watch_id: str) -> None:
    with _file_lock():
        watches = _read_watches_unlocked()
        watches = [w for w in watches if w.get('id') != watch_id]
        if watches:
            _write_watches_unlocked(watches)
        else:
            try:
                WATCHES_FILE.unlink()
            except OSError:
                pass


def _read_watches_unlocked() -> list[dict]:
    """Read watches.json — caller must hold the file lock."""
    if not WATCHES_FILE.exists():
        return []
    try:
        with open(WATCHES_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _write_watches_unlocked(watches: list[dict]) -> None:
    """Write watches.json atomically — caller must hold the file lock."""
    tmp = WATCHES_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(watches, f, indent=2)
    tmp.replace(WATCHES_FILE)


# ── Trigger read/write ─────────────────────────────────────────────────────────

def get_pending_triggers() -> list[dict]:
    conn = _get_db()
    try:
        c = conn.cursor()
        c.execute("SELECT id, spec, price FROM triggers WHERE delivered = 0")
        rows = c.fetchall()

        triggers = []
        for row_id, spec_str, price in rows:
            try:
                spec = json.loads(spec_str)
                triggers.append({'spec': spec, 'price': price})
                c.execute("UPDATE triggers SET delivered = 1 WHERE id = ?",
                          (row_id,))
            except json.JSONDecodeError:
                pass

        conn.commit()
        return triggers
    finally:
        conn.close()


def write_triggered(spec: dict, price: float) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO triggers (watch_id, spec, price, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (spec.get('id', ''), json.dumps(spec), price, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


# ── Daemon main loop ──────────────────────────────────────────────────────────

async def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    # Graceful shutdown on SIGTERM/SIGINT
    import signal
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        from agenthood import AgentHood, load_config
        from agenthood.notify import build_notifiers, notify_all
        from agenthood.watcher import PriceWatcher, extract_quote_price

        # Pre-initialize the SQLite schema before any concurrent tasks
        _get_db().close()

        cfg = load_config()

        async with AgentHood(cfg) as agent:

            async def _watcher_price_fn(ticker: str) -> float:
                raw = await agent._call_tool_direct(
                    "robinhood_get_equity_quotes",
                    {"symbols": [ticker.upper()]},
                )
                price = extract_quote_price(raw)
                if price is not None:
                    return price
                raise ValueError(
                    f"Cannot parse price for {ticker} from: {str(raw)[:200]!r}"
                )

            active_tasks: dict[str, asyncio.Task] = {}
            active_watchers: dict[str, PriceWatcher] = {}

            while True:
                if stop_event.is_set():
                    logger.info("Received shutdown signal, exiting...")
                    break
                if not WATCHES_FILE.exists():
                    break

                watches = await asyncio.to_thread(list_watches)
                if not watches:
                    break

                current_ids = {
                    w.get('id') for w in watches if 'id' in w
                }

                # Cancel removed watches
                for wid in list(active_tasks.keys()):
                    if wid not in current_ids:
                        active_tasks[wid].cancel()
                        del active_tasks[wid]
                        active_watchers.pop(wid, None)

                # Add new watches
                for spec in watches:
                    wid = spec.get('id')
                    if wid and wid not in active_tasks:

                        async def on_trigger(watcher, price, _spec=spec):
                            title = (
                                f"AgentHood: {_spec['ticker']} triggered"
                            )
                            body = (
                                f"{_spec['ticker']} hit ${price:,.2f} "
                                f"(threshold {_spec['direction']} "
                                f"${_spec['threshold']:,.2f})"
                            )
                            await asyncio.to_thread(write_triggered, _spec, price)

                            notifiers = build_notifiers(
                                _spec.get('notify', [])
                            )
                            await notify_all(notifiers, title, body)

                            await asyncio.to_thread(remove_watch, _spec.get('id'))

                        w = PriceWatcher(
                            ticker=spec["ticker"],
                            threshold=spec["threshold"],
                            direction=spec["direction"],
                            action=spec["action"],
                            shares=spec["shares"],
                            interval=spec.get("interval", 30),
                            get_price=_watcher_price_fn,
                            on_trigger=on_trigger,
                        )
                        active_watchers[wid] = w
                        active_tasks[wid] = asyncio.create_task(w.run())

                await asyncio.sleep(60)

            # Cancel all active watcher tasks on exit
            for task in active_tasks.values():
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks.values(), return_exceptions=True)

    finally:
        try:
            PID_FILE.unlink()
        except OSError:
            pass


if __name__ == '__main__':
    asyncio.run(main())
