"""`.update` -- upgrade agenthood from PyPI without leaving the REPL."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from .theme import C_ACCENT, C_DIM, C_ERR, C_GREEN, C_WARN, console

PYPI_URL = "https://pypi.org/pypi/agenthood/json"
PYPI_TIMEOUT = 10


def _installed_version() -> str:
    from .. import __version__
    return __version__


def _is_editable() -> bool:
    """True if agenthood is installed from a source checkout.

    Upgrading an editable install would replace the developer's working copy
    with a release build -- silently undoing whatever they were working on.
    """
    import agenthood
    pkg = Path(agenthood.__file__).resolve().parent
    # An editable install points at the source tree rather than site-packages.
    return "site-packages" not in pkg.parts and "dist-packages" not in pkg.parts


def _latest_version() -> str | None:
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=PYPI_TIMEOUT) as resp:
            return json.load(resp)["info"]["version"]
    except Exception:
        return None


def _parse(v: str) -> tuple:
    """Best-effort version ordering; falls back to string compare."""
    parts = []
    for chunk in v.split("."):
        num = "".join(c for c in chunk if c.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def _in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _externally_managed() -> bool:
    """True if this interpreter refuses installs without an override (PEP 668).

    A virtualenv is never externally managed, but sysconfig reports the *base*
    interpreter's stdlib, so the marker file shows up there too -- check for a
    venv first or the override gets added where it is not needed.
    """
    if _in_virtualenv():
        return False
    import sysconfig
    stdlib = sysconfig.get_paths().get("stdlib", "")
    return bool(stdlib) and Path(stdlib, "EXTERNALLY-MANAGED").exists()


def _pip_command() -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "agenthood"]
    # Homebrew and other distro Pythons refuse to touch site-packages without
    # this; agenthood is a user-facing app, not a system library.
    if _externally_managed():
        cmd.append("--break-system-packages")
    return cmd


def _run_upgrade() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            _pip_command(), capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return False, "pip timed out after 5 minutes"
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, "\n".join(tail[-5:]) or f"pip exited {proc.returncode}"
    return True, (proc.stdout or "").strip()


async def run_update() -> None:
    """Check PyPI and upgrade if a newer release exists."""
    current = _installed_version()
    console.print(f"  [{C_DIM}]installed[/{C_DIM}] [{C_ACCENT}]{current}[/{C_ACCENT}]")

    if _is_editable():
        console.print(
            f"  [{C_WARN}]! editable install detected -- not upgrading[/{C_WARN}]\n"
            f"  [{C_DIM}]agenthood is running from a source checkout. "
            f"Upgrading would overwrite it with a release build.\n"
            f"  Use git to update, or reinstall with 'pip install agenthood'."
            f"[/{C_DIM}]"
        )
        return

    console.print(f"  [{C_DIM}]checking PyPI...[/{C_DIM}]")
    latest = await asyncio.to_thread(_latest_version)
    if latest is None:
        console.print(f"  [{C_ERR}]! could not reach PyPI[/{C_ERR}]")
        return

    if _parse(latest) <= _parse(current):
        console.print(
            f"  [{C_GREEN}]already up to date[/{C_GREEN}] "
            f"[{C_DIM}](latest is {latest})[/{C_DIM}]"
        )
        return

    console.print(
        f"  [{C_ACCENT}]{current}[/{C_ACCENT}] -> "
        f"[{C_GREEN}]{latest}[/{C_GREEN}]  [{C_DIM}]upgrading...[/{C_DIM}]"
    )
    ok, detail = await asyncio.to_thread(_run_upgrade)
    if not ok:
        console.print(f"  [{C_ERR}]! upgrade failed[/{C_ERR}]")
        for line in detail.splitlines():
            console.print(f"  [{C_DIM}]{line}[/{C_DIM}]")
        return

    console.print(
        f"  [{C_GREEN}]upgraded to {latest}[/{C_GREEN}]  "
        f"[{C_DIM}]restart agenthood to load it[/{C_DIM}]"
    )
