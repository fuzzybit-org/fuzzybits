"""Ctrl-C stops the server promptly while a calculation is sampling (#203).

Each case starts the real server in its own process group with an unbounded calculation
(``MCSCRAM_NUM_TRIALS=0``, no early stop), waits until the calculation process reports progress,
sends SIGINT, and checks that the server exits promptly, cleanly and without leaving the
calculation process behind.

- ``group`` sends SIGINT to the whole process group, as a terminal's Ctrl-C does: the server and
  the spawned calculation process (and any other child) receive it.
- ``server`` sends it to the server process only (``kill -INT``).

``fzb serve`` runs as ``python -m fuzzybits._cli serve``: the runtime images replace the wheel's
``fzb`` script with the native ``fzb``, which has no ``serve``. It is skipped when the installed
``fuzzybits`` predates ``fzb serve``.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and process groups"),
]

SERVER_SRC = Path(__file__).resolve().parents[1] / "src"
CHINESE = Path(__file__).resolve().parents[3] / "input" / "Aralia" / "chinese.xml"
STARTUP_S = 60.0
# Shutdown must not wait for the calculation: CalculationSupervisor.stop() escalates to SIGKILL
# only after 10 s, so a bound below that fails if SIGTERM no longer ends the calculation.
SHUTDOWN_S = 8.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(port: int, method: str, path: str) -> dict[str, Any]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def _gone(pid: int) -> bool:
    """True once ``pid`` has exited (a zombie waiting to be reaped counts as exited)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] in {"Z", "X"}
        except OSError:
            return True
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return not out.stdout.strip() or out.stdout.strip().startswith("Z")


def _command(entry: str, port: int) -> list[str]:
    flags = ["--host", "127.0.0.1", "--port", str(port), "--backend", "omp", "--no-ui"]
    if entry == "fzb serve":
        cli = importlib.import_module("fuzzybits._cli")
        if not hasattr(cli, "cmd_serve"):
            pytest.skip("the installed fuzzybits has no `fzb serve`")
        return [sys.executable, "-m", "fuzzybits._cli", "serve", *flags]
    # What the `fuzzybits-server` console script runs (`-m fuzzybits_server.app` would import the
    # module twice: as __main__ and for uvicorn's factory).
    return [sys.executable, "-c", "import sys; from fuzzybits_server.app import main; main(sys.argv[1:])", *flags]


@pytest.mark.parametrize("target", ["group", "server"])
@pytest.mark.parametrize("entry", ["fuzzybits-server", "fzb serve"])
def test_sigint_during_calculation_exits_promptly(entry: str, target: str, tmp_path: Path) -> None:
    if not CHINESE.is_file():
        pytest.skip(f"missing {CHINESE}")
    port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(p for p in (str(SERVER_SRC), env.get("PYTHONPATH", "")) if p),
            "ACPP_VISIBILITY_MASK": "omp",
            "FUZZYBITS_BACKEND": "omp",
            "FUZZYBITS_SERVER_CACHE_DIR": str(tmp_path / "cache"),
            "MCSCRAM_INPUT_FILES": str(CHINESE),
            "MCSCRAM_NUM_TRIALS": "0",  # unbounded: the calculation runs until stopped
            "MCSCRAM_EARLY_STOP": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("MCSCRAM_AUTOSTART", None)
    log = tmp_path / "server.log"
    with log.open("wb") as out:
        proc = subprocess.Popen(
            _command(entry, port), env=env, cwd=tmp_path, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
    calc_pid: int | None = None
    try:
        deadline = time.monotonic() + STARTUP_S
        while True:
            assert proc.poll() is None, f"server exited early ({proc.returncode}):\n{log.read_text()}"
            try:
                _request(port, "GET", "/api/v1/healthz")
                break
            except OSError:
                assert time.monotonic() < deadline, f"server did not answer:\n{log.read_text()}"
                time.sleep(0.2)

        started = _request(port, "POST", "/api/v1/start")
        assert started["accepted"], started
        while True:
            status = _request(port, "GET", "/api/v1/status")
            if status["status"] == "running" and status["update_count"] >= 2:
                calc_pid = int(status["pid"])
                break
            assert status["status"] in {"running", "not_started"}, status
            assert time.monotonic() < deadline, f"no progress: {status}\n{log.read_text()}"
            time.sleep(0.2)
        assert not _gone(calc_pid)

        sent = time.monotonic()
        if target == "group":
            os.killpg(proc.pid, signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
        try:
            code = proc.wait(timeout=SHUTDOWN_S + 20)
        except subprocess.TimeoutExpired:
            pytest.fail(f"{entry} still running {SHUTDOWN_S + 20:g} s after SIGINT:\n{log.read_text()}")
        elapsed = time.monotonic() - sent
        text = log.read_text(errors="replace")
        print(f"{entry} ({target}): exit {code} {elapsed:.2f} s after SIGINT")

        assert elapsed < SHUTDOWN_S, f"{entry} took {elapsed:.1f} s to exit after SIGINT:\n{text}"
        assert code == 0, f"exit {code}:\n{text}"
        assert "Traceback" not in text, text
        assert "Application shutdown complete" in text, text
        child_deadline = time.monotonic() + 2.0
        while not _gone(calc_pid) and time.monotonic() < child_deadline:
            time.sleep(0.05)
        assert _gone(calc_pid), f"calculation process {calc_pid} outlived the server"
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        if calc_pid is not None and not _gone(calc_pid):
            os.kill(calc_pid, signal.SIGKILL)
