"""PTY golden tests for the progress line (#207).

The renderer is shared C++ code, so these drive the Python ``fzb`` under a pseudo-terminal and
replay the captured bytes through :mod:`pyte`. That checks the three properties the legacy display
got wrong (#205): no frame wraps, redraws never drift, and the final screen is only the summary.
"""

from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")
pyte = pytest.importorskip("pyte")

import fuzzybits  # noqa: E402

# CSI sequences the renderer is allowed to emit, plus erase-line / cursor motion.
ESCAPE = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")

CURSOR_HIDE = b"\x1b[?25l"
CURSOR_SHOW = b"\x1b[?25h"


def _fzb() -> Path:
    """The installed `fzb` console script (installed wheel, else one on PATH)."""
    name = "fzb.exe" if sys.platform == "win32" else "fzb"
    for exe in (Path(sys.executable).parent / name, Path(sys.executable).resolve().parent / name):
        if exe.is_file():
            return exe
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    pytest.skip(f"fzb console script not installed next to {sys.executable} nor on PATH")


def _compile_chinese(tmp_path: Path) -> Path:
    """A model large enough to take more than one coalescing interval on the OMP backend."""
    pcfb = tmp_path / "progress.pcfb"
    env = dict(os.environ, ACPP_VISIBILITY_MASK="omp", FUZZYBITS_BACKEND="omp")
    done = subprocess.run(
        [str(_fzb()), "compile", "input/Aralia/chinese.xml", "-o", str(pcfb), "--no-progress"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        pytest.skip(f"fzb compile failed: {done.stderr[-400:]}")
    return pcfb


def _run_in_pty(
    argv: list[str],
    cwd: Path,
    cols: int,
    rows: int = 24,
    *,
    stdout_to_pipe: bool = True,
    interrupt_after: float | None = None,
    interrupt_on_output: bytes | None = None,
    interrupt_timeout: float = 90.0,
) -> tuple[bytes, int]:
    """Run `argv` with stderr on a pseudo-terminal of `cols` x `rows`; return its bytes.

    Progress goes to stderr only, so by default stdout is a pipe and never mixed into the captured
    stream. Otherwise the result line would look like a progress frame and could be mistaken for a
    wrap.

    `interrupt_after` sends SIGINT after a wall-clock delay; `interrupt_on_output` instead waits for
    the first frame (`CURSOR_HIDE`) before signalling. The latter is deterministic: a fixed delay
    can land while the child is still importing, where Python's own handler turns the signal into a
    traceback and an exit code of 1.
    """
    master, slave = pty.openpty()
    # The window size must be set before the child starts: the renderer re-reads it per frame, but
    # the first frame is what proves there is no wrap.
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    env = dict(
        os.environ,
        ACPP_VISIBILITY_MASK="omp",
        FUZZYBITS_BACKEND="omp",
        TERM="xterm-256color",
        LC_ALL="C.UTF-8",
        LANG="C.UTF-8",
        COLUMNS=str(cols),
        LINES=str(rows),
    )
    env.pop("FUZZYBITS_PROGRESS", None)
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=slave,
        stdout=subprocess.PIPE if stdout_to_pipe else slave,
        stderr=slave,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave)
    chunks: list[bytes] = []
    deadline = None if interrupt_after is None else time.time() + interrupt_after
    give_up = None if interrupt_on_output is None else time.time() + interrupt_timeout
    interrupted = False
    try:
        while True:
            now = time.time()
            due = deadline is not None and now >= deadline
            if interrupt_on_output is not None and not interrupted:
                due = due or (interrupt_on_output in b"".join(chunks))
                if give_up is not None and now >= give_up and not due:
                    proc.kill()
                    pytest.fail(f"no {interrupt_on_output!r} within {interrupt_timeout} s")
            if due and not interrupted:
                # Signal the process group so the child sees SIGINT exactly as a Ctrl-C would.
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                interrupted = True
            ready, _, _ = select.select([master], [], [], 60.0)
            if not ready:
                proc.kill()
                pytest.fail("fzb run produced no output for 60 s")
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
            if proc.poll() is not None and not select.select([master], [], [], 0.1)[0]:
                break
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait(timeout=30)
    return b"".join(chunks), proc.returncode


def _screen(raw: bytes, cols: int, rows: int = 24) -> pyte.Screen:
    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)
    stream.feed(raw)
    return screen


def _non_blank(screen: pyte.Screen) -> list[str]:
    return [line.rstrip() for line in screen.display if line.strip()]


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


@pytest.fixture(scope="module")
def chinese_pcfb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile_chinese(tmp_path_factory.mktemp("progress-line"))


@pytest.mark.parametrize("cols", [40, 80, 120])
def test_python_fzb_run_pty_golden(chinese_pcfb: Path, tmp_path: Path, cols: int):
    raw, _ = _run_in_pty(
        [str(_fzb()), "run", str(chinese_pcfb), "--trials", "4e8", "--backend", "omp"],
        cwd=Path.cwd(),
        cols=cols,
    )
    assert raw, "no bytes captured"
    assert CURSOR_HIDE in raw, "the cursor was never hidden, so no frame was drawn"
    # Exactly one cursor-show: the legacy display sent it twice (#205).
    assert raw.count(CURSOR_SHOW) == 1, f"cursor shown {raw.count(CURSOR_SHOW)} times"

    # No frame may be wider than the terminal: pyte wraps at `cols`, so any wrapped frame shows up
    # as a redraw block that grew a row. The final screen is the strongest check: it must hold the
    # summary and the result line only.
    screen = _screen(raw, cols)
    lines = _non_blank(screen)
    assert lines, "final screen is blank"
    # stdout is a pipe, so stderr holds only the summary: the live block is gone and nothing else
    # was left behind.
    assert any("sampled" in line for line in lines), f"no summary on screen: {lines}"
    assert len(lines) == 1, f"the block was left behind: {lines}"


def test_python_fzb_run_pty_no_frame_exceeds_width(chinese_pcfb: Path, tmp_path: Path):
    """Every drawn line is at most `cols` cells: pyte would wrap anything longer."""
    cols = 40
    raw, _ = _run_in_pty(
        [str(_fzb()), "run", str(chinese_pcfb), "--trials", "2e8", "--backend", "omp"],
        cwd=Path.cwd(),
        cols=cols,
    )
    # Strip escape sequences and check the raw text lines the renderer emitted. A wrapped frame
    # would put a continuation line at column 0 that no `\\r` started.
    text = ESCAPE.sub(b"", raw).decode("utf-8", "replace")
    offenders = [
        line
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line and len(line) > cols
    ]
    assert not offenders, f"lines wider than {cols} cells: {offenders[:3]}"


@pytest.mark.parametrize("args", [["--no-progress"], ["--quiet"], ["-q"]])
def test_flags_suppress_progress(chinese_pcfb: Path, args: list[str]):
    raw, _ = _run_in_pty(
        [str(_fzb()), "run", str(chinese_pcfb), "--trials", "2e7", "--backend", "omp", *args],
        cwd=Path.cwd(),
        cols=80,
    )
    assert b"\x1b[" not in raw, f"escape sequences with {args}: {raw[:200]!r}"
    assert CURSOR_HIDE not in raw
    assert b"sampling" not in raw


def test_non_tty_stderr_has_no_progress(chinese_pcfb: Path):
    """A pipe is not a terminal, so nothing animated may reach stderr."""
    env = dict(os.environ, ACPP_VISIBILITY_MASK="omp", FUZZYBITS_BACKEND="omp")
    env.pop("FUZZYBITS_PROGRESS", None)
    done = subprocess.run(
        [str(_fzb()), "run", str(chinese_pcfb), "--trials", "2e7", "--backend", "omp"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == "", f"unexpected stderr chatter: {done.stderr!r}"
    assert done.stdout.strip(), "no result printed"


def test_env_disables_progress(chinese_pcfb: Path):
    env = dict(os.environ, ACPP_VISIBILITY_MASK="omp", FUZZYBITS_BACKEND="omp", FUZZYBITS_PROGRESS="0")
    done = subprocess.run(
        [str(_fzb()), "run", str(chinese_pcfb), "--trials", "2e7", "--backend", "omp"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "\x1b[" not in done.stderr


def _native_fzb() -> Path | None:
    """The in-tree native `fzb`, when this checkout was built (it is not in a wheel venv)."""
    repo = Path(__file__).resolve().parents[2]
    name = "fzb.exe" if sys.platform == "win32" else "fzb"
    for candidate in sorted((repo / "build").glob(f"*/targets/fzb/{name}")):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or _native_fzb() is None,
    reason="SIGINT needs the native fzb (POSIX signal handlers) and a built checkout",
)
def test_ctrl_c_restores_terminal_and_prints_partial_summary(chinese_pcfb: Path):
    """Native CLI: SIGINT shows the cursor once and leaves a partial summary, not a stale block.

    Here the renderer installs its own SIGINT/SIGTERM handler (the process still has the default
    disposition), so this exercises the async-signal-safe path.
    """
    native = _native_fzb()
    assert native is not None
    raw, code = _run_in_pty(
        [str(native), "run", str(chinese_pcfb), "--trials", "4e9", "--backend", "omp"],
        cwd=Path.cwd(),
        cols=80,
        # Wait for the first frame: a fixed delay can land during import, where the signal takes a
        # different path entirely.
        interrupt_on_output=CURSOR_HIDE,
    )
    # The handler restores the cursor, prints the partial summary, then re-raises with SIG_DFL
    # (`SA_RESETHAND`), so the process dies by the signal: `-SIGINT` here, 130 in a shell.
    assert code in {128 + int(signal.SIGINT), -int(signal.SIGINT)}, f"unexpected exit {code}"
    assert CURSOR_HIDE in raw, "no frame was drawn before SIGINT"
    assert raw.count(CURSOR_SHOW) == 1, f"cursor shown {raw.count(CURSOR_SHOW)} times"
    assert b"interrupted after" in raw, "no partial summary after SIGINT"


def test_python_fzb_ctrl_c_is_interruptible(chinese_pcfb: Path):
    """Python CLI: SIGINT stops the run through `PyErr_CheckSignals` and exits 130 cleanly.

    CPython owns SIGINT and the run releases the GIL, so the binding polls the interpreter from a
    sink; without it the run could not be interrupted at all. The exit code is 130 and there is no
    traceback, because `_cli.main` maps KeyboardInterrupt.
    """
    raw, code = _run_in_pty(
        [str(_fzb()), "run", str(chinese_pcfb), "--trials", "4e9", "--backend", "omp"],
        cwd=Path.cwd(),
        cols=80,
        interrupt_on_output=CURSOR_HIDE,
    )
    tail = raw[-400:].decode("utf-8", "replace")
    assert code == 128 + int(signal.SIGINT), f"expected exit 130, got {code}; tail={tail!r}"
    assert CURSOR_HIDE in raw, "no frame was drawn before SIGINT"
    assert raw.count(CURSOR_SHOW) == 1, f"cursor shown {raw.count(CURSOR_SHOW)} times"
    assert b"interrupted after" in raw, f"no partial summary after SIGINT; tail={tail!r}"
    assert b"Traceback" not in raw, "KeyboardInterrupt must not print a traceback"
