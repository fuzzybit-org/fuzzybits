"""``fzb serve`` (#203): hands its arguments to ``fuzzybits_server.app.main``.

The in-process tests replace ``fuzzybits_server`` with a stand-in module, so they run whether or
not the server is installed (``scripts/ci/python-pytest.sh`` installs only the engine wheel) and
never start uvicorn. The subprocess tests use whatever is installed and skip otherwise.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import subprocess
import sys
import types
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402
import fuzzybits._cli as _cli  # noqa: E402

MISSING = "fzb serve needs the fuzzybits-server package: pip install 'fuzzybits[server]'"


class _FakeServer:
    """Stand-in for ``fuzzybits_server.app``: records calls to ``main``."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def main(self, argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
        assert argv is not None, "fzb serve must pass argv, never fall back to sys.argv"
        self.calls.append((list(argv), prog))


def _install_fake_server(monkeypatch: pytest.MonkeyPatch, main: Callable[..., None] | None) -> None:
    """Put a ``fuzzybits_server`` package in ``sys.modules``; ``main=None`` makes its app unimportable."""
    package = types.ModuleType("fuzzybits_server")
    package.__spec__ = importlib.machinery.ModuleSpec("fuzzybits_server", None, is_package=True)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "fuzzybits_server", package)
    if main is None:
        monkeypatch.setitem(sys.modules, "fuzzybits_server.app", None)
    else:
        app = types.ModuleType("fuzzybits_server.app")
        app.__spec__ = importlib.machinery.ModuleSpec("fuzzybits_server.app", None)
        app.main = main
        package.app = app
        monkeypatch.setitem(sys.modules, "fuzzybits_server.app", app)
    monkeypatch.setattr(_cli, "_server_version", lambda: fuzzybits.__version__)


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> _FakeServer:
    recorder = _FakeServer()
    _install_fake_server(monkeypatch, recorder.main)
    return recorder


def test_serve_forwards_arguments_unchanged(fake_server: _FakeServer, capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "--host",
        "0.0.0.0",
        "--port",
        "9001",
        "--data-dir",
        "models dir",
        "--cache-dir",
        "~/cache",
        "--backend",
        "omp",
        "--engine",
        "legacy",
        "--no-ui",
        "--open",
        "--not-a-server-flag",
    ]
    assert _cli.main(["serve", *argv]) == 0
    # Order, spelling and unknown flags reach the server's parser untouched: it alone validates.
    assert fake_server.calls == [(argv, "fzb serve")]
    assert capsys.readouterr().err == ""


def test_serve_without_arguments(fake_server: _FakeServer) -> None:
    assert _cli.main(["serve"]) == 0
    assert fake_server.calls == [([], "fzb serve")]


def test_serve_missing_package(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # A None entry makes find_spec report the package as absent, installed or not.
    monkeypatch.setitem(sys.modules, "fuzzybits_server", None)
    assert _cli.main(["serve", "--port", "9001"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == MISSING


def test_serve_broken_server_install(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # The package is found but importing its app fails (for example uvicorn is missing).
    _install_fake_server(monkeypatch, None)
    assert _cli.main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "fzb serve: cannot import fuzzybits_server" in err
    assert MISSING in err


def test_serve_warns_on_version_mismatch(
    fake_server: _FakeServer, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_cli, "_server_version", lambda: "0.0.1")
    assert _cli.main(["serve", "--port", "9001"]) == 0
    assert fake_server.calls == [(["--port", "9001"], "fzb serve")]
    err = capsys.readouterr().err
    assert f"fuzzybits-server 0.0.1 does not match fuzzybits {fuzzybits.__version__}" in err
    assert f"pip install 'fuzzybits[server]=={fuzzybits.__version__}'" in err


def test_serve_older_server_reads_sys_argv(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # A fuzzybits-server from before #203 has `main()` without parameters and reads sys.argv.
    seen: list[list[str]] = []

    def old_main() -> None:
        seen.append(list(sys.argv))

    _install_fake_server(monkeypatch, old_main)
    monkeypatch.setattr(_cli, "_server_version", lambda: "0.18.6")
    before = list(sys.argv)
    assert _cli.main(["serve", "--port", "9001", "--no-ui"]) == 0
    assert seen == [["fzb serve", "--port", "9001", "--no-ui"]]
    assert sys.argv == before
    assert "fuzzybits-server 0.18.6 does not match" in capsys.readouterr().err


def test_serve_no_warning_without_distribution_metadata(
    fake_server: _FakeServer, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A source checkout on PYTHONPATH has no metadata: nothing to compare, no warning.
    monkeypatch.setattr(_cli, "_server_version", lambda: None)
    assert _cli.main(["serve"]) == 0
    assert capsys.readouterr().err == ""


def test_root_usage_lists_serve(capsys: pytest.CaptureFixture[str]) -> None:
    assert _cli.main(["--help"]) == 0
    assert "  serve " in capsys.readouterr().out


def _fzb() -> Path:
    name = "fzb.exe" if sys.platform == "win32" else "fzb"
    for exe in (Path(sys.executable).parent / name, Path(sys.executable).resolve().parent / name):
        if exe.is_file():
            return exe
    pytest.skip(f"fzb console script not installed next to {sys.executable}")


def test_serve_missing_package_subprocess() -> None:
    if importlib.util.find_spec("fuzzybits_server") is not None:
        pytest.skip("fuzzybits-server is installed")
    proc = subprocess.run([str(_fzb()), "serve", "--help"], text=True, capture_output=True, check=False)
    assert proc.returncode == 1
    assert proc.stderr.strip() == MISSING


def test_serve_help_is_the_server_parser() -> None:
    pytest.importorskip("fuzzybits_server.config")
    proc = subprocess.run([str(_fzb()), "serve", "--help"], text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage: fzb serve ")
    for flag in ("--host", "--port", "--data-dir", "--cache-dir", "--backend", "--engine", "--no-ui", "--open"):
        assert flag in proc.stdout
