"""Tests for :mod:`scripts.analysis.terminal_capture`.

Every typescript is synthesised in the test, so nothing here needs a built
binary, a PTY, or a container.  The one exception is the ``record`` test, which
skips itself when ``script(1)`` is missing or is not the util-linux build.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "scripts" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import terminal_capture as tc  # noqa: E402

CSI = "\x1b["

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def write_capture(
    directory: Path,
    name: str,
    body: bytes,
    timing: list[tuple[float, int]] | None = None,
    banner: bool = True,
) -> Path:
    """Write a synthetic ``script(1)`` typescript (and timing) and return its path."""
    payload = body
    if banner:
        payload = b'Script started on 2026-01-01 00:00:00+00:00 [COMMAND="demo"]\n' + payload
    payload += b'\nScript done on 2026-01-01 00:00:01+00:00 [COMMAND_EXIT_CODE="0"]\n'
    path = directory / f"{name}.typescript"
    path.write_bytes(payload)
    if timing is not None:
        (directory / f"{name}.timing").write_text(
            "".join(f"{delay:.6f} {count}\n" for delay, count in timing), encoding="utf-8"
        )
    return path


def frame(*lines: str, up: int | None = None) -> bytes:
    """Build one block refresh: ``ESC[<n>A`` then the lines, CRLF-terminated."""
    count = len(lines) if up is None else up
    body = "".join(
        f"\x1b[37m\r{line}\x1b[0m\r\n" for line in lines
    )
    return f"{CSI}{count}A".encode() + body.encode()


def bar(width: int, filled: int, *, tail: str = "") -> str:
    """Return a progress-bar line exactly ``width`` cells wide (ASCII)."""
    inner = "#" * filled + "-" * (width - filled)
    return f"[{inner}]{tail}"


# --------------------------------------------------------------------------
# Width measurement
# --------------------------------------------------------------------------


def test_display_width_counts_cells_not_codepoints() -> None:
    assert tc.display_width("abc") == 3
    # CJK: two cells for one code point.
    assert tc.display_width("漢字") == 4
    # Ambiguous-width punctuation is one cell in a non-CJK terminal, matching wcwidth.
    assert tc.display_width("■") == 1
    assert tc.display_width("ε") == 1
    # A combining mark adds no cell.
    assert tc.display_width("e\u0301") == 1
    # The naive answer would be 2 and 3 respectively.
    assert len("漢字") == 2
    assert len("e\u0301") == 2


def test_max_line_width_sees_wide_glyphs(tmp_path: Path) -> None:
    """A CJK line is wider in cells than its ``len()`` suggests."""
    narrow = "ab"  # 2 cells, 2 code points
    wide = "漢漢"  # 4 cells, 2 code points
    body = frame(bar(20, 10) + narrow, bar(20, 10) + wide)
    path = write_capture(tmp_path, "wide", body)

    transcript = tc.load_typescript(path)
    refreshes = tc.split_refreshes(transcript, cols=80)

    assert tc.max_line_width(refreshes) == tc.display_width(bar(20, 10) + wide)
    assert tc.max_line_width(refreshes) == 26
    # If widths were computed with len(), both lines would measure 24 and the
    # wide glyphs would be under-counted by two cells.
    assert len(bar(20, 10) + wide) == 24


# --------------------------------------------------------------------------
# Frame that fits
# --------------------------------------------------------------------------


def test_frame_that_fits_has_zero_drift(tmp_path: Path) -> None:
    body = b"".join(frame(bar(60, i), bar(60, i + 1)) for i in range(0, 8, 2))
    path = write_capture(tmp_path, "fits", body)

    transcript = tc.load_typescript(path)
    refreshes = tc.split_refreshes(transcript, cols=80)

    assert [r.kind for r in refreshes] == ["block"] * 4
    assert [r.up for r in refreshes] == [2, 2, 2, 2]
    assert [r.physical_rows for r in refreshes] == [2, 2, 2, 2]
    assert tc.drift_per_refresh(refreshes, 80) == [0, 0, 0, 0]
    assert tc.max_line_width(refreshes) == 62


# --------------------------------------------------------------------------
# Frame that wraps
# --------------------------------------------------------------------------


def test_wrapping_frame_reports_expected_drift(tmp_path: Path) -> None:
    """Three 100-cell lines in an 80-column terminal cost six rows, not three."""
    wide = bar(100, 50)
    assert tc.display_width(wide) == 102
    body = b"".join(frame(wide, wide, wide) for _ in range(4))
    path = write_capture(tmp_path, "wraps", body)

    transcript = tc.load_typescript(path)
    refreshes = tc.split_refreshes(transcript, cols=80)

    # Each line wraps exactly once: 102 cells over 80 -> 2 rows.
    assert [r.physical_rows for r in refreshes] == [6, 6, 6, 6]
    assert [r.up for r in refreshes] == [3, 3, 3, 3]
    assert tc.drift_per_refresh(refreshes, 80) == [3, 3, 3, 3]

    report = tc.analyse(transcript, cols=80, rows=24)
    assert report.drift_rows_per_refresh == 3
    assert report.width_exceeded is True
    assert report.max_width == 102
    assert report.total_drift == 12
    assert report.requested_rows == 12
    assert report.physical_rows == 24


def test_growing_block_drifts_without_wrapping(tmp_path: Path) -> None:
    """Adding a bar each frame costs one row when ``move_up`` counts the old block.

    Every line fits, so this drift is structural, not a wrapping artefact; it is
    reported separately as ``startup_drift`` while the steady-state figure stays
    zero.
    """
    frames = []
    for count in range(1, 5):
        lines = [bar(20, count)] * count
        # move_up uses the *previous* frame's bar count.
        frames.append(frame(*lines, up=count - 1))
    # Then hold the block at full size so the tail is genuinely steady.
    frames.extend(frame(*([bar(20, 4)] * 4), up=4) for _ in range(6))
    path = write_capture(tmp_path, "growing", b"".join(frames))

    report = tc.analyse(tc.load_typescript(path), cols=80)
    assert report.width_exceeded is False
    assert report.startup_drift == 4
    assert report.drift_rows_per_refresh == 0
    assert "startup drift" in report.text()


def test_exactly_full_width_line_does_not_wrap(tmp_path: Path) -> None:
    """An 80-cell line in an 80-column terminal occupies one row."""
    line = bar(80, 40)
    assert tc.display_width(line) == 82
    body = frame(line)
    path = write_capture(tmp_path, "exact", body)

    refreshes = tc.split_refreshes(tc.load_typescript(path), cols=82)
    assert [r.physical_rows for r in refreshes] == [1]
    assert tc.drift_per_refresh(refreshes, 82) == [0]


def test_final_frame_is_excluded_from_drift_totals(tmp_path: Path) -> None:
    """Output printed after the display finished must not inflate the drift.

    In practice the last cursor-up frame carries the whole report, so it is far
    taller than a display frame.  ``analyse`` detects that and reports it
    separately instead of folding it into the drift totals.
    """
    display = b"".join(frame(bar(90, i)) for i in range(0, 4))
    # The last cursor-up frame swallows the report printed on completion.
    report_lines = [f"  <row n=\"{n}\"/>".ljust(30) for n in range(20)]
    trailing = f"{CSI}1A".encode() + "".join(
        f"\r{line}\r\n" for line in report_lines
    ).encode()
    path = write_capture(tmp_path, "trailing", display + trailing)

    report = tc.analyse(tc.load_typescript(path), cols=80)
    assert report.refresh_count == 5
    # Four display frames of 92 cells -> 2 rows each, one cursor-up each.
    assert report.requested_rows == 4
    assert report.physical_rows == 8
    assert report.total_drift == 4
    assert report.final_frame_rows == 20
    assert "final frame" in report.text()


# --------------------------------------------------------------------------
# Single-line CR redraw
# --------------------------------------------------------------------------


def test_cr_redraw_single_line(tmp_path: Path) -> None:
    """A bare CR rewrites the row; the widest write decides the row width."""
    body = (
        b"\rprogress   0%"
        b"\rprogress  50%"
        b"\rprogress 100%\r\n"
        b"done\r\n"
    )
    path = write_capture(tmp_path, "cr", body)

    transcript = tc.load_typescript(path)
    refreshes = tc.split_refreshes(transcript, cols=80)

    assert len(refreshes) == 3
    assert {r.kind for r in refreshes} == {"line"}
    assert [r.up for r in refreshes] == [0, 0, 0]
    # No cursor-up at all, so a line redraw never drifts.
    assert tc.drift_per_refresh(refreshes, 80) == [0, 0, 0]
    assert tc.max_line_width(refreshes) == len("progress 100%")
    assert refreshes[-1].lines[0].text == "progress 100%"


# --------------------------------------------------------------------------
# Cursor show/hide balance
# --------------------------------------------------------------------------


def test_cursor_balance_detects_single_restore(tmp_path: Path) -> None:
    body = b"\x1b[?25l" + frame(bar(20, 5)) + b"\x1b[?25h"
    path = write_capture(tmp_path, "balance", body)

    report = tc.analyse(tc.load_typescript(path), cols=80)
    assert report.cursor_hide == 1
    assert report.cursor_show == 1
    assert report.cursor_balanced is True
    assert "balanced" in report.text()


def test_cursor_balance_detects_double_restore(tmp_path: Path) -> None:
    """Restoring twice is the real-world bug: hide once, show at exit twice."""
    body = b"\x1b[?25l" + frame(bar(20, 5)) + b"\x1b[?25h\x1b[?25h"
    path = write_capture(tmp_path, "unbalanced", body)

    report = tc.analyse(tc.load_typescript(path), cols=80)
    assert report.cursor_hide == 1
    assert report.cursor_show == 2
    assert report.cursor_balanced is False
    assert "UNBALANCED" in report.text()


# --------------------------------------------------------------------------
# Timing / cadence
# --------------------------------------------------------------------------


def test_cadence_groups_bursts(tmp_path: Path) -> None:
    frames = b"".join(frame(bar(20, i)) for i in range(6))
    per_frame = len(frames) // 6
    timing = [(0.5, per_frame)] * 6  # uniform 0.5 s gaps -> one burst per frame
    path = write_capture(tmp_path, "cadence", frames, timing=timing)

    report = tc.analyse(tc.load_typescript(path), cols=80)
    assert report.cadence["frames_per_s"] == pytest.approx(2.0, abs=0.2)
    assert report.cadence["bursts"] == 6
    assert report.cadence["max_gap_s"] == pytest.approx(0.5, abs=0.01)


def test_missing_timing_yields_zero_duration(tmp_path: Path) -> None:
    path = write_capture(tmp_path, "notiming", frame(bar(20, 5)))
    transcript = tc.load_typescript(path)
    assert transcript.timing == ()
    assert transcript.duration == 0.0
    assert transcript.time_at(0) == 0.0


def test_explicit_timing_path_is_honoured(tmp_path: Path) -> None:
    path = write_capture(tmp_path, "explicit", frame(bar(20, 5)))
    sidecar = tmp_path / "elsewhere.timing"
    sidecar.write_text("1.5 999\n", encoding="utf-8")
    transcript = tc.load_typescript(path, sidecar)
    assert transcript.timing == ((1.5, 999),)
    assert transcript.duration == 1.5


# --------------------------------------------------------------------------
# Screen reconstruction
# --------------------------------------------------------------------------


def test_screens_at_renders_expected_text(tmp_path: Path) -> None:
    pytest.importorskip("pyte")
    body = b"\x1b[?25l" + frame(bar(20, 5)) + frame(bar(20, 15)) + b"\x1b[?25h"
    path = write_capture(tmp_path, "screen", body)

    snapshots = tc.screens_at(tc.load_typescript(path), cols=40, rows=6, at=[1.0])

    assert len(snapshots) == 1
    lines = [line for line in snapshots[0].lines if line.strip()]
    assert lines == [bar(20, 15)]
    assert snapshots[0].width == 22


def test_screens_at_before_pattern_stops_just_before_output(tmp_path: Path) -> None:
    pytest.importorskip("pyte")
    body = (
        b"\x1b[?25l"
        + frame("working")
        + b"<report>\r\n<p>1</p>\r\n</report>\r\n"
        + b"\x1b[?25h"
    )
    path = write_capture(tmp_path, "before", body)

    snapshots = tc.screens_at(
        tc.load_typescript(path), cols=40, rows=6, before_pattern="<report>"
    )

    assert [snap.name for snap in snapshots] == ["before_report"]
    assert "working" in snapshots[0].text()
    assert "<report>" not in snapshots[0].text()


def test_screens_at_rejects_unmatched_pattern(tmp_path: Path) -> None:
    pytest.importorskip("pyte")
    path = write_capture(tmp_path, "nomatch", frame("working"))
    with pytest.raises(tc.CaptureError, match="matched nothing"):
        tc.screens_at(tc.load_typescript(path), cols=40, rows=6, before_pattern="absent")


def test_screens_at_rejects_invalid_regex(tmp_path: Path) -> None:
    pytest.importorskip("pyte")
    path = write_capture(tmp_path, "badre", frame("working"))
    with pytest.raises(tc.CaptureError, match="invalid --before-pattern"):
        tc.screens_at(tc.load_typescript(path), cols=40, rows=6, before_pattern="(")


def test_screens_at_rejects_bad_geometry(tmp_path: Path) -> None:
    pytest.importorskip("pyte")
    path = write_capture(tmp_path, "geometry", frame("working"))
    with pytest.raises(tc.CaptureError, match="geometry must be positive"):
        tc.screens_at(tc.load_typescript(path), cols=0, rows=10, at=[1.0])


# --------------------------------------------------------------------------
# PNG rendering (optional dependency)
# --------------------------------------------------------------------------


def _has_pillow_and_font() -> bool:
    try:
        import PIL  # noqa: F401, PLC0415 - dependency probe
    except ImportError:
        return False
    return any(Path(candidate).is_file() for candidate in tc.FONT_CANDIDATES)


@pytest.mark.skipif(not _has_pillow_and_font(), reason="Pillow or a monospaced font is missing")
def test_render_png_writes_a_file(tmp_path: Path) -> None:
    pytest.importorskip("pyte")
    path = write_capture(tmp_path, "png", b"\x1b[?25l" + frame(bar(20, 5)) + b"\x1b[?25h")
    snapshot = tc.screens_at(tc.load_typescript(path), cols=40, rows=4, at=[1.0])[0]

    out = tc.render_png(snapshot, tmp_path / "frame.png")

    assert out.is_file()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_png_without_pillow_raises_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyte")
    path = write_capture(tmp_path, "nopillow", frame("x"))
    snapshot = tc.screens_at(tc.load_typescript(path), cols=40, rows=4, at=[1.0])[0]

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] == "PIL":
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(tc.CaptureError, match="Pillow"):
        tc.render_png(snapshot, tmp_path / "frame.png")


# --------------------------------------------------------------------------
# Recording (needs util-linux script(1))
# --------------------------------------------------------------------------


requires_script = pytest.mark.skipif(
    not tc.script_supports_timing(),
    reason="util-linux script(1) with --timing is not available",
)


@requires_script
def test_record_captures_a_real_pty(tmp_path: Path) -> None:
    python = shutil.which("python3") or sys.executable
    result = tc.record(
        "pty",
        [python, "-c", "print('hello from a pty')"],
        out_dir=tmp_path,
        cols=60,
        rows=10,
    )

    assert result.exit_code == 0
    assert result.typescript.is_file()
    transcript = tc.load_typescript(result.typescript)
    assert b"hello from a pty" in transcript.body


@requires_script
def test_record_sets_requested_environment(tmp_path: Path) -> None:
    python = shutil.which("python3") or sys.executable
    result = tc.record(
        "envprobe",
        [python, "-c", "import os; print('COLS=' + os.environ.get('TC_PROBE', ''))"],
        out_dir=tmp_path,
        cols=40,
        rows=8,
        env={"TC_PROBE": "sentinel"},
    )
    transcript = tc.load_typescript(result.typescript)
    assert b"COLS=sentinel" in transcript.body


def test_record_rejects_empty_command(tmp_path: Path) -> None:
    with pytest.raises(tc.CaptureError, match="needs a command"):
        tc.record("empty", [], out_dir=tmp_path)


def test_record_rejects_bad_geometry(tmp_path: Path) -> None:
    with pytest.raises(tc.CaptureError, match="geometry must be positive"):
        tc.record("geom", ["true"], out_dir=tmp_path, cols=0)


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


def test_cli_analyse_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    body = b"".join(frame(bar(100, i)) for i in range(3))
    path = write_capture(tmp_path, "cli", body)
    out = tmp_path / "report.json"

    code = tc.main(
        [
            "analyse",
            "--typescript",
            str(path),
            "--cols",
            "80",
            "--rows",
            "24",
            "--json",
            str(out),
        ]
    )

    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    # Three frames of 102 cells -> two rows each against up=1, so one row of
    # drift per frame.
    assert payload["drift"]["total_rows"] == 3
    assert payload["width"]["exceeded"] is True
    assert "up=1" in capsys.readouterr().out


def test_cli_render_writes_text_only(tmp_path: Path) -> None:
    pytest.importorskip("pyte")
    body = b"\x1b[?25l" + frame("hello") + b"\x1b[?25h"
    path = write_capture(tmp_path, "clirender", body)
    out = tmp_path / "frames"

    code = tc.main(
        [
            "render",
            "--typescript",
            str(path),
            "--cols",
            "40",
            "--rows",
            "4",
            "--at",
            "1.0",
            "--out",
            str(out),
            "--no-png",
        ]
    )

    assert code == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == ["clirender_t100.txt"]
    assert "hello" in (out / written[0]).read_text(encoding="utf-8")


def test_cli_reports_missing_typescript(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = tc.main(["analyse", "--typescript", str(tmp_path / "nope.typescript"), "--cols", "80"])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_cli_record_requires_a_command(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        tc.main(["record", "--name", "x", "--out", str(tmp_path)])
