#!/usr/bin/env python3
"""Capture and analyse the terminal output of interactive CLI tools.

The tool exists to make terminal progress displays measurable instead of
anecdotal.  It records a command inside a real PTY at a fixed size, rebuilds
the screen at chosen instants, and derives quantitative facts from the raw
byte stream: how wide each emitted line really is, how many rows a redrawn
block consumes, and by how much the cursor bookkeeping drifts.

Three subcommands share one implementation:

``record``
    Run a command under ``script(1)`` in a PTY of exactly ``--cols`` x
    ``--rows`` and write ``<name>.typescript`` plus ``<name>.timing``.
``render``
    Replay a recording through :mod:`pyte` and dump the screen at selected
    fractions of the recording (or immediately before a regex match) as
    ``.txt`` and, when Pillow is installed, ``.png``.
``analyse``
    Report, per refresh, the measured line widths in cells against the
    terminal width, how many rows each refresh physically consumed, the
    cursor-up accounting (requested versus physical rows, i.e. drift), the
    refresh cadence and the cursor show/hide balance.  Emits JSON and a short
    text summary.

The library functions :func:`load_typescript`, :func:`screens_at`,
:func:`max_line_width` and :func:`drift_per_refresh` are the supported entry
points for tests; everything else is an implementation detail.

Method
------
A "refresh" is one redraw of a block of lines.  Two idioms are recognised:

* **block refresh** -- ``ESC[<n>A`` (cursor up *n*) followed by the rewritten
  lines.  This is what ``indicators::DynamicProgress`` and similar libraries
  do.  The block is re-emitted in full each time.
* **line refresh** -- a bare ``CR`` that returns to column 0 without a
  following ``LF``, used by single-line spinners and counters.

Both are measured the same way.  For every physical line of the payload the
*emitted* width is the widest ``CR``-delimited segment of that line, because a
``CR`` returns to column 0 and the row is therefore as wide as its widest
write.  Escape sequences are removed first, so only printable cells count.
The number of rows a line occupies is ``ceil(width / cols)`` (minimum one).
A refresh whose payload occupies more rows than the ``ESC[<n>A`` it opened
with pushes the whole block down by the difference, which is the *drift*.

Locale caveat
-------------
Ambiguous-width glyphs (``\u25a0`` BLACK SQUARE, Greek letters, box drawing)
are one cell wide in a non-CJK terminal and two in a CJK one.  Both
:mod:`wcwidth` and :mod:`pyte` resolve this from the process locale, and older
``pyte`` builds return ``-1`` for any non-ASCII glyph under a POSIX ctype,
which makes ``Screen.draw()`` silently drop the rest of the line.  Set a UTF-8
ctype (``LC_ALL=C.UTF-8``) before calling :func:`screens_at`; the helper
:func:`ensure_utf8_locale` does it for you and the ``record`` subcommand sets
it in the child environment by default.  Widths are always computed with
:mod:`wcwidth`, never from ``len()``, so a wide character counts as two cells.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "CaptureError",
    "Line",
    "Refresh",
    "Report",
    "Snapshot",
    "Transcript",
    "analyse",
    "display_width",
    "drift_per_refresh",
    "ensure_utf8_locale",
    "load_typescript",
    "main",
    "max_line_width",
    "record",
    "render_png",
    "screens_at",
    "split_refreshes",
]

# --------------------------------------------------------------------------
# Escapes
# --------------------------------------------------------------------------

#: ANSI/ECMA-48 CSI sequence: parameters, intermediates, final byte.
CSI_RE = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]")
#: Cursor up *n* rows -- the block-redraw idiom.
CURSOR_UP_RE = re.compile(rb"\x1b\[(\d*)A")
CURSOR_HIDE_RE = re.compile(rb"\x1b\[\?25l")
CURSOR_SHOW_RE = re.compile(rb"\x1b\[\?25h")
#: A CR that restarts the current row without advancing it (single-line redraw).
_LINE_CR_RE = re.compile(rb"\r(?=[^\n])")

_SCRIPT_HEADER_RE = re.compile(rb"^Script started on [^\n]*\n")
_SCRIPT_FOOTER_RE = re.compile(rb"\nScript done on [^\n]*\n?$")

#: Default environment for :func:`record`; override with ``--env``.
DEFAULT_ENV = {
    "TERM": "xterm-256color",
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8",
}

#: Fonts tried in order when rendering PNG frames.  DejaVu Sans Mono is the
#: canonical choice (it is what Linux CI has); the rest keep macOS usable.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
    "/Library/Fonts/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Supplemental/Menlo.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
)

#: xterm base colours used when rasterising frames.
_BASE_COLOURS = {
    "black": "#3b3b3b",
    "red": "#cc5555",
    "green": "#55cc55",
    "brown": "#cdcd55",
    "yellow": "#e5e510",
    "blue": "#5555cc",
    "magenta": "#cc55cc",
    "cyan": "#55cccc",
    "white": "#e5e5e5",
    "default": "#d0d0d0",
}

_DEFAULT_FG = _BASE_COLOURS["default"]
_BACKGROUND = "#141414"

#: How many times taller than the tallest display frame the last frame must be
#: before it is treated as trailing output (a report) rather than a display frame.
TRAILING_FACTOR = 3


class CaptureError(RuntimeError):
    """Raised when a recording or a required dependency is unavailable."""


# --------------------------------------------------------------------------
# Locale and width
# --------------------------------------------------------------------------


def ensure_utf8_locale() -> str:
    """Set a UTF-8 ctype for the process and return the locale that stuck.

    ``pyte`` and ``wcwidth`` both consult the ctype when deciding the width of
    non-ASCII glyphs.  Under ``LC_ALL=C`` many builds report ``-1``, which makes
    ``pyte`` drop the remainder of the line.  This is idempotent and cheap.
    """
    import locale

    for candidate in ("C.UTF-8", "C.utf8", "en_US.UTF-8", "UTF-8"):
        try:
            locale.setlocale(locale.LC_ALL, candidate)
            return locale.setlocale(locale.LC_ALL, None)
        except locale.Error:
            continue
    return locale.setlocale(locale.LC_ALL, None)


def _char_width(char: str) -> int:
    """Return the cell width of one character (0, 1 or 2)."""
    try:
        from wcwidth import wcwidth  # noqa: PLC0415 - optional dependency
    except ImportError:  # pragma: no cover - exercised only without wcwidth
        if unicodedata.combining(char):
            return 0
        return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    width = wcwidth(char)
    return max(width, 0)


def display_width(text: str) -> int:
    """Return the number of terminal cells ``text`` occupies.

    Wide (CJK) characters count as two cells and combining marks as zero.
    Never use ``len()`` for this: it is wrong for both.
    """
    return sum(_char_width(char) for char in text)


def _rows_for(cells: int, cols: int) -> int:
    """Return the physical rows a line of ``cells`` occupies in ``cols``."""
    if cells <= 0:
        return 1
    return max(1, math.ceil(cells / cols))


# --------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Transcript:
    """A ``script(1)`` recording with its timing sidecar.

    ``raw`` is the file as written on disk, including ``script``'s own
    "Script started/done" banner lines.  ``body_start`` and ``body_end`` bound
    the part that the child process actually produced.  All byte offsets
    returned by this module are absolute offsets into ``raw`` so that timing
    records and frame positions stay consistent without realigning anything.

    ``timing`` holds ``(delay_seconds, byte_count)`` pairs in file order.  Some
    ``script`` builds count only the child's bytes, others count the whole file;
    ``timing_base`` records which, and :meth:`time_at` accounts for it.
    """

    path: Path
    raw: bytes
    timing: tuple[tuple[float, int], ...] = ()
    body_start: int = 0
    body_end: int = 0
    timing_base: int = 0

    @property
    def name(self) -> str:
        """Stem of the recording file, used to name output frames."""
        return self.path.stem

    @property
    def body(self) -> bytes:
        """Child output with the ``script`` banner removed."""
        return self.raw[self.body_start:self.body_end]

    @property
    def duration(self) -> float:
        """Total recorded duration in seconds, or ``0.0`` without timing."""
        return sum(delay for delay, _ in self.timing)

    def time_at(self, offset: int) -> float:
        """Return the elapsed seconds at absolute byte ``offset``."""
        if not self.timing:
            return 0.0
        target = offset - self.timing_base
        elapsed = 0.0
        position = 0
        for delay, count in self.timing:
            position += count
            elapsed += delay
            if target < position:
                return elapsed
        return elapsed


def _parse_timing(path: Path) -> tuple[tuple[float, int], ...]:
    """Parse a ``script --timing`` file into ``(delay, bytes)`` pairs."""
    records: list[tuple[float, int]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            records.append((float(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return tuple(records)


def load_typescript(
    path: str | os.PathLike[str],
    timing: str | os.PathLike[str] | None = None,
) -> Transcript:
    """Load a ``script(1)`` typescript and its optional timing file.

    Parameters
    ----------
    path:
        The ``.typescript`` file.  ``script`` banner lines are detected and
        excluded from :attr:`Transcript.body`.
    timing:
        The ``.timing`` sidecar.  When omitted, ``<stem>.timing`` next to
        ``path`` is used if it exists.

    Raises
    ------
    CaptureError
        If ``path`` does not exist or an explicit timing file is missing.
    """
    ts_path = Path(path)
    if not ts_path.is_file():
        raise CaptureError(f"typescript not found: {ts_path}")
    raw = ts_path.read_bytes()

    body_start = 0
    match = _SCRIPT_HEADER_RE.match(raw)
    if match:
        body_start = match.end()
    body_end = len(raw)
    match = _SCRIPT_FOOTER_RE.search(raw)
    if match and match.start() >= body_start:
        body_end = match.start()

    timing_path: Path | None
    if timing is None:
        candidate = ts_path.with_suffix(".timing")
        timing_path = candidate if candidate.is_file() else None
    else:
        timing_path = Path(timing)
        if not timing_path.is_file():
            raise CaptureError(f"timing file not found: {timing_path}")

    records = _parse_timing(timing_path) if timing_path else ()
    counted = sum(count for _, count in records)
    if records and counted == len(raw) - body_start - (len(raw) - body_end):
        base = body_start
    else:
        base = 0
    return Transcript(
        path=ts_path,
        raw=raw,
        timing=records,
        body_start=body_start,
        body_end=body_end,
        timing_base=base,
    )


# --------------------------------------------------------------------------
# Refresh decomposition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    """One physical line of a refresh payload.

    ``text`` is what the terminal shows last, ``cells`` is the widest
    ``CR``-delimited segment (the width that decides wrapping) and ``rows`` is
    how many terminal rows that width occupies.
    """

    text: str
    cells: int
    rows: int


@dataclass(frozen=True)
class Refresh:
    """One redraw of a block (``ESC[<n>A``) or of a single line (bare ``CR``)."""

    index: int
    kind: str  # "block" or "line"
    up: int  # cursor-up count; 0 for a line refresh
    payload: bytes
    offset: int  # absolute offset of the payload start
    time: float
    lines: tuple[Line, ...]

    @property
    def physical_rows(self) -> int:
        """Rows this payload actually consumes on screen."""
        return sum(line.rows for line in self.lines)

    @property
    def drift(self) -> int:
        """Rows the block slips down: physical rows minus cursor-up count.

        A single-line redraw never moves the cursor up, so its drift is always
        ``0``; only block refreshes can accumulate drift.
        """
        if self.kind == "line":
            return 0
        return self.physical_rows - self.up

    @property
    def max_cells(self) -> int:
        """Width of the widest line in this refresh."""
        return max((line.cells for line in self.lines), default=0)


def _measure_payload(payload: bytes, cols: int) -> tuple[Line, ...]:
    """Split a refresh payload into measured physical lines."""
    text = CSI_RE.sub(b"", payload).decode("utf-8", "replace")
    raw_lines = text.split("\n")
    if raw_lines and not raw_lines[-1].strip("\r"):
        raw_lines.pop()
    measured: list[Line] = []
    for raw_line in raw_lines:
        # ``script(1)`` on Linux records the tty's ONLCR translation, so lines
        # end in ``\r\r\n``; a trailing CR neither shows nor occupies a cell.
        segments = raw_line.rstrip("\r").split("\r")
        cells = max((display_width(segment) for segment in segments), default=0)
        shown = next((seg for seg in reversed(segments) if seg), "")
        measured.append(Line(text=shown, cells=cells, rows=_rows_for(cells, cols)))
    return tuple(measured)


def split_refreshes(transcript: Transcript, cols: int) -> list[Refresh]:
    """Decompose a recording into :class:`Refresh` records.

    Block refreshes are preferred: when the stream contains at least one
    ``ESC[<n>A`` the recording is treated as a block-redrawn display and only
    those frames are returned.  Otherwise, bare ``CR`` rewrites are reported as
    single-line refreshes.  Output before the first cursor-up is not a refresh
    and is not returned.
    """
    if cols <= 0:
        raise CaptureError(f"cols must be positive, got {cols}")
    body = transcript.body
    base = transcript.body_start

    marks = list(CURSOR_UP_RE.finditer(body))
    if marks:
        refreshes: list[Refresh] = []
        for index, mark in enumerate(marks):
            end = marks[index + 1].start() if index + 1 < len(marks) else len(body)
            payload = body[mark.end():end]
            offset = base + mark.end()
            refreshes.append(
                Refresh(
                    index=index,
                    kind="block",
                    up=int(mark.group(1) or 1),
                    payload=payload,
                    offset=offset,
                    time=transcript.time_at(offset),
                    lines=_measure_payload(payload, cols),
                )
            )
        return refreshes

    # Single-line redraw: a CR not followed by LF restarts the row.
    marks = list(_LINE_CR_RE.finditer(body))
    refreshes = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(body)
        payload = body[mark.end():end].split(b"\n", 1)[0]
        offset = base + mark.end()
        refreshes.append(
            Refresh(
                index=index,
                kind="line",
                up=0,
                payload=payload,
                offset=offset,
                time=transcript.time_at(offset),
                lines=_measure_payload(payload, cols),
            )
        )
    return refreshes


def _coerce_refreshes(
    source: Transcript | Sequence[Refresh], cols: int
) -> list[Refresh]:
    """Accept either a transcript or an already-split refresh sequence."""
    if isinstance(source, Transcript):
        return split_refreshes(source, cols)
    return list(source)


def drift_per_refresh(
    source: Transcript | Sequence[Refresh], cols: int
) -> list[int]:
    """Return the drift in rows for each block refresh.

    A value of ``0`` means the cursor-up count matched the rows the payload
    occupied.  A positive value is how far the block slipped down on that
    refresh.  Line refreshes are always ``0`` because they never move up.
    """
    return [refresh.drift for refresh in _coerce_refreshes(source, cols)]


def max_line_width(
    source: Transcript | Sequence[Refresh], cols: int = 80
) -> int:
    """Return the widest emitted line, in cells, across all refreshes.

    ``cols`` only matters when ``source`` is a :class:`Transcript` that has not
    been split yet; widths themselves never depend on the terminal size.
    """
    refreshes = _coerce_refreshes(source, cols)
    return max((refresh.max_cells for refresh in refreshes), default=0)


# --------------------------------------------------------------------------
# Screen reconstruction
# --------------------------------------------------------------------------


@dataclass
class Snapshot:
    """A reconstructed screen at one instant."""

    name: str
    time: float
    offset: int
    lines: list[str]
    cols: int
    rows: int
    screen: object = field(repr=False, default=None)

    @property
    def width(self) -> int:
        """Widest non-empty line on this screen."""
        return max((display_width(line) for line in self.lines), default=0)

    def text(self) -> str:
        """Screen contents as text, one line per row."""
        return "\n".join(line.rstrip() for line in self.lines) + "\n"


def _import_pyte():
    """Import :mod:`pyte`, raising a helpful error when it is missing."""
    try:
        import pyte  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise CaptureError(
            'screen reconstruction needs pyte: pip install "fuzzybits[test]"'
        ) from exc
    return pyte


def _resolve_offsets(
    transcript: Transcript,
    at: Iterable[float] | None,
    before_pattern: str | None,
) -> list[tuple[str, float, int]]:
    """Turn requested times/patterns into ``(label, time, offset)`` triples."""
    body = transcript.body
    duration = transcript.duration
    targets: list[tuple[str, float, int]] = []

    for value in at or ():
        if value > 1.0:
            seconds = value
            label = f"t{seconds:07.3f}s"
        else:
            seconds = duration * value
            label = f"t{int(round(value * 100)):03d}"
        offset = _offset_for_time(transcript, seconds)
        targets.append((label, seconds, offset))

    if before_pattern:
        try:
            pattern = re.compile(before_pattern.encode("utf-8"))
        except re.error as exc:
            raise CaptureError(f"invalid --before-pattern: {exc}") from exc
        match = pattern.search(body)
        if match is None:
            raise CaptureError(f"--before-pattern matched nothing: {before_pattern}")
        offset = transcript.body_start + match.start()
        targets.append((f"before_{_slug(before_pattern)}", transcript.time_at(offset), offset))

    return targets


def _offset_for_time(transcript: Transcript, seconds: float) -> int:
    """Return the absolute byte offset reached after ``seconds``."""
    if not transcript.timing:
        return transcript.body_start + len(transcript.body)
    elapsed = 0.0
    position = transcript.timing_base
    for delay, count in transcript.timing:
        elapsed += delay
        position += count
        if elapsed >= seconds:
            break
    return min(max(position, transcript.body_start), transcript.body_end)


def _slug(text: str) -> str:
    """Make a filesystem-safe label out of an arbitrary string."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return (slug[:40] or "pattern").lower()


def screens_at(
    transcript: Transcript,
    cols: int,
    rows: int,
    at: Iterable[float] | None = None,
    before_pattern: str | None = None,
) -> list[Snapshot]:
    """Reconstruct the screen at given fractions and/or before a pattern.

    Parameters
    ----------
    transcript:
        Recording to replay.
    cols, rows:
        Terminal geometry to emulate; must match the size used to record.
    at:
        Fractions of the recording duration in ``(0, 1]``.  Values above ``1``
        are interpreted as seconds.
    before_pattern:
        Regular expression; the screen is captured immediately before its first
        match in the child output.  Useful for "the frame just before the
        summary table was printed".

    Notes
    -----
    Sets a UTF-8 ctype via :func:`ensure_utf8_locale` before replaying, because
    ``pyte`` drops the rest of a line when a glyph's width is unknown.
    """
    pyte = _import_pyte()
    ensure_utf8_locale()
    if cols <= 0 or rows <= 0:
        raise CaptureError(f"geometry must be positive, got {cols}x{rows}")

    targets = _resolve_offsets(transcript, at, before_pattern)
    snapshots: list[Snapshot] = []
    for label, seconds, offset in targets:
        screen = pyte.Screen(cols, rows)
        stream = pyte.ByteStream(screen)
        stream.feed(transcript.body[: offset - transcript.body_start])
        snapshots.append(
            Snapshot(
                name=label,
                time=seconds,
                offset=offset,
                lines=[line.rstrip() for line in screen.display],
                cols=cols,
                rows=rows,
                screen=screen,
            )
        )
    return snapshots


def _colour(spec: str | None) -> str:
    """Map a ``pyte`` colour name to a hex string."""
    if not spec:
        return _DEFAULT_FG
    if spec.startswith("#"):
        return spec
    name = spec.lower()
    if name in _BASE_COLOURS:
        return _BASE_COLOURS[name]
    if name.startswith("bright"):
        base = _BASE_COLOURS.get(name[6:])
        if base:
            channels = (int(base[i:i + 2], 16) for i in (1, 3, 5))
            return "#" + "".join(f"{min(255, channel + 40):02x}" for channel in channels)
    return _DEFAULT_FG


def render_png(snapshot: Snapshot, path: str | os.PathLike[str], font_size: int = 15) -> Path:
    """Rasterise a snapshot to PNG, honouring per-cell foreground colour.

    Raises
    ------
    CaptureError
        If Pillow or a monospaced font is unavailable.  PNG output is optional;
        callers that only need text should not call this.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415 - optional
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise CaptureError(
            'PNG output needs Pillow: pip install "fuzzybits[test,png]"'
        ) from exc

    font_path = next((p for p in FONT_CANDIDATES if Path(p).is_file()), None)
    if font_path is None:
        raise CaptureError(
            "no monospaced font found; install fonts-dejavu-core or pass a font"
        )
    font = ImageFont.truetype(font_path, font_size)
    cell_w = font.getlength("M")
    ascent, descent = font.getmetrics()
    cell_h = ascent + descent
    pad = 12
    image = Image.new(
        "RGB",
        (int(snapshot.cols * cell_w) + 2 * pad, snapshot.rows * cell_h + 2 * pad),
        _BACKGROUND,
    )
    draw = ImageDraw.Draw(image)
    screen = snapshot.screen
    for y in range(snapshot.rows):
        for x in range(snapshot.cols):
            cell = screen.buffer[y][x]
            if cell.data and cell.data != " ":
                draw.text(
                    (pad + x * cell_w, pad + y * cell_h),
                    cell.data,
                    font=font,
                    fill=_colour(cell.fg),
                )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


@dataclass
class Recording:
    """Result of :func:`record`."""

    name: str
    typescript: Path
    timing: Path
    exit_code: int
    duration: float


def script_supports_timing() -> bool:
    """Return whether the local ``script(1)`` is the util-linux build."""
    binary = shutil.which("script")
    if binary is None:
        return False
    try:
        proc = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return False
    return "--timing" in (proc.stdout + proc.stderr)


def record(
    name: str,
    command: Sequence[str],
    out_dir: str | os.PathLike[str] = ".",
    cols: int = 120,
    rows: int = 40,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
) -> Recording:
    """Record ``command`` in a PTY of ``cols`` x ``rows``.

    Uses ``script(1)`` so the child sees a real terminal and behaves exactly as
    it would interactively.  Writes ``<name>.typescript`` and ``<name>.timing``
    into ``out_dir``.

    Raises
    ------
    CaptureError
        If ``script(1)`` is missing, is not the util-linux build (macOS and BSD
        ``script`` lack ``--timing``), the command is empty, or the geometry is
        invalid.
    """
    if not command:
        raise CaptureError("record needs a command to run")
    if cols <= 0 or rows <= 0:
        raise CaptureError(f"geometry must be positive, got {cols}x{rows}")
    if not script_supports_timing():
        raise CaptureError(
            "record needs util-linux script(1) with --timing; install util-linux "
            "or record on Linux (macOS/BSD script lacks --timing)"
        )

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    typescript = directory / f"{name}.typescript"
    timing = directory / f"{name}.timing"

    child_env = dict(os.environ)
    child_env.update(DEFAULT_ENV)
    child_env.update(env or {})

    shell_command = (
        f"stty cols {cols} rows {rows}; exec {shlex.join(list(command))}"
    )
    argv = [
        shutil.which("script") or "script",
        "-q",
        f"--timing={timing}",
        "-e",
        "-c",
        shell_command,
        str(typescript),
    ]
    started = _monotonic()
    try:
        proc = subprocess.run(
            argv,
            env=child_env,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(f"record timed out after {timeout}s: {name}") from exc
    duration = _monotonic() - started
    if not typescript.is_file():  # pragma: no cover - defensive
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise CaptureError(f"script(1) produced no typescript for {name}: {detail}")
    return Recording(
        name=name,
        typescript=typescript,
        timing=timing if timing.is_file() else Path(),
        exit_code=proc.returncode,
        duration=duration,
    )


def _monotonic() -> float:
    """Wall clock helper, separated so tests can monkeypatch it if needed."""
    import time

    return time.monotonic()


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


@dataclass
class Report:
    """Quantitative summary of one recording.

    ``requested_rows``/``physical_rows``/``total_drift`` cover the *display*
    frames only.  The final cursor-up frame normally also carries whatever the
    process printed after the display finished (a report, a summary line), so it
    is excluded from those totals and reported separately as
    ``final_frame_rows`` and ``trailing_max_width``.
    """

    typescript: str
    cols: int
    rows: int
    duration: float
    refresh_count: int
    kind: str
    up: int
    drift_rows_per_refresh: int
    requested_rows: int
    physical_rows: int
    total_drift: int
    max_width: int
    width_exceeded: bool
    cursor_hide: int
    cursor_show: int
    cadence: dict[str, float | int]
    overflow_lines: list[dict[str, object]]
    per_refresh: list[dict[str, object]]
    final_frame_rows: int = 0
    trailing_max_width: int = 0
    startup_drift: int = 0

    @property
    def cursor_balanced(self) -> bool:
        """Whether the cursor was restored as often as it was hidden."""
        return self.cursor_hide == self.cursor_show

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable form."""
        return {
            "typescript": self.typescript,
            "cols": self.cols,
            "rows": self.rows,
            "duration_s": round(self.duration, 4),
            "kind": self.kind,
            "refreshes": self.refresh_count,
            "steady_state": {
                "up": self.up,
                "drift_rows_per_refresh": self.drift_rows_per_refresh,
            },
            "drift": {
                "requested_rows": self.requested_rows,
                "physical_rows": self.physical_rows,
                "total_rows": self.total_drift,
                "startup_rows": self.startup_drift,
                "final_frame_rows": self.final_frame_rows,
            },
            "width": {
                "max_cells": self.max_width,
                "terminal_cols": self.cols,
                "exceeded": self.width_exceeded,
                "trailing_max_cells": self.trailing_max_width,
            },
            "cadence": self.cadence,
            "cursor": {
                "hide": self.cursor_hide,
                "show": self.cursor_show,
                "balanced": self.cursor_balanced,
            },
            "overflow_lines": self.overflow_lines,
            "per_refresh": self.per_refresh,
        }

    def text(self) -> str:
        """Return the short human-readable report."""
        lines = [
            f"typescript      {self.typescript}",
            f"duration        {self.duration:.2f} s",
            f"terminal        {self.cols}x{self.rows}",
            f"refreshes       {self.refresh_count} ({self.kind})",
            "drift           "
            f"up={self.up}, +{self.drift_rows_per_refresh} rows/refresh "
            f"({self.requested_rows} requested, {self.physical_rows} physical, "
            f"{self.total_drift:+d} over {self.refresh_count - 1} frames)",
            f"width           widest {self.max_width} cells vs {self.cols} cols"
            f"{' -> exceeds' if self.width_exceeded else ' -> fits'}",
            f"cadence         {self.cadence.get('bursts', 0)} bursts, "
            f"median gap {self.cadence.get('median_gap_s', 0.0):.3f} s, "
            f"max {self.cadence.get('max_gap_s', 0.0):.3f} s, "
            f"{self.cadence.get('frames_per_s', 0.0):.2f} refreshes/s",
            f"cursor          hidden {self.cursor_hide}x, "
            f"shown {self.cursor_show}x "
            f"({'balanced' if self.cursor_balanced else 'UNBALANCED'})",
        ]
        if self.final_frame_rows:
            lines.append(
                f"final frame     {self.final_frame_rows} rows "
                f"(widest {self.trailing_max_width} cells; excluded from totals)"
            )
        if self.startup_drift:
            lines.append(
                f"startup drift   {self.startup_drift:+d} rows while the block grew "
                "(cursor-up uses the previous frame's bar count)"
            )
        if self.overflow_lines:
            lines.append("overflowing lines:")
            for entry in self.overflow_lines:
                lines.append(
                    f"  {entry['cells']:>4} cells x{entry['refreshes']:<4} "
                    f"{entry['text']}"
                )
        return "\n".join(lines) + "\n"


def _cadence(times: Sequence[float], threshold: float = 0.02) -> dict[str, float | int]:
    """Summarise inter-refresh gaps and group them into bursts."""
    if len(times) < 2:
        return {
            "bursts": len(times),
            "median_gap_s": 0.0,
            "min_gap_s": 0.0,
            "max_gap_s": 0.0,
            "frames_per_s": 0.0,
        }
    gaps = [b - a for a, b in zip(times, times[1:]) if b >= a]
    span = times[-1] - times[0]
    bursts = 1
    for gap in gaps:
        if gap >= threshold:
            bursts += 1
    ordered = sorted(gaps)
    return {
        "bursts": bursts,
        "median_gap_s": round(ordered[len(ordered) // 2], 4),
        "min_gap_s": round(ordered[0], 4),
        "max_gap_s": round(ordered[-1], 4),
        "frames_per_s": round((len(times) - 1) / span, 3) if span > 0 else 0.0,
    }


def _modal(values: Sequence[int]) -> int:
    """Return the most common value, preferring the latest on a tie."""
    if not values:
        return 0
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best = max(counts.values())
    for value in reversed(values):
        if counts[value] == best:
            return value
    return values[-1]  # pragma: no cover - unreachable


def _startup_drift(display: Sequence[Refresh]) -> int:
    """Return rows lost while the block was still growing.

    Progress libraries that call ``move_up(previous_bar_count)`` land the frame
    one row lower for every bar appended after the first, even when every line
    fits the terminal.  Such frames are identifiable because they drift without
    containing a wrapped line; steady-state wrap drift always involves at least
    one line wider than the terminal.
    """
    return sum(
        refresh.drift
        for refresh in display
        if refresh.drift > 0 and not any(line.rows > 1 for line in refresh.lines)
    )


def analyse(
    transcript: Transcript,
    cols: int,
    rows: int = 0,
    steady_fraction: float = 0.4,
) -> Report:
    """Measure a recording and return a :class:`Report`.

    ``rows`` is recorded for context only; the measurements depend on ``cols``.
    ``steady_fraction`` selects the tail of the run used for the "typical"
    drift figure, because the first frames still grow the block one bar at a
    time and are not representative.
    """
    refreshes = split_refreshes(transcript, cols)
    kind = refreshes[0].kind if refreshes else "none"

    block = [refresh for refresh in refreshes if refresh.kind == "block"]
    counted = block or list(refreshes)
    # The last cursor-up frame swallows everything printed after the display
    # finished (a report, a closing summary), so it stops describing the display
    # itself.  Detect that instead of assuming it: a real report is many times
    # taller than any display frame, whereas a frame that merely gained one more
    # bar is not.
    final = counted[-1] if counted else None
    body_frames = counted[:-1] if len(counted) > 1 else list(counted)
    peak = max((refresh.physical_rows for refresh in body_frames), default=0)
    trailing = (
        final is not None
        and len(counted) > 1
        and final.physical_rows > TRAILING_FACTOR * peak
    )
    display = body_frames if trailing else list(counted)
    start = int(len(display) * steady_fraction)
    steady = display[start:] or display

    overflow_counter: dict[str, dict[str, object]] = {}
    for refresh in display:
        for line in refresh.lines:
            if line.cells > cols:
                entry = overflow_counter.setdefault(
                    line.text, {"text": line.text, "cells": line.cells, "refreshes": 0}
                )
                entry["refreshes"] = int(entry["refreshes"]) + 1

    overflow_lines = sorted(
        overflow_counter.values(), key=lambda item: (-int(item["cells"]), str(item["text"]))
    )[:20]

    return Report(
        typescript=str(transcript.path),
        cols=cols,
        rows=rows,
        duration=transcript.duration,
        refresh_count=len(refreshes),
        kind=kind,
        up=_modal([refresh.up for refresh in steady]),
        drift_rows_per_refresh=_modal([refresh.drift for refresh in steady]),
        requested_rows=sum(refresh.up for refresh in display),
        physical_rows=sum(refresh.physical_rows for refresh in display),
        total_drift=sum(refresh.drift for refresh in display),
        max_width=max((refresh.max_cells for refresh in display), default=0),
        width_exceeded=any(
            line.cells > cols for refresh in display for line in refresh.lines
        ),
        cursor_hide=len(CURSOR_HIDE_RE.findall(transcript.body)),
        cursor_show=len(CURSOR_SHOW_RE.findall(transcript.body)),
        cadence=_cadence([refresh.time for refresh in display]),
        overflow_lines=overflow_lines,
        startup_drift=_startup_drift(display),
        final_frame_rows=final.physical_rows if trailing else 0,
        trailing_max_width=final.max_cells if trailing else 0,
        per_refresh=[
            {
                "index": refresh.index,
                "kind": refresh.kind,
                "up": refresh.up,
                "physical_rows": refresh.physical_rows,
                "drift": refresh.drift,
                "max_cells": refresh.max_cells,
                "wrapped_lines": sum(1 for line in refresh.lines if line.rows > 1),
                "time_s": round(refresh.time, 4),
                "widths": [line.cells for line in refresh.lines],
            }
            for refresh in refreshes
        ],
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_env(pairs: Iterable[str]) -> dict[str, str]:
    """Parse repeated ``KEY=VAL`` options."""
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise CaptureError(f"--env expects KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def _parse_at(text: str) -> list[float]:
    """Parse a comma-separated list of fractions or seconds."""
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            if item.endswith("%"):
                values.append(float(item[:-1]) / 100.0)
            elif item.endswith("s"):
                values.append(float(item[:-1]))
            else:
                values.append(float(item))
        except ValueError as exc:
            raise CaptureError(f"--at expects numbers, got {item!r}") from exc
    if not values:
        raise CaptureError("--at needs at least one value")
    return values


def _cmd_record(args: argparse.Namespace) -> int:
    result = record(
        name=args.name,
        command=args.command,
        out_dir=args.out,
        cols=args.cols,
        rows=args.rows,
        env=_parse_env(args.env),
        cwd=args.cwd,
        timeout=args.timeout,
    )
    print(f"recorded  {result.typescript}")
    if result.timing.name:
        print(f"timing    {result.timing}")
    print(f"exit      {result.exit_code}  ({result.duration:.2f} s)")
    return 0 if result.exit_code == 0 else result.exit_code


def _cmd_render(args: argparse.Namespace) -> int:
    transcript = load_typescript(args.typescript, args.timing)
    at = _parse_at(args.at) if args.at else None
    snapshots = screens_at(
        transcript,
        cols=args.cols,
        rows=args.rows,
        at=at,
        before_pattern=args.before_pattern,
    )
    if not snapshots:
        raise CaptureError("nothing to render: pass --at and/or --before-pattern")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for snapshot in snapshots:
        label = f"{args.name or transcript.name}_{snapshot.name}"
        text_path = out_dir / f"{label}.txt"
        text_path.write_text(snapshot.text(), encoding="utf-8")
        written = [str(text_path)]
        if not args.no_png:
            try:
                png_path = render_png(snapshot, out_dir / f"{label}.png")
            except CaptureError as exc:
                print(f"png skipped for {label}: {exc}", file=sys.stderr)
            else:
                written.append(str(png_path))
        print(f"{label}: t={snapshot.time:.2f}s width={snapshot.width} -> "
              + ", ".join(written))
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    transcript = load_typescript(args.typescript, args.timing)
    report = analyse(transcript, cols=args.cols, rows=args.rows)
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"json      {target}")
    if not args.quiet:
        sys.stdout.write(report.text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the command line interface."""
    parser = argparse.ArgumentParser(
        prog="terminal_capture",
        description="Record, render and analyse the terminal output of CLI tools.",
    )
    sub = parser.add_subparsers(dest="command_name", required=True)

    rec = sub.add_parser("record", help="record a command in a fixed-size PTY")
    rec.add_argument("--name", required=True, help="output basename (<name>.typescript)")
    rec.add_argument("--out", default=".", help="output directory (default: .)")
    rec.add_argument("--cols", type=int, default=120, help="terminal columns")
    rec.add_argument("--rows", type=int, default=40, help="terminal rows")
    rec.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="extra environment variable for the child (repeatable)",
    )
    rec.add_argument("--cwd", default=None, help="working directory for the child")
    rec.add_argument("--timeout", type=float, default=None, help="kills the run after N s")
    rec.add_argument("child", nargs=argparse.REMAINDER, metavar="-- command...")
    rec.set_defaults(handler=_cmd_record, command=[])

    ren = sub.add_parser("render", help="reconstruct screens from a recording")
    ren.add_argument("--typescript", required=True, help="path to <name>.typescript")
    ren.add_argument("--timing", default=None, help="path to <name>.timing")
    ren.add_argument("--cols", type=int, required=True, help="terminal columns")
    ren.add_argument("--rows", type=int, required=True, help="terminal rows")
    ren.add_argument("--out", default="frames", help="output directory")
    ren.add_argument("--name", default=None, help="output basename override")
    ren.add_argument(
        "--at",
        default="0.15,0.5,1.0",
        help="comma-separated fractions of the run, or seconds with an 's' suffix",
    )
    ren.add_argument(
        "--before-pattern",
        default=None,
        help="also capture the screen immediately before this regex matches",
    )
    ren.add_argument("--no-png", action="store_true", help="write .txt only")
    ren.set_defaults(handler=_cmd_render)

    ana = sub.add_parser("analyse", help="measure widths, drift and cadence")
    ana.add_argument("--typescript", required=True, help="path to <name>.typescript")
    ana.add_argument("--timing", default=None, help="path to <name>.timing")
    ana.add_argument("--cols", type=int, required=True, help="terminal columns")
    ana.add_argument("--rows", type=int, default=0, help="terminal rows (context only)")
    ana.add_argument("--json", default=None, help="write the full report to this path")
    ana.add_argument("--quiet", action="store_true", help="suppress the text report")
    ana.set_defaults(handler=_cmd_analyse)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command_name == "record":
        command = list(args.child)
        if command and command[0] == "--":
            command = command[1:]
        args.command = command
        if not command:
            parser.error("record needs a command after --")
    try:
        return int(args.handler(args))
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
