# Analysis scripts

Helpers for measuring the tools in this repository rather than eyeballing them.
They are developer utilities, not part of the shipped package.

## Terminal capture and analysis (`terminal_capture.py`)

`scram-cli --watch` and friends draw multi-line progress blocks that redraw in
place. Whether such a block behaves depends on facts that are hard to see by
looking at a terminal: how wide each emitted line really is in cells, how many
rows a redraw actually consumes, and whether the cursor bookkeeping in the
drawing library agrees with that. `terminal_capture.py` records the output,
reconstructs the screen, and measures all of it.

### Requirements

`pyte` and `wcwidth` are needed for `render` and `analyse` and are in the
`test` extra (`pip install "fuzzybits[test]"`). `record` needs util-linux
`script(1)` with `--timing`, i.e. Linux; macOS and BSD `script` lack the flag
and `record` exits with an explanatory error. PNG frames additionally need
Pillow (`pip install pillow`) and a monospaced font (DejaVu Sans Mono, or a
system fallback); without them `render` still writes the `.txt` screens.

### Usage

Record a command in a real PTY at a fixed size. The child inherits a UTF-8
locale and `TERM=xterm-256color` by default:

```bash
python3 scripts/analysis/terminal_capture.py record \
    --name watch-120x40 --out /tmp/cap --cols 120 --rows 40 \
    -- build/generic/targets/scram/scram-cli input/Aralia/chinese.xml \
       --probability --watch
```

This writes `/tmp/cap/watch-120x40.typescript` and `.timing`. Reconstruct the
screen at 15 %, 50 % and 100 % of the run, and immediately before the report
was printed:

```bash
python3 scripts/analysis/terminal_capture.py render \
    --typescript /tmp/cap/watch-120x40.typescript \
    --cols 120 --rows 40 --out /tmp/cap/frames \
    --at 0.15,0.5,1.0 --before-pattern '<report'
```

`--at` accepts fractions of the recorded duration or seconds (`30s`, `0.5`).
`--before-pattern` takes a regex matched against the raw output; the frame is
captured just before the first match, which is the way to see the display as it
looked before a summary scrolled it away.

Measure the recording:

```bash
python3 scripts/analysis/terminal_capture.py analyse \
    --typescript /tmp/cap/watch-120x40.typescript \
    --cols 120 --rows 40 --json /tmp/cap/report.json
```

### What `analyse` measures

A *refresh* is one redraw. Two idioms are recognised: a block refresh, where
`ESC[<n>A` moves the cursor up *n* rows and the block is rewritten in full, and
a single-line refresh, where a bare `CR` rewrites the current row. For each
refresh the tool reports:

* **Line widths in cells**, measured on the escape-stripped payload against the
  terminal width. A line width is the widest `CR`-delimited write on that row,
  because a `CR` returns to column 0. Lines wider than the terminal are listed
  with how often they occurred.
* **Rows per refresh**: `ceil(width / cols)` per line, minimum one.
* **Drift**: physical rows consumed minus the cursor-up count. A positive value
  is how far the block slipped down. The steady-state figure is the modal
  drift over the tail of the run; `startup drift` accounts separately for rows
  lost while the block was still growing, which happens when the drawing
  library issues `move_up` with the *previous* frame's bar count.
* **Refresh cadence**: inter-refresh gaps, how many bursts of updates there
  were, and the mean rate.
* **Cursor show/hide balance**: how often the cursor was hidden versus
  restored. An imbalance means the terminal is left with an invisible cursor.
* **Whether the output exceeded the width** at all, and the widest line seen.

Output is a short text report on stdout and, with `--json`, the full data set
including a per-refresh array. The last cursor-up frame normally also carries
whatever the process printed after the display finished; when it is many times
taller than a display frame it is reported as `final frame` and excluded from
the drift totals.

Measurements were used to characterise `scram-cli --watch`: at 120x40 the six
bar lines are 120–126 cells wide, so three to four of them wrap, and the block
drifts roughly four rows per refresh; at 160x40 nothing wraps and the
steady-state drift is zero.

### Library use

`scripts/analysis/terminal_capture.py` is importable so that tests can assert on
recorded or synthetic output without shelling out:

```python
import sys
sys.path.insert(0, "scripts/analysis")
import terminal_capture as tc

transcript = tc.load_typescript("cap.typescript")
snapshots = tc.screens_at(transcript, cols=120, rows=40, at=[1.0])
refreshes = tc.split_refreshes(transcript, cols=120)
print(tc.max_line_width(refreshes), tc.drift_per_refresh(refreshes, 120))
report = tc.analyse(transcript, cols=120, rows=40)
```

`load_typescript`, `screens_at`, `max_line_width` and `drift_per_refresh` are
the supported entry points; the rest of the module is an implementation detail.

### Locale caveat for ambiguous-width glyphs

Widths depend on the locale. Block-drawing and punctuation glyphs such as
`■` (U+25A0), Greek letters such as `ε`, and box-drawing characters have
*ambiguous* East Asian width: one cell in a non-CJK terminal, two in a CJK one.
`wcwidth` and `pyte` both consult the ctype, and older `pyte` builds return `-1`
for any non-ASCII glyph under a POSIX ctype, which makes `Screen.draw()` drop
the rest of the line.

Two consequences:

* `record` exports `LC_ALL=C.UTF-8` and `LANG=C.UTF-8` unless `--env`
  overrides them, so the child's `display_width()` behaves like a normal
  terminal's. A program that measures its own output with a POSIX ctype under a
  UTF-8 terminal can compute the wrong bar geometry; record both ways with
  `--env LC_ALL=C` to compare.
* `screens_at` calls `ensure_utf8_locale()` before replaying. Widths in
  `analyse` always go through `wcwidth`, never `len()`, so a CJK character
  counts as two cells and a combining mark as zero.

Where the real terminal is CJK, pass `--env LC_ALL=ja_JP.UTF-8` (or whichever
CJK locale is installed on both the emitter and the capture host) so that both
sides agree on ambiguous widths.
