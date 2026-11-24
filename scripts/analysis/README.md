# scripts/analysis

Offline analyses of fault-tree models and terminal output. Python ≥ 3.12.

## `module_census.py` — module census and task-parallel bounds

Requirements: the Python standard library only.

Purpose: measure how a model splits into **modules**, i.e. subgraphs connected
to the rest only through their root. That split decides whether a bottom-up,
module-wise compilation could gain from task parallelism. This is measurement 2
of [#213]. The intended consumer is the exact module-collapse pre-pass of
[#21], which collapses modules into node-budgeted BDDs. Sibling modules in the
module tree are independent and could be collapsed concurrently, so the
speedup is bounded by total work divided by the critical path of the module
tree. Raw-graph results for Aralia and the synthetic sets are posted on
[#213].

[#213]: https://github.com/fuzzybit-org/fuzzybits/issues/213
[#21]: https://github.com/fuzzybit-org/fuzzybits/issues/21

### Usage

```bash
# Markdown table for every *.xml below a directory (recursive)
python scripts/analysis/module_census.py input/Aralia

# Files and globs (quote globs; ** is recursive), all metrics to CSV and JSON.
# The synthetic models are the input/synthetic-models git submodule:
#   git submodule update --init input/synthetic-models
python scripts/analysis/module_census.py --csv census.csv --json census.json \
    input/Aralia/nus9601.xml 'input/synthetic-models/models/2_to_100k/openpsa/ft-1000-0*.xml'

# Other core counts for the schedule bound; keep going past broken files
python scripts/analysis/module_census.py --cores 4,16,64 --keep-going --markdown --csv - models/
```

| option | effect |
|---|---|
| `PATH...` | a MEF file, a directory (searched recursively for `*.xml`) or a glob |
| `--cores 8,48` | core counts P for the greedy-schedule bound `S_P` (default 8 and 48) |
| `--csv OUT`, `--json OUT` | write all metrics; `-` writes to stdout |
| `--markdown` | print the markdown table even when `--csv` or `--json` is given (it is printed by default otherwise) |
| `--fold-not` | treat single-operand `not`/`nand`/`nor` as pass-through (structure only) |
| `--keep-going` | report a failing file and continue; the exit status is still 1 |

A malformed, unreadable or cyclic file stops the run with exit status 1 and a
message `module_census: error: <path>: <reason>`. A missing path or an empty
glob is reported the same way.

As a library (put `scripts/analysis` on `sys.path`):

```python
import module_census as mc

result = mc.census_file("input/Aralia/das9205.xml", cores=(8, 48))
result.modules, result.s_inf, result.speedups[8]  # 17, 3.43, 3.43

details = mc.analyse_graph(mc.parse_mef("input/Aralia/edf9206.xml"))
details.module_names(), details.tree.local_edges

mc.self_check_random_dags(trials=600, seed=213)  # raises on any disagreement
```

The building blocks are separate functions:

- `parse_mef`
- `normalize_pass_through`
- `detect_modules`
- `build_module_tree`
- `critical_path`, `greedy_list_schedule` and `speedup_bounds`
- `census_graph` and `census_file`
- `definition_modules` and `self_check_random_dags`, the brute-force check
  used by the tests

### Output columns

The markdown table shows the main columns; CSV and JSON carry every field of
`CensusResult`.

| column | meaning |
|---|---|
| `gates`, `basic_events`, `edges` | after normalisation, within the root cones; gates include anonymous nested formulas |
| `modules` | number of modules |
| `nontrivial_modules` | modules whose local part has ≥ 2 gates |
| `shared_modules` | modules with ≥ 2 distinct parent gates |
| `total_local_work` (W) | summed local work; equals `edges` |
| `lw_max`, `lw_p90`, `lw_median` | local work over nontrivial modules (p90 by nearest rank) |
| `largest_share` | largest local work of any module divided by W |
| `max_local_gates` | largest number of gates in one local part |
| `max_local_vars` | largest number of local variables in one module |
| `tree_depth`, `max_width`, `max_width_nontrivial` | module-tree depth (root module = 1) and modules per depth level |
| `critical_path` (CP) | heaviest root-to-leaf chain of local work in the module tree |
| `s_inf`, `s_<P>` | W / CP and W / T_P (see below) |
| `roots`, `virtual_root_local_gates` | parentless gates; gates not owned by any real module when there are several roots |
| `named_gates`, `anonymous_gates`, `defined_basic_events` | raw counts from the file |
| `pass_through_removed` | gates removed by normalisation |
| `parse_s`, `census_s` | wall time |
| `warnings` | e.g. undefined gates, CCF groups ignored |

### Method

**Parsing.** The MEF file is streamed with `xml.etree.ElementTree.iterparse`.
The encoding comes from the declaration or the byte-order mark, so UTF-16
files work.

- Every `define-gate` and every nested formula becomes a gate node. The
  formulas are `and`, `or`, `atleast`, `not`, `xor`, `nand`, `nor`, `null`,
  `iff`, `imply` and `cardinality`.
- Basic events, house events and constants are leaves.
- References (`gate`, `basic-event`, `house-event`, untyped `event`) are
  resolved after the whole file is read. An untyped reference goes to the
  gate, basic event or house event with that name, in that order.
- Formula nesting is processed iteratively.

**Normalisation.** References are forwarded through pass-through gates, to a
fixed point:

- the pass-through gates are single-operand `and`/`or`/`null` gates, a
  `define-gate` whose formula is a bare reference, and single-operand
  `atleast` gates with k ≤ 1;
- duplicate operands of idempotent gates are merged, which can create new
  pass-through gates;
- `atleast` (k-of-n) stays one gate;
- `not` stays a gate unless `--fold-not` is given.

The roots are the parentless gates of the raw graph after forwarding. With
more than one root, a zero-cost virtual root is added over them, so that the
union of the root cones is analysed.

**Module detection** follows Dutuit & Rauzy (1996) and runs in O(V + E).

1. One iterative DFS (no recursion; depth can be in the tens of thousands)
   with a global clock that ticks on every edge and every gate exit. For each
   node it records the first visit and the last visit (updated on every
   revisit); for each gate it also records the exit date.
2. Bottom-up in exit order, each gate gets the minimum first visit and the
   maximum last visit over its descendants.
3. Gate v is a module iff all those dates lie strictly inside
   (first(v), exit(v)). Leaves count, so one shared basic event breaks
   modularity.

**Module tree and local work.**

- Each non-module gate belongs to its nearest module ancestor. That module's
  **local part** is the module itself plus those gates, with child modules
  treated as single variables.
- Module cones are laminar, so the module structure is a **tree**. A module may
  have several parent gates, but they always lie in one parent module's local
  part. The code checks this.
- **Local work** is the cost proxy: the number of operand edges out of local
  gates. Summed over all modules it equals the edge count exactly.
- **Local variables** are the distinct leaves and child modules a local part
  references.

**Scheduling bounds.**

- Tasks are the module-tree nodes, weighted by local work. A module may start
  once all its child modules are done. A shared module is one task.
- `s_inf` = W / CP is the work/span bound with unlimited cores.
- `s_<P>` = W / T_P, where T_P is the makespan of a non-preemptive greedy list
  schedule: whenever a core is free, it takes the ready task with the largest
  summed local work from the task up to the root.
- Every such schedule satisfies `max(W/P, CP) ≤ T_P ≤ W/P + CP` (Graham), and
  the code checks it. So `s_<P> ≤ min(P, s_inf)`.

**Validation.** `self_check_random_dags` runs the full census on random
multi-root DAGs. It compares the linear-time detection with the definition: v
is a module iff every proper descendant has all its parents inside cone(v).
The tests in `tests/python/test_module_census.py` run 600 such models, plus
hand-made DAGs, a 30k-gate chain and three public Aralia models.

### Limits and caveats

- **The cost proxy is linear in local edges.** BDD construction is not. Its
  cost depends on variable order and can grow exponentially with the number of
  local variables, which is why `max_local_vars` is reported. The S values are
  bounds under this proxy, not predicted wall-clock speedups.
- **Raw graphs differ from compiled graphs.** The census runs on the MEF graph
  after pass-through removal only. Module counts after compilation differ:
  - the legacy preprocessor (`benchmarks/compiler.csv`, C2 and above)
    normalises, coalesces and decomposes gates, and synthesises sub-modules by
    grouping modular arguments. On Aralia its counts are mostly higher; das9601
    goes the other way (28 here, 1 at C2).
  - At C0 and C1 the legacy preprocessor does not run module detection.
  - #213 asks for the census on the compiled `ir::pc` graph. That needs this
    detection run inside the compiler; the raw-graph numbers size the question
    and serve as a correctness and smoke set.
- **The analysis is structural only.** Negations, constants and house-event
  values are not propagated, and house events count as variables. CCF groups
  are ignored, with a warning.
- **Performance.** For the largest public synthetic models (about 54k gates,
  36 MB UTF-16), parsing takes about 1 s and the census about 0.5 s.

## `terminal_capture.py` — terminal capture and analysis for CLI output

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

## `ftp-reference/` — exact reference values for FTP models

Tools that quantify a CAFTA FTP model without fuzzybits' compiler: `ftp2mef.py` (FTP to OpenPSA MEF for MEF-only
tools), `ftp_cudd.py` and `ftp_cudd.c` (the exact Boolean probability from a CUDD BDD, evaluated in rational
arithmetic) and `ftp_cudd_check.py` (a BDD dump against direct evaluation of the tree). See
[`ftp-reference/README.md`](ftp-reference/README.md).
