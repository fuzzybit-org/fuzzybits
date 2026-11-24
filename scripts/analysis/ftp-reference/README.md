# ftp-reference: exact reference values for FTP models

Tools that give a CAFTA FTP model a probability without fuzzybits' compiler, to check what it computes.

| tool | gives | needs |
|---|---|---|
| `ftp2mef.py` | the model as OpenPSA MEF, for MEF-only tools such as the legacy `scram-cli` | Python standard library |
| `ftp_cudd.py`, `ftp_cudd.c` | the exact Boolean top-event probability, as a rational, from a CUDD BDD | CUDD 3.0.0, a C compiler, Python |
| `ftp_cudd_check.py` | a check of a `ftp_cudd.py` BDD dump against direct evaluation of the tree | Python standard library |

All of them give the fixtures' `boolean` value: `-NAME` is a plain Boolean NOT (`ftp2mef.py` refuses it), not FTP
delete-term semantics, and `I` flags are ignored.

## `ftp2mef.py`: CAFTA FTP to OpenPSA MEF, negation-free models

Purpose: hand a negation-free FTP model (OR, AND, k-of-n gates; `IMPORT` values) to MEF-only tools, such as the
legacy `scram-cli` (ZBDD, MOCUS, BDD), for an independent Boolean reference value. It refuses complemented operands:
a `-NAME` means a delete term or an atom in FTP, and MEF `<not>` would silently change that meaning.

```bash
python3 scripts/analysis/ftp-reference/ftp2mef.py input/ftp/synthetic/name_500_06.ftp /tmp/m.xml
scram-cli /tmp/m.xml --zbdd --probability --mcub --cut-off 1e-28 -l 60 -o /tmp/z.xml
```

Limits: exactly one `PROCESS` target; `LIMIT`, headers and import flags are dropped; values are written with Python
`repr`, so they round-trip the doubles the FTP reader parses.

## `ftp_cudd.py`, `ftp_cudd.c`: exact FTP top-event probability with a CUDD BDD

Requirements: CUDD 3.0.0 (C library), a C compiler, and Python (standard library only; tested with 3.14 and gcc
15.2). Linux; the peak-RSS figure the driver prints assumes Linux `ru_maxrss` units.

Purpose: an exact Boolean reference value for an FTP model, independent of fuzzybits and of cut-set truncation.

- `ftp_cudd.py` parses the FTP file and writes a plain `.net` file: variables in DFS first-occurrence order from
  the targets, gates in topological order.
- `ftp_cudd.c` builds the BDD with CUDD automatic reordering, prints double and long-double probabilities, and
  dumps the BDD. It carries `(P, 1 - P)` per regular node, so it never subtracts either.
- `ftp_cudd.py` then evaluates the dump in exact rational arithmetic. Every probability is an integer over
  `S = 10^d`, and a node at level `l` carries an integer numerator over `S^(L - l)`, so complemented edges are
  `S^k - N` with no rounding.

Build CUDD outside the checkout. After a `git clone` of github.com/ivmai/cudd, the timestamps make `make` try to
rerun automake, so touch the generated files first:

```bash
cd cudd && touch aclocal.m4 && sleep 1 && touch configure Makefile.in config.h.in
mkdir ../build && cd ../build && ../cudd/configure --prefix=$CUDD CFLAGS=-O3 && make -j16 && make install
cc -O2 -std=c11 -I$CUDD/include scripts/analysis/ftp-reference/ftp_cudd.c -o $WORK/ftp_cudd $CUDD/lib/libcudd.a -lm
```

Run it in a scratch directory. The `.bdd` dump of a large model is tens of MB.

```bash
python3 -I scripts/analysis/ftp-reference/ftp_cudd.py input/ftp/synthetic/name_500_06.ftp \
    --bin $WORK/ftp_cudd --workdir $WORK/run --reorder symm-sift --max-mem-mb 30000 --json $WORK/run/p.json
# reuse an existing dump (exact evaluation only)
python3 -I scripts/analysis/ftp-reference/ftp_cudd.py MODEL.ftp --bin $WORK/ftp_cudd --workdir $WORK/run --eval-only
# BDD against direct tree evaluation: random points, minimal true points and all their single flips
python3 -I scripts/analysis/ftp-reference/ftp_cudd_check.py MODEL.ftp $WORK/run TARGET --samples 20000 --flip-probes 100
```

| option | effect |
|---|---|
| `--targets T1,T3` | targets (default: `PROCESS`, else the roots) |
| `--order dfs\|file` | initial variable order: DFS first occurrence (default), or `IMPORT` order |
| `--reorder` | `sift`, `sift-converge`, `group-sift`, `group-sift-converge`, `symm-sift` or `none` (CUDD autodyn) |
| `--max-mem-mb` | CUDD memory limit (`Cudd_SetMaxMemory`); a failing operation stops the run |
| `--final-reduce` | one extra `CUDD_REORDER_SIFT_CONVERGE` pass after the build |

Semantics:

- `+` is OR, `*` is AND, and an integer k is at-least-k-of-n, built by the DP `a[j] = ite(x, a[j-1], a[j])`.
- `-NAME` is a plain Boolean NOT.
- Basic events are independent with the `IMPORT` values, read exactly from their decimal strings.
- Without `PROCESS`, the targets are the roots.
- Implicit import blocks (no `IMPORT` header, as in `conformance/header_limit`) are accepted.

Checks run:

- `conformance/k_of_n_continuation` TOP 0.63729810322530873721…, `process_multi_target` T1 0.622, T2 0.464, T3
  0.098, and `header_limit` TOP 0.224 all match `expected.json` to within 2e-16 relative. The expected values are
  double computations, so T3 is stored as 0.09800000000000002.
- On `name_500_06`, `ftp_cudd_check.py` found 0 mismatches in 20,000 random points and 100,000 single-flip probes
  around 100 minimal cut sets. The `--order file` BDD gave 0 mismatches in 5,000 random points and 30,000 probes.

Measured on `name_500_06` (391 gates, 500 events, 48-core Linux host, one thread each):

| reorder | result |
|---|---|
| `symm-sift` | 696 s build (660 s of it reordering, 29 reorderings), 767,161 nodes, peak 2.52 M live nodes, CUDD memory 404 MB |
| `symm-sift`, `--order file` | 401 s build (396 s reordering, 34 reorderings), 710,572 nodes, peak 1.95 M live nodes, CUDD memory 386 MB; same exact rational |
| `sift` | stopped by a 45 min timeout after gate 387/391 (2231 s, 1.50 M nodes, peak 3.48 M live nodes); slower than `symm-sift` at every gate |
| `sift-converge` | gate 370/391 after 1090 s; stopped at 41 min |

Exact `P(root80000)` = 9.743884454277023434299479904857e-12. An independent quantification of the same file at
truncation 1e-28 gives 1.444062e-11 for both the rare-event sum and the min-cut upper bound over its minimal cut
sets, 1.48 times the exact value.

- Both are upper bounds for a coherent tree with independent events.
- They agree with each other because every cut set has a small probability (all sampled minimal cut sets have 10
  events).
- That agreement says nothing about the overlap between cut sets that share events, which neither bound accounts
  for. The size of that overlap term has not been measured separately.

The exact rational evaluation of the dump needs about 0.9 GB RSS and 10 s.

Limits:

- The build is single-threaded. Reordering takes most of the time.
- Progress lines print only when a gate finishes, so one large AND/OR fold can be silent for minutes.
- Only the final BDD is dumped, so a run stopped by a timeout leaves no partial result.
