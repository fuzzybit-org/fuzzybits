# Changelog

## Unreleased

### Breaking changes

- Basic events are sampled by a bit-sliced Philox sampler with exact 64-bit thresholds (#164, #17).
  A trial is true iff its 64-bit uniform is below `floor(p · 2^64)`, compared most significant bit
  first across the 64 trials of a word, so a word costs about 4 Philox blocks instead of 16.
  - Every `p >= 2^-64` is sampled at a nonzero rate. The 32-bit sampler turned `p < 2^-32` (about
    2.3e-10) into 0 and rounded other probabilities down by up to 2^-32.
  - The random stream changes: estimates for a given model and trial count differ from earlier
    releases within sampling error. There is no option to restore the old stream; earlier releases
    reproduce earlier results.
  - The unused SFC64, WyRand and xorshift128 headers in `src/mc/prng` are removed.
- The random stream changes again: a basic event's trials are drawn at the Philox counter (node, device, run-wide
  bitpack index), keyed by the run's seed, instead of at (node, position in its kernel block, bitpack in the pass,
  pass). A trial's bits no longer depend on how the layers group basic events into kernels or on the pass size the
  device's memory allows. Estimates for a given model and trial count differ from earlier releases within sampling
  error; earlier releases reproduce earlier results. The seed is 372, the default of every entry point, until
  `--seed` accepts other values (#65).
- Common-cause failure groups are expanded by default in `fzb compile`, `fuzzybits.compile`, the
  native `compile_pcfb` / `run_probability` bindings and `fuzzybits-server`. Results change for
  models that define CCF groups and were compiled without `--ccf`.
  - Disable it with `--no-ccf` (`fzb compile`), `ccf=False` (Python) or `MCSCRAM_CCF=false` (server).
  - The choice is recorded in the `compile.options` (`--no-ccf`) and `compile.ccf` metadata of the `.pcfb`.
  - `fzb compile --ccf`, the earlier opt-in, is gone (an unknown option, #262). `scram-cli` keeps its own
    `--ccf`.
- Python `compile(watch="a,b")` no longer splits the string on `,`, `;` or `:`: pass a list.
- The server's compile cache key now includes the effective `ccf` setting, so blobs cached
  without expansion are not reused.
- FTP names are compared without regard to ASCII case (#240): `g1` and `G1` are one element, named
  as its gate line or first import record spells it, with an `io.ftp.name-case` warning for each
  element written in another case. A project that used two such names for different elements now
  reads them as one (or fails with `io.ftp.duplicate-gate`). To keep them apart, rename one.
- In an FTP tree section a line `NAME op ...` now always defines a gate, and any other line
  continues the open gate, indented or not. An indented continuation line whose second operand is
  an event named with digits only (`  A 7 B`) used to add operands; it now defines an at-least gate.
  To keep it a continuation, move that operand off the second position.
- FTP targets are probability queries by default (#244). `fzb compile` and `fuzzybits.compile`
  used to write a monotone-hull query for every FTP target, which `fzb run` refuses, because the
  reader recorded `ftp.negation = delete_term` for every project and the driver took the query type
  from it. The query type is now requested, never read from the model:
  - Without `--kind` (or `kinds` in the config document), every target is a `PROBABILITY` query and
    the file is pcfb minor 1. Results change only where a target's cone has a complement.
  - To keep the previous default, pass `--kind hull` (`kind="hull"`, or `"kinds": "hull"` in the config
    document). `--kind probability --kind hull` writes both queries.
  - The reader no longer records the `ftp.negation` attribute. `fzb inspect` and
    `fuzzybits.read_project` drop it from `conventions`, and the `compile.ftp.default-negation`
    note is gone.
- FTP `I` values are initiating-event frequencies (#243). By default a target whose cone holds initiating
  events becomes one conditioned query per initiator (`<target>|<initiator>`, that initiator TRUE and the
  others FALSE), and the file gains a query group per target: its frequency `F = Σ f_i · P(target | I_i)`.
  - `--initiator-values per-year|per-month|per-week|per-day|per-hour` names the unit, `per-year` by default.
    The unit is recorded in each group and printed with its results.
  - A target that may hold with every initiating event FALSE is an error (`compile.pra_to_pc.mixed-units`).
  - The initiator rule no longer applies by default: with frequencies `--hull-max-initiators N` is an error
    (`none` is accepted).
  - To keep the previous reading (values up to 1 as probabilities, above 1 rejected, the initiator rule),
    pass `--initiator-values probability` (`initiator_values="probability"`, or `"initiator_values":
    "probability"` in the config document).
  - The reader marks every `I` value as a frequency, and the `io.ftp.initiator-frequency` warning is gone.
    `compile.pra_to_pc.frequency-as-probability` is now a note.
- The FTP initiator rule (`--hull-max-initiators N`) applies to hull queries only and has no default (#262).
  With `--initiator-values probability`, probability queries used to be ANDed with
  `not atleast(2; initiating events)` by default, because the reader recorded `ftp.max_initiators_per_cutset = 1`
  for every project. A probability query is now the probability of its target: on the `initiator_rule` fixture
  `TOP` is 0.45, not 0.288.
  - The reader no longer records `ftp.max_initiators_per_cutset`, and its option
    `io.ftp.max_initiators_per_cutset` is gone. `fzb inspect` and `fuzzybits.read_project` drop it from
    `conventions`. The note `compile.ftp.default-initiator-rule` is gone.
  - `--hull-max-initiators N` (or `hull_max_initiators` in Python, `"hull": {"max_initiators": N}` in the config
    document) without a hull query is `compile.ftp.invalid-option`. To apply the cut-set rule, request the hull:
    `--kind hull --hull-max-initiators 1`.
  - A target with both kinds and the rule gets two queries on two tensors: the probability query on the target,
    the hull query on the target under the rule.
  - The `.pcfb` metadata records `compile.query.<i>.literal_mode` on hull queries only, and
    `compile.query.<i>.max_initiators_per_cutset` only on queries whose target holds the delete term. The warning
    `compile.pra_to_pc.complemented-initiator` is emitted for hull targets only.
  - There is no opt-out: P(target and at most N initiating events) is no longer a query fzb computes.
- The FTP option `--complement-events atom|negation` is gone (#262), from `fzb compile` and `fzb inspect`, the
  native and the Python `fzb`, `fuzzybits.compile(complement_events=...)`, `fuzzybits.read_project(complement_events=...)`,
  the config document key `complement_events` and the reader option `io.ftp.complement_events`. The FTP reader
  always marks a complemented basic event (`-EVENT`) as a literal, as `atom` did; `-GATE` is never marked. The mark
  does not change the probability; only hull queries read it.
  - Passing the flag is an unknown-option error, the Python keyword a `TypeError`, and the config key
    `compile.ftp.invalid-config`. The `compile.options` metadata and the `--report` options no longer have
    `complement_events`, and `fzb inspect` always counts every `-EVENT` in `literal_operands`.
  - Replacement for `negation`: `--hull-literals plain` (`hull_literals="plain"`, or `"hull": {"literals":
    "plain"}` in the config document) computes the same plain hull. It records
    `compile.query.<i>.literal_mode = plain`, and it warns `compile.pra_to_pc.complement-under-negation` (and, with
    the initiator rule, `compile.pra_to_pc.complemented-initiator`) for marked events where unmarked ones warned
    nothing.
  - `atom` needs no replacement: it was the default.
- The FTP query options have one name per concept (#262). Replacements, flags first, then Python and the config
  document:
  - `--negation boolean|hull|both` becomes `--kind probability|hull`, repeated or comma-separated for both
    (`kind=` a string or a list; `"kinds"`). `both` and `monotone_hull` are not accepted.
  - `--literals atom_split|omega_l|negation` becomes `--hull-literals split|fixed|plain` (`hull_literals=`;
    `"hull": {"literals": ...}`). It is an error when no target asks for the hull.
  - `--max-initiators-per-cutset N|none` becomes `--hull-max-initiators N|none` (`hull_max_initiators=`;
    `"hull": {"max_initiators": ...}`).
  - `--initiator-values frequency --initiator-unit UNIT` becomes `--initiator-values UNIT`; `--initiator-values
    probability` stays. `--initiator-unit`, `initiator_unit=` and `"initiator_unit"` are gone.
  - Config targets take `kinds` and `hull` instead of `negation` and `literals`; the removed keys are
    `compile.ftp.invalid-config`.
  - `--no-ccf` with FTP input is an error (`ccf=False` a `ValueError`): FTP projects have no common-cause
    groups. `-c` help names both pipelines, and the FTP error for `-c 2` to `-c 8` says those are the SCRAM
    preprocessing levels of MEF input.
  - A target asking for both kinds has one entry per query in `compile.options` and in the report (`kind`,
    `hull_literals`, `hull_max_initiators`) instead of a `negation` array and `literals`; the top level has
    `kinds`, `hull` and `initiator_values` instead of `max_initiators_per_cutset` and `initiator_unit`.
  - The `.pcfb` metadata drops `compile.query.<i>.negation` (the kind is `Query.kind`), adds
    `compile.query.<i>.target` (the target a query observes, without `|<initiator>`), and spells
    `compile.query.<i>.literal_mode` `split`, `fixed` or `plain`. `fzb inspect --json` and `fuzzybits.inspect`
    list `query_kinds` (`probability` or `hull`), and the text summary prints them.

### Changes

- `fzb run`, `fuzzybits.run` and the server evaluate at-least, at-most and exactly gates correctly when a
  trial's count of true inputs reaches 256 (#279). The kernel counted in 8 bits, so such gates gave wrong
  results with no diagnostic (100 of 300 inputs at p = 0.9: 0.004 instead of 1). Gates under 256 inputs give
  the same results as before for a given seed.
- Runs on HIP no longer fail with `hipMallocManaged() failed` when a layer's AND or OR gates are all constants
  (#277).
- The gate kernels evaluate one bitpack per work-item (CPUs: 8 bitpacks as one vector), one kernel per
  connective and layer (#274, #275). k-of-n gates count bit-sliced instead of one trial per work-item, and at most
  k and exactly k no longer need NOT passes. In the kernel microbenchmark k-of-n gates are 5.5-56x faster on GPUs
  and 12-240x on CPUs, and AND, OR, XOR, NAND, NOR, NOT and NULL gates of up to 16 inputs 1.3-3x faster on NVIDIA
  GPUs, 5-117x on a Radeon Pro 5500M and 1.1-6.6x on CPUs. Gates whose inputs other gates of a small layer share
  can be slower than before: 64-input AND gates, 32 per layer, up to 1.4x on an RTX 3070, and 64-input OR gates,
  128 per layer, 1.8x on CPUs (#281).
- The execution plan and the kernel DAG name what computes a node by the native IR's op code (`ir::pc::op_code`)
  instead of SCRAM's connective. Partitions of at-least gates report `atleast` instead of `k/n`. Constant gates
  report `ones` or `zeros`, and no kernel computes them: their bitpacks are written once, as for basic events of
  constant law. Results are the same for a given seed.
- The Monte Carlo queue is in order, and kernels are submitted without dependency events, so AdaptiveCpp submits
  each kernel directly to the device instead of through its DAG scheduler (#286). On a model of about 88,000
  gates at the default device-memory budget, `fzb run` is about 2x faster on a Tesla V100S and 1.7-1.8x on an
  RTX 3070, and unchanged on CPUs. Results are the same for a given seed.
- `fzb serve` starts `fuzzybits-server` from the Python `fzb` (#203). Install both with
  `pip install 'fuzzybits[server]'`, a new extra pinned to the same version, then run, for example,
  `fzb serve --data-dir ./models --open`. Its arguments go unchanged to the server's parser, so it
  takes the `fuzzybits-server` flags (`fzb serve --help`). Without the server it exits 1 with the
  install command, and it warns if the two versions differ. The native `fzb` lists `serve` as
  Python-only and refuses it. `fuzzybits_server.app.main` takes `argv` and `prog`.
- On Windows, a fuzzybits run (the `fzb` CLI, or Python) no longer terminates with `0xC0000409`
  when another fuzzybits process started from the same directory has loaded the same OpenMP kernel.
  AdaptiveCpp is patched to give each process its own kernel library file.
- `scram-cli` no longer crashes on exit on Windows (`0xC0000005` after a complete run). It
  shut libxml2 down before freeing its input schema validator.
- The Python `fzb` writes UTF-8 to stdout and stderr. On Windows, `fzb --help` and `--json`
  output piped or redirected to a cp1252 stream raised `UnicodeEncodeError`.
- `fzb run` and `fuzzybits.run` print the query groups of a file after its queries (the weighted sum
  of the estimates and a conservative interval), and `fzb inspect --exact` and `fuzzybits.exact_groups`
  their exact values. `--exposure DURATION` (`1y`, `1mo`, `720h`; `exposure=` in Python) adds the
  probability of at least one occurrence over the period, `1 − e^{−F·τ}`. `fzb export --to nombrilo`
  carries the groups in `fuzzybits.pcfb.query_groups`; `--strict` drops them with a warning.
- pcfb is the git submodule `extern/pcfb` (pin: its gitlink; `toolchain.env` no longer pins it, so a
  pcfb bump does not rebuild the toolchain image). It is pinned to the query-group amendment of pcfb
  1.2 (`SubCircuit.query_groups`), on top of the literal-atom amendment. Run
  `git submodule update --init extern/pcfb` in existing checkouts.
- `fzb run` answers every query of a file whose queries are not all in the first query's cone, such as
  an FTP project with several targets; such files failed with `unknown event`.
- `fzb run`, `fuzzybits.run` and the server evaluate `ONES` / `ZEROS` and repeated operands as pcfb defines
  them (#259). They gave wrong results with no diagnostic:
  - a `ONES` that is a query target or is named read as FALSE: `fzb compile -c 0 --override NAME=1`, and at
    the default level every target that folds to TRUE, such as an OR over a basic event with p = 1;
  - an unnamed constant or a repeated operand changed the gate it fed, depending on its position among the
    operands: frequency-mode conditions at `-c 0` (`AND(A, IE, B)` read as `B`), or `AND(A, A, B)`.

  `scram-cli` read a top event that folds to TRUE as FALSE. `fzb compile` wrote an OpenPSA MEF top event that
  folds to FALSE as TRUE (`fzb inspect --exact` gave 1); recompile such files. A PDAG gate that holds the
  constant beside other arguments is now refused before sampling.
- AMD GPUs through AdaptiveCpp's HIP backend (`--backend hip`, `backend="hip"`; `auto` picks
  cuda, then hip, then omp) in the `generic` wheel and images. Needs the ROCm 7 user space and
  access to `/dev/kfd` and a render node (#228).
- `fzb compile` and `fuzzybits.compile` take `--target` and `--watch` repeatedly, and
  `--target @FILE` / `--watch @FILE` read one name per line (the Python API takes a name or a
  sequence). A file that is missing, is not a regular file or lists no names is an error.
- OpenPSA/MEF and nombrilo input take one target; several targets, a repeated name, and
  `--watch` on nombrilo input are errors instead of being ignored.
- The FTP reader accepts the files projects export (#240; rules in `docs/ftp-input.md`, Grammar):
  - names hold any punctuation, such as `%IE`, `G~1`, `SYS.A`, `PUMP#1`, `V(2)` or `$V1`;
  - keywords and the `I` flag in any letter case (`import`, `Endtree`, `i`);
  - `;` comments, values written `1.`, `.5`, `1.E-02` or `1.0D-01`, and continuation lines without
    indentation;
  - the value `-1` as probability 1 (`io.ftp.minus-one-value` warning);
  - `MAXSIZE n`, recorded as `ftp.maxsize` and in `fzb inspect` conventions (`MAXSIZE`), never applied;
  - a UTF-8 byte-order mark, and control bytes as blanks.
- Tokens after `ENDTREE`, `IMPORT`, `LIMIT x` and `MAXSIZE n` are a warning (`io.ftp.ignored-tokens`)
  instead of an error.
