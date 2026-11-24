# Lab

Experiments that `develop` has not committed to. The rules are in
[docs/planning.md §6](../docs/planning.md#6-the-lab); in short:

- **One directory per experiment**, `lab/<topic>/`, holding:
  - `README.md`: the charter;
  - sources and tests;
  - `FINDINGS.md`: written when the experiment ends.
- **Imports go one way.** Lab code may use core code. Core code never includes or imports lab code; the
  `scope` check fails it, and no label accepts it.
- **Lab work changes only `lab/`.** A core change an experiment needs is a separate product issue, judged on
  product terms, or it waits for graduation.
- **Never in shipped artifacts.** Lab code is never in wheels, images or the default `fzb`, and never changes
  defaults, formats, `toolchain.env` or required CI.
- **At most two experiments in progress.** Each runs at most two weeks, extended at most once. At the end it
  graduates, is extended, or is archived: tag `lab-archive/<topic>`, delete the directory, keep `FINDINGS.md`
  content on the issue.

## Charter (`lab/<topic>/README.md`)

```markdown
# <topic>

Issue: #N (milestone Lab, label lab:<topic>) · Owner: @who · Started: YYYY-MM-DD · Time box: YYYY-MM-DD

## Hypothesis
One sentence that can be false.

## Measurement
What is measured, on which data (benchmark dataset v1, conformance fixtures, ...), and with which command.

## Kill criterion
The result that ends the experiment.

## Graduation needs
RFC (if it changes semantics or a format), product milestone, the default it does not change, the lab copy
that gets deleted.
```

## Build

There is no lab code yet. The first experiment to move in adds `lab/CMakeLists.txt`, included from the top-level
`CMakeLists.txt` only under `option(FUZZYBITS_LAB "Build lab experiments" OFF)`. It also adds a non-required CI
job that builds with `-DFUZZYBITS_LAB=ON`, so lab code keeps compiling against `develop` without blocking it.
