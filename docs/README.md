# Documentation

These pages state what CI or a reproducible command proves. Plans live in
[ROADMAP.md](../ROADMAP.md) and in issues. How to contribute: [CONTRIBUTING.md](../CONTRIBUTING.md).
How work is planned, scoped and checked: [planning.md](planning.md).

## Using fuzzybits

| Page | Covers |
|---|---|
| [python.md](python.md) | Python package, `fzb` on the wheel, API, progress logs, building wheels; where code lives (C++ engine, thin Python) |
| [ftp-input.md](ftp-input.md) | CAFTA FTP projects through `fzb compile` |
| [nombrilo.md](nombrilo.md) | nombrilo circuit files ↔ `.pcfb` |
| [images.md](images.md) | Runtime container images: tags, pulling, running |
| [benchmarks.md](benchmarks.md) | Benchmark dataset, harness, PR gate, published history |

## Internals

| Page | Covers |
|---|---|
| [IR-PRA-REQUIREMENTS.md](IR-PRA-REQUIREMENTS.md) | Contract for `ir::pra`, the PRA domain model |
| [IR-PC-REQUIREMENTS.md](IR-PC-REQUIREMENTS.md) | Contract for `ir::pc`, the probabilistic circuit IR |
| [pra-pc-ir-architecture.md](pra-pc-ir-architecture.md) | Design analysis: PRA → PC lowering and projection |
| [ir-pc-passes.md](ir-pc-passes.md) | `ir::pc` passes and compilation levels |
| [pc-pcfb.md](pc-pcfb.md) | `ir::pc` ↔ `.pcfb` writer and reader |
| [rfc/README.md](rfc/README.md) | RFC lifecycle (proposed, accepted, implemented) and index |
| [rfc/monotone-hull-queries.md](rfc/monotone-hull-queries.md) | RFC (proposed): monotone-hull probability queries |
| [architecture.md](architecture.md) | Monte Carlo engine design (from the mcSCRAM era; partly dated, #112) |

## Planning

| Page | Covers |
|---|---|
| [planning.md](planning.md) | Goals, milestones, triage, PR scope rules, the lab, drift detection, agent rules |
| [../lab/README.md](../lab/README.md) | Lab experiments: charter, time box, graduation |

## Building, CI and releases

| Page | Covers |
|---|---|
| [../scripts/dev/README.md](../scripts/dev/README.md) | `ws.sh` remote build workspaces: the preferred way to build and test |
| [build-variants.md](build-variants.md) | CMake presets, toolchain pins, per-variant recipes, macOS local development |
| [ci-toolchain.md](ci-toolchain.md) | CI toolchain image, tags, GPU runner mapping |
| [cuda-builds.md](cuda-builds.md) | CUDA 12.8 images, build args, Nsight |
| [releasing.md](releasing.md) | Publishing `fuzzybits` and `fuzzybits-server` to PyPI |
