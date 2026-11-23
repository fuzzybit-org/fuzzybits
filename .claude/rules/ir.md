---
paths:
  - "src/ir/**"
---
# IR (`src/ir`)

- `ir::pra` and `ir::pc` are SCRAM-free libraries. Their contracts are in
  `docs/IR-PRA-REQUIREMENTS.md` and `docs/IR-PC-REQUIREMENTS.md`, cited by section (e.g. "§12").
  Read the relevant section before changing a type, invariant or diagnostic.
- `ir::pc` stays a pure computation graph. Domain metadata belongs in `ir::pra` or in the
  projection (`docs/pra-pc-ir-architecture.md`).
- Passes (`src/ir/pc/passes`) return a new circuit plus a transform report; `docs/ir-pc-passes.md`
  lists the passes and compilation levels. Update it when a pass or level changes.
