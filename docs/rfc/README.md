# RFCs

An RFC fixes semantics, a format or a public contract before code depends on it. Policy:
[planning.md §5](../planning.md#5-defaults-formats-rfcs-and-benchmarks).

## Header

```markdown
# RFC: <title>

Status: proposed | accepted | implemented | rejected | withdrawn
Issue: #N
Accepted: YYYY-MM-DD by @who          (once accepted)
```

The health report reads the `Status:` line and the issue numbers on the `Issue:` line.

## Lifecycle

| Status | Means | Code allowed |
|---|---|---|
| `proposed` | under discussion | `lab/` only, behind an opt-in; no core change may be justified by it alone |
| `accepted` | the maintainer agreed, recorded in the header | product issues in a milestone; format changes may merge |
| `implemented` | the product follows the RFC; later changes need an amendment | as for any product code |
| `rejected` / `withdrawn` | final; the document stays as a record | none |

- **Acceptance is a separate commit or PR** that changes only the header. Merging the RFC document does not accept
  it.
- **An amendment** is a new `proposed` section. It is accepted the same way.

## Index

| RFC | Status | Notes |
|---|---|---|
| [monotone-hull-queries.md](monotone-hull-queries.md) | proposed | partly implemented on `develop`; frozen until accepted or withdrawn (ROADMAP.md, Lab) |
