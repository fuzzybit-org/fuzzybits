"""Structure counts of a parsed FTP project (keys follow the private model cards)."""

from __future__ import annotations

import re
from collections import Counter

from .ftp import AND, ATLEAST, OR, Project

# 2^-32 is the engine sampler's Bernoulli threshold (#17).
SAMPLER_MIN_P = 2.0**-32
_NCNAME = re.compile(r"[A-Za-z_][A-Za-z0-9._-]*\Z")


def depth(project: Project) -> int | None:
    """Gates on the longest root-to-event path (an event alone has depth 0);
    ``None`` when the gate graph has a cycle."""
    gates = project.gates
    memo: dict[str, int] = {}
    for start in gates:
        if start in memo:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        on_path = {start}
        while stack:
            name, i = stack[-1]
            kids = gates[name].children
            if i < len(kids):
                stack[-1] = (name, i + 1)
                child = kids[i][0]
                if child in gates and child not in memo:
                    if child in on_path:
                        return None
                    on_path.add(child)
                    stack.append((child, 0))
                continue
            memo[name] = 1 + max((memo.get(c, 0) for c, _ in kids), default=0)
            on_path.discard(name)
            stack.pop()
    return max(memo.values(), default=0)


def structure_counts(project: Project) -> dict:
    """Counts used by fixtures and cards. They describe the files and do not
    depend on the duplicate-import policy: an event counts as I-tagged, or as
    having a value, when any of its import lines does."""
    gates = project.gates
    kinds = Counter(g.kind for g in gates.values())
    k_of_n = Counter(f"{g.k}/{len(g.children)}" for g in gates.values() if g.kind == ATLEAST)
    imported = {imp.name for imp in project.imports}
    leaves = {c for g in gates.values() for c, _ in g.children if c not in gates}
    events = (leaves | imported) - set(gates)
    values: dict[str, set[float]] = {}
    for imp in project.imports:
        if imp.name in events:
            values.setdefault(imp.name, set()).add(imp.value)

    def having(pred) -> set[str]:
        return {e for e, vs in values.items() if any(pred(v) for v in vs)}

    initiators = project.initiators() & events
    names = set(gates) | events
    return {
        "gates": len(gates),
        "gates_by_kind": {OR: kinds[OR], AND: kinds[AND], ATLEAST: kinds[ATLEAST]},
        "atleast_k_of_n": dict(sorted(k_of_n.items(), key=lambda kv: tuple(map(int, kv[0].split("/"))))),
        "edges": sum(len(g.children) for g in gates.values()),
        "single_input_gates": sum(1 for g in gates.values() if len(g.children) == 1),
        "max_fan_in": max((len(g.children) for g in gates.values()), default=0),
        "depth": depth(project),
        "basic_events": len(events),
        "import_lines": len(project.imports),
        "complemented_edges": sum(neg for g in gates.values() for _, neg in g.children),
        "complemented_edges_to_basic_events": sum(
            neg and c not in gates for g in gates.values() for c, neg in g.children
        ),
        "roots": project.roots(),
        "process_targets": list(project.process),
        "limit": project.limit,
        "initiator_tagged_events": len(initiators),
        "initiator_tagged_events_with_p_1": len(initiators & having(lambda p: p == 1.0)),
        "events_p_eq_1": len(having(lambda p: p == 1.0)),
        "events_p_eq_0": len(having(lambda p: p == 0.0)),
        "events_0_lt_p_lt_2^-32": len(having(lambda p: 0.0 < p < SAMPLER_MIN_P)),
        "names_not_valid_xml_ncname": sum(1 for n in names if not _NCNAME.match(n)),
        "duplicate_imports": {
            name: [
                {"value": o.value, "initiator": o.initiator, "file": o.location.file, "line": o.location.line}
                for o in occ
            ]
            for name, occ in project.duplicate_imports().items()
        },
    }
