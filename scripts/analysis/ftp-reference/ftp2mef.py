"""Minimal CAFTA FTP -> OpenPSA MEF converter for negation-free models (AND, OR, k-of-n; IMPORT values).

Usage: python3 ftp2mef.py IN.ftp OUT.xml
Refuses complemented operands: the point is an independent exact Boolean value, not delete-term semantics.
"""
import sys
from xml.sax.saxutils import quoteattr

src, dst = sys.argv[1], sys.argv[2]
gates, order, imports, process = {}, [], {}, []
section, current = "tree", None
for raw in open(src):
    line = raw.rstrip("\r\n")
    text = line.strip()
    if not text or text.startswith("*"):
        continue
    keyword = text.split()[0]
    if keyword == "ENDTREE":
        section = "after"; continue
    if keyword == "PROCESS":
        section = "process"; process += text.split()[1:]; continue
    if keyword == "IMPORT":
        section = "import"; continue
    if keyword == "LIMIT":
        section = "after"; continue
    if section == "tree":
        if line[0].isspace() and current:
            gates[current][1].extend(text.split()); continue
        parts = text.split()
        current = parts[0]
        gates[current] = [parts[1], parts[2:]]
        order.append(current)
    elif section == "process":
        process += text.split()
    elif section == "import":
        parts = text.split()
        imports[parts[1]] = float(parts[0])

assert len(process) == 1, process
for name, (_, ops) in gates.items():
    for op in ops:
        if op.startswith("-"):
            sys.exit(f"complemented operand {op} in {name}: not supported")
        if op not in gates and op not in imports:
            sys.exit(f"undefined {op}")

def ref(name):
    return f'<gate name={quoteattr(name)}/>' if name in gates else f'<basic-event name={quoteattr(name)}/>'

out = ['<?xml version="1.0"?>', "<opsa-mef>", '<define-fault-tree name="FT">']
for name in [process[0]] + [g for g in order if g != process[0]]:
    kind, ops = gates[name]
    # MEF requires 2 <= min < n for atleast, so 1-of-n is written as OR and n-of-n as AND.
    k = None if kind in "+*" else int(kind)
    if k is not None and not 1 <= k <= len(ops):
        sys.exit(f"{name}: k = {k} outside 1..{len(ops)}")
    if len(ops) == 1:
        # MEF AND/OR need two or more arguments; a single operand is a pass-through (null) gate.
        head, tail = "", ""
    elif kind == "+" or k == 1:
        head, tail = "<or>", "</or>"
    elif kind == "*" or k == len(ops):
        head, tail = "<and>", "</and>"
    else:
        head, tail = f'<atleast min="{k}">', "</atleast>"
    body = "".join(ref(op) for op in ops)
    out.append(f'<define-gate name={quoteattr(name)}>{head}{body}{tail}</define-gate>')
out.append("</define-fault-tree>")
out.append("<model-data>")
for name, value in imports.items():
    out.append(f'<define-basic-event name={quoteattr(name)}><float value="{value!r}"/></define-basic-event>')
out.append("</model-data></opsa-mef>")
open(dst, "w").write("\n".join(out) + "\n")
print(f"gates={len(gates)} events={len(imports)} top={process[0]}")
