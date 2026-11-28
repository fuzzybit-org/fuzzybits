"""Aggregate GPU busy time, launch gaps and CUDA API time from an nsys SQLite export (no names printed
except kernel classes).

Tracing adds a cost to every launch: on a run of 81k short kernels the traced span was 2.3x the untraced
wall time, so "busy%" understates occupancy. Compare the kernel total with an untraced run instead.
"""
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
strings = dict(db.execute("SELECT id, value FROM StringIds"))
kern = db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
api = db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start").fetchall()
if not kern:
    print("no kernels"); raise SystemExit
first, last = kern[0][0], max(k[1] for k in kern)
busy, cur_s, cur_e = 0, kern[0][0], kern[0][1]
for s, e, _ in kern[1:]:
    if s > cur_e:
        busy += cur_e - cur_s; cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
gaps = [max(0, kern[i + 1][0] - kern[i][1]) for i in range(len(kern) - 1)]
classes = {}
for s, e, n in kern:
    name = strings.get(n, "?")
    sampler = "basic_event" in name or "kernel::sampler::" in name or "6kernel7sampler" in name
    c = "basic_event" if sampler else "tally" if "tally" in name else "gate/other"
    t = classes.setdefault(c, [0, 0]); t[0] += 1; t[1] += e - s
apis = {}
for s, e, n in api:
    name = strings.get(n, "?").split("_v")[0]
    t = apis.setdefault(name, [0, 0]); t[0] += 1; t[1] += e - s
ms = 1e-6
print(f"kernels {len(kern)}  span {(last - first) * ms:.1f} ms  busy {busy * ms:.1f} ms  busy% {100 * busy / (last - first):.1f}")
print(f"gap median {statistics.median(gaps) / 1e3:.1f} us  p90 {sorted(gaps)[int(0.9 * len(gaps))] / 1e3:.1f} us  "
      f"mean kernel {busy / len(kern) / 1e3:.1f} us")
for c, (n, t) in sorted(classes.items()):
    print(f"  {c:12s} n={n:6d} total {t * ms:9.1f} ms  mean {t / n / 1e3:8.1f} us")
for name, (n, t) in sorted(apis.items(), key=lambda kv: -kv[1][1])[:6]:
    print(f"  api {name:28s} n={n:7d} total {t * ms:9.1f} ms  mean {t / n / 1e3:7.1f} us")
