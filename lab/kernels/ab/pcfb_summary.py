# Summarize pcfb_ab.py JSON files: per target and model, head/base sample-time ratio (geometric mean over
# rounds, 95% t interval), medians, and whether base and head hits agree in every round (same seed, trials).
import json, math, statistics, sys
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
hits_by_model = {}
for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"\n## {d['host']} {d['target']} trials={d['trials']}")
    print("| model | base med s | head med s | head/base | 95% CI | rounds | hits base = head |")
    print("|---|---:|---:|---:|---|---:|---|")
    for model, rows in d["models"].items():
        ok = [r for r in rows if r["base"].get("sample_s") and r["head"].get("sample_s")]
        logs = [math.log(r["head"]["sample_s"] / r["base"]["sample_s"]) for r in ok]
        same = all(r["base"].get("hits") == r["head"].get("hits") for r in ok)
        hits = sorted({(r["base"].get("n"), r["base"].get("hits")) for r in ok} |
                      {(r["head"].get("n"), r["head"].get("hits")) for r in ok})
        hits_by_model.setdefault(model, {})[f"{d['host']}:{d['target']}"] = hits
        if len(logs) >= 2:
            m, sd = statistics.mean(logs), statistics.stdev(logs)
            half = T_975.get(len(logs) - 1, 2.0) * sd / math.sqrt(len(logs))
            ci = f"[{math.exp(m - half):.3f}, {math.exp(m + half):.3f}]"
        else:
            m, ci = (logs[0] if logs else float("nan")), ""
        bm = statistics.median(r["base"]["sample_s"] for r in ok) if ok else float("nan")
        hm = statistics.median(r["head"]["sample_s"] for r in ok) if ok else float("nan")
        fails = len(rows) - len(ok)
        print(f"| {model} | {bm:.4f} | {hm:.4f} | {math.exp(m):.3f} | {ci} | {len(ok)}{f' ({fails} failed)' if fails else ''} | "
              f"{'yes' if same else 'NO'} |")
print("\n## hits (n, hits) per model across targets")
for model, per in hits_by_model.items():
    print(model, per)
