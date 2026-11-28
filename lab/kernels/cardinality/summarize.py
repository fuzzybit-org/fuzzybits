# Summarize bench_card output: per device, per shape, baseline (prod or lowered) vs each new variant.
import sys, collections
for path in sys.argv[1:]:
    dev = None
    rows = collections.OrderedDict()
    def flush():
        if not rows: return
        variants = []
        for shp, vs in rows.items():
            for v in vs:
                if v not in variants and not v.startswith(('prod', 'lowered')): variants.append(v)
        print(f"\n## {dev}")
        print("| shape | G | baseline ms | " + " | ".join(variants) + " | best× |")
        print("|---|---:|---:|" + "---:|" * len(variants) + "---:|")
        for shp, vs in rows.items():
            base = [(k, v) for k, v in vs.items() if k.startswith(('prod', 'lowered'))]
            bms = base[0][1][0] if base else None
            bok = base[0][1][1] if base else 'ok'
            cells = []
            best = None
            for v in variants:
                if v in vs:
                    ms, ok, gb, _ = vs[v]
                    cells.append(f"{ms:.3f}" + ("" if ok == 'ok' else " WRONG"))
                    if ok == 'ok': best = ms if best is None else min(best, ms)
                else:
                    cells.append("")
            G = vs[next(iter(vs))][3]
            bstr = "" if bms is None else (f"{bms:.3f}" + ("" if bok == 'ok' else " WRONG"))
            sp = "" if (bms is None or best is None) else f"{bms / best:.1f}"
            print(f"| {shp} | {G} | {bstr} | " + " | ".join(cells) + f" | {sp} |")
        rows.clear()
    for line in open(path):
        if line.startswith('# device:'):
            flush(); dev = line[len('# device: '):].split('|')[0].strip() + ' (' + path.split('/')[-1] + ')'
        elif line.startswith('R\t') and not line.startswith('R\tshape'):
            f = line.rstrip('\n').split('\t')
            shp, G, var, ms, gb, ok = f[1], f[5], f[7], float(f[8]), float(f[9]), f[10]
            rows.setdefault(shp, collections.OrderedDict())[var] = (ms, ok, gb, G)
    flush()
