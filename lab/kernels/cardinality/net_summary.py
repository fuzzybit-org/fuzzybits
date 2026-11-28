# Summarize bench_network output: per device and shape, the cardinality kernel's time, the best network's time,
# measured network/cardinality against the naive (bytes) prediction, and whether the hypothesis held:
# networks (1 < k < n) slower than the cardinality kernel; at k = 1 or n the plain OR/AND at least as fast.
import sys

for path in sys.argv[1:]:
    dev = None
    rows = {}
    for line in open(path):
        if line.startswith('# device:'):
            dev = line[len('# device: '):].split('|')[0].strip()
        elif line.startswith('R\t') and not line.startswith('R\tshape'):
            f = line.rstrip('\n').split('\t')
            if 'skipped' in f[-1]:
                continue
            shape, n, k, form, G = f[1], int(f[2]), int(f[3]), f[4], int(f[5])
            card, t1, t2, net, ratio, naive, ok = (float(f[7]), float(f[8]), float(f[9]), float(f[10]), float(f[11]),
                                                   float(f[12]), f[13].split()[0])
            rows.setdefault(shape, []).append(dict(n=n, k=k, form=form, G=G, card=card, t1=t1, t2=t2, net=net,
                                                   ratio=ratio, naive=naive, ok=ok))
    print(f"\n## {dev} ({path.split('/')[-1]})")
    print("| shape | form | G | card ms | net ms (1st + 2nd) | net/card | naive | measured/naive | hypothesis |")
    print("|---|---|---:|---:|---|---:|---:|---:|---|")
    for shape, rs in rows.items():
        for r in rs:
            one = r['form'] == 'one'
            held = (r['ratio'] <= 1.0) if one else (r['ratio'] > 1.0)
            verdict = ('held' if held else 'FAILED') + ('' if r['ok'] == 'ok' else ' (WRONG bits)')
            net = f"{r['net']:.3f}" + ('' if one else f" ({r['t1']:.3f} + {r['t2']:.3f})")
            print(f"| {shape} | {r['form']} | {r['G']} | {r['card']:.3f} | {net} | {r['ratio']:.2f} | {r['naive']:.2f} | "
                  f"{r['ratio'] / r['naive']:.2f} | {verdict} |")
