# Summarize the endpoint experiments: per configuration (--pad, --order, --pool) and shape n:n, the cardinality
# kernel's time, the AND kernel's time, and AND/cardinality.
import collections, sys

for path in sys.argv[1:]:
    dev = None
    table = collections.OrderedDict()
    for line in open(path):
        cfg, _, rest = line.partition('\t')
        if rest.startswith('# device:'):
            dev = rest[len('# device: '):].split('|')[0].strip()
        elif rest.startswith('R\t') and not rest.startswith('R\tshape'):
            f = rest.rstrip('\n').split('\t')
            table.setdefault(cfg, {})[f[1]] = (float(f[7]), float(f[10]), f[13].split()[0])
    shapes = sorted({s for v in table.values() for s in v}, key=lambda s: int(s.split(':')[0]))
    print(f"\n## {dev} ({path.split('/')[-1]}): cardinality ms / AND ms (AND/card)")
    print("| config | " + " | ".join(shapes) + " |")
    print("|---|" + "---|" * len(shapes))
    for cfg, v in table.items():
        cells = []
        for s in shapes:
            if s in v:
                c, a, ok = v[s]
                cells.append(f"{c:.3f} / {a:.3f} ({a / c:.2f})" + ('' if ok == 'ok' else ' WRONG'))
            else:
                cells.append('')
        print(f"| {cfg} | " + " | ".join(cells) + " |")
