import json, sys, collections
D = json.load(open(sys.argv[1])); th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
rows = {r['ep']: r for r in json.load(open(sys.argv[1].replace('.json', '_rows.json')))}
G = {int(k): v for k, v in json.load(open('diffgt.json')).items()}
idx = {r['ep']: r for r in json.load(open('index.json'))}
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
def ctr(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
def inside(p, b): return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
def match(box, gt): return iou(box, gt) >= 0.3 or (inside(ctr(box), gt) and inside(ctr(gt), box))
def good_gt(g):
    s = g.get('start_box')
    if not s: return False
    area = (s[2] - s[0]) * (s[3] - s[1])
    return area < 20000 and not (g.get('end_box') and iou(s, g['end_box']) > 0) 
res = {}
for t in (1, 2):
    E = [e for e in G if rows[e]['t'] == t and good_gt(G[e])]
    c = collections.Counter(); conf_t = collections.defaultdict(collections.Counter); dconf = collections.Counter()
    for e in E:
        r = rows[e]; gt = G[e]['start_box']; dets = [d for d in D['first/%04d' % e] if d[5] >= th]
        m = [d for d in dets if match(d[1:5], gt)]
        c['localized (any class)'] += bool(m)
        tb = [d for d in dets if d[0] == r['tgt']]
        if not tb: c['target class: none'] += 1
        elif match(max(tb, key=lambda d: d[5])[1:5], gt): c['target class: on the right object'] += 1
        else: c['target class: on another object'] += 1
        conf_t[r['tgt']][max(m, key=lambda d: d[5])[0] if m else '(no box)'] += 1
        if 'end_box' in G[e]:
            p = ctr(G[e]['end_box']); c['dest gt'] += 1
            db = [d for d in dets if d[0] == r['dest']]
            if db and inside(p, max(db, key=lambda d: d[5])[1:5]): c['dest class box holds the placed object'] += 1
            hold = [d for d in dets if d[0].startswith(('plate_', 'organizer_')) and inside(p, d[1:5])]
            dconf[(r['dest'], max(hold, key=lambda d: d[5])[0] if hold else '(none)')] += 1
    n = len(E)
    print(f'== task {t}: {n} episodes with a clean start box (conf >= {th})')
    for k in ('localized (any class)', 'target class: on the right object', 'target class: on another object', 'target class: none'):
        print(f'  {k:40s} {c[k]:4d}  {100 * c[k] / n:5.1f}%')
    print(f"  {'dest class box holds the placed object':40s} {c['dest class box holds the placed object']:4d} / {c['dest gt']}  {100 * c['dest class box holds the placed object'] / max(1, c['dest gt']):5.1f}%")
    print('  target -> what the detector calls it (box on the target):')
    for k, v in sorted(conf_t.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(v.values()); print(f'    {k:18s} n={tot:2d} right={v[k]:2d}  ', ', '.join(f'{a}:{b}' for a, b in v.most_common(4)))
    print('  destination -> class of the box holding the placed object:', sorted(dconf.items(), key=lambda kv: -kv[1])[:20])
    res[t] = dict(n=n, **c)
json.dump(res, open(sys.argv[1].replace('.json', f'_eval{th}.json'), 'w'))
