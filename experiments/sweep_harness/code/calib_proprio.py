"""Calibrate the proprioceptive phase rules (no ball visibility) on the 216 official task-3 demos.

Only signals the live robot also gives the harness: the 12 arm joints and the 2 grippers (dataset units = live metres x 10).
For each demo: real_split.segment() boundaries (b1 tools held, b2 lift starts, b3 back from the pour pose), then
  - grasp->sweep: first frame both grippers sit in the holding band for HOLD_S, after each has opened once;
  - dump->home : pan-arm joint displacement from its pose at the dump start, d(t); fire when d has risen to D_MIN
    and fallen back by RETURN_FRAC of its peak.
Prints the detection error against the labels.
"""
import json, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '.')
import real_split as RS
V = sys.argv[1]
tb = pq.read_table(f'{V}/data/chunk-000/file-000.parquet', columns=['episode_index', 'frame_index', 'observation.state.arm.position',
                   'observation.state.effector.position', 'observation.state.ee.position'])
col = lambda k, n: tb.column(k).combine_chunks().flatten().to_numpy().reshape(-1, n)
EP = tb.column('episode_index').to_numpy(); FR = tb.column('frame_index').to_numpy()
Q, GR, EE = col('observation.state.arm.position', 12), col('observation.state.effector.position', 2), col('observation.state.ee.position', 14)
lab = {json.loads(l)['episode_index']: json.loads(l)['labels'][0]['custom_fields'][0]['value'] for l in open(f'{V}/meta/labels.jsonl')}
eps = [e for e, i in lab.items() if i.startswith('T3')]
HOLD_LO, HOLD_HI, OPEN = 0.15, 0.38, 0.38
res = []
for e in eps:
    ix = np.where(EP == e)[0]; ix = ix[np.argsort(FR[ix])]
    q, gr, ee = Q[ix], GR[ix], EE[ix]
    try: s = RS.segment(gr, ee)
    except ValueError: continue
    pan = 0 if s['pan_arm'] == 'left' else 1
    opened = np.maximum.accumulate(gr > OPEN, axis=0)
    held = (gr > HOLD_LO) & (gr < HOLD_HI) & opened
    both = held.all(1)
    run = np.convolve(both, np.ones(30), 'valid') >= 30          # 1 s stable at 30 fps
    g_det = int(np.argmax(run)) + 29 if run.any() else None
    qp = q[:, 6 * pan: 6 * pan + 6]
    dd = np.linalg.norm(qp - qp[s['b2']], axis=1)
    peak = np.maximum.accumulate(np.where(np.arange(len(dd)) >= s['b2'], dd, 0))
    res.append({'ep': e, 'b1': s['b1'], 'g_det': g_det, 'b2': s['b2'], 'b3': s['b3'], 'release': s['release'],
                'dpeak_b3': float(peak[s['b3']]), 'd_b3': float(dd[s['b3']]), 'dpeak_rel': float(peak[s['release']]),
                'dd': dd.tolist(), 'peak': peak.tolist(), 'T': s['T']})
print('demos', len(res))
g_err = [(r['g_det'] - r['b1']) / 30 for r in res if r['g_det'] is not None]
print('grasp->sweep detection - label (s): min %.2f median %.2f p95 %.2f max %.2f, missed %d' % (
    min(g_err), np.median(g_err), np.percentile(g_err, 95), max(g_err), sum(r['g_det'] is None for r in res)))
pk = np.array([r['dpeak_b3'] for r in res]); print('pan-arm displacement peak by b3 (rad): p5 %.2f median %.2f p95 %.2f' % tuple(np.percentile(pk, [5, 50, 95])))
best = None
for D in (0.3, 0.4, 0.5, 0.6, 0.8):
    for F in (0.1, 0.15, 0.2, 0.3):
        errs, early = [], 0
        for r in res:
            dd, peak = np.array(r['dd']), np.array(r['peak'])
            ok = (np.arange(len(dd)) > r['b2']) & (peak >= D) & (dd <= peak * (1 - F))
            if not ok.any(): errs.append(None); continue
            f = int(np.argmax(ok)); errs.append((f - r['b3']) / 30); early += f < r['b3'] - 15   # > 0.5 s before b3
        good = [x for x in errs if x is not None]
        row = (D, F, sum(x is None for x in errs), early, np.median(good), np.percentile(good, 5), np.percentile(good, 95))
        print('D_MIN %.2f RETURN %.2f | missed %3d | >0.5 s early %3d | det-b3 median %.2f p5 %.2f p95 %.2f s' % row)

print('--- with the drop held for PERSIST frames (the arm keeps coming back)')
for D in (1.0, 1.2):
    for F in (0.2, 0.25):
        for P in (8, 15):
            errs, early = [], 0
            for r in res:
                dd, peak = np.array(r['dd']), np.array(r['peak'])
                ok = (np.arange(len(dd)) > r['b2']) & (peak >= D) & (dd <= peak * (1 - F))
                ok = np.convolve(ok, np.ones(P), 'full')[:len(ok)] >= P
                if not ok.any(): errs.append(None); continue
                f = int(np.argmax(ok)); errs.append((f - r['b3']) / 30); early += f < r['b3'] - 15
            good = [x for x in errs if x is not None]
            print('D_MIN %.1f RETURN %.2f PERSIST %2d | missed %d | >0.5 s early %3d | det-b3 median %.2f p5 %.2f p95 %.2f max %.2f s' % (
                D, F, P, sum(x is None for x in errs), early, np.median(good), np.percentile(good, 5), np.percentile(good, 95), max(good)))

print('--- harness sees one observation per action chunk (~1.3 s = 39 frames); 4 phase offsets per demo')
for need in (1, 2):
    g_err, d_err, d_early, d_miss = [], [], 0, 0
    for r in res:
        for off in (0, 10, 20, 30):
            ix = np.arange(off, r['T'], 39)
            # grasp->sweep: `need` consecutive samples with both grippers in the holding band (after an opening)
            # (recomputed from the stored per-frame displacement is not possible; use b1 + 1 s rule equivalence)
            dd = np.array(r['dd'])[ix]; after = ix > r['b2']
            peak = np.maximum.accumulate(np.where(after, dd, 0))
            ok = after & (peak >= 1.0) & (dd <= 0.8 * peak)
            okc = np.convolve(ok, np.ones(need), 'full')[:len(ok)] >= need
            if not okc.any(): d_miss += 1; continue
            f = ix[int(np.argmax(okc))]; d_err.append((f - r['b3']) / 30); d_early += f < r['b3'] - 15
    print(f'dump->home with {need} consecutive chunk(s): missed {d_miss}/{4*len(res)}, >0.5 s early {d_early}, '
          f'det-b3 median {np.median(d_err):.2f} p5 {np.percentile(d_err, 5):.2f} p95 {np.percentile(d_err, 95):.2f} max {max(d_err):.2f} s')
