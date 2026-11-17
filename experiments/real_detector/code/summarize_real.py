"""summary.json for the real-scene cue A/B: per condition and task, first-approach rate (Wilson 95%) and success; paired
differences against the plain-text model of the same starting point on identical scenes (bootstrap 95%)."""
import os
import json, random, statistics
from pathlib import Path

R = Path(os.path.expandvars('$SCRATCH/lingbot/final_real_20261112'))
rows = {}
for p in (R / 'results').glob('*/*/result.json'):
    r = json.loads(p.read_text()); rows.setdefault(p.parent.parent.name, []).append(r)


def wilson(k, n, z=1.96):
    if not n: return [None, None]
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    d = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / den
    return [max(0, c - d), min(1, c + d)]


groups = []
for cond, rs in sorted(rows.items()):
    for task in (1, 2, 'all'):
        sel = [r for r in rs if task == 'all' or r['task'] == task]
        if not sel: continue
        n, k, s = len(sel), sum(r['first_approach_correct'] for r in sel), sum(r['success'] for r in sel)
        warm = [x for r in sel for x in r['latency_ms'][1:]]
        groups.append({'condition': cond, 'task': task, 'n': n, 'first_correct': k, 'first_rate': k / n, 'first_95ci': wilson(k, n),
                       'successes': s, 'success_rate': s / n, 'no_approach': sum(r['first_approach'] is None for r in sel),
                       'cue_fallbacks': sum(r['mode'] != 'text' and not r['cue_applied'] for r in sel),
                       'warm_latency_median_ms': statistics.median(warm) if warm else None})
pairs = []
REF = {'s22': 's22_text_text_oracle'}
for cond in sorted(rows):
    arm = cond.split('_')[0]; ref = REF.get(arm)
    if not ref or cond == ref or ref not in rows: continue
    A = {r['item']: r for r in rows[ref]}; B = {r['item']: r for r in rows[cond]}
    for task in (1, 2, 'all'):
        pr = [(A[i], B[i]) for i in sorted(set(A) & set(B)) if A[i]['scene_fingerprint'] == B[i]['scene_fingerprint']
              and (task == 'all' or A[i]['task'] == task)]
        if not pr: continue
        dl = [int(b['first_approach_correct']) - int(a['first_approach_correct']) for a, b in pr]
        rng = random.Random(42)
        boot = sorted(sum(rng.choices(dl, k=len(dl))) / len(dl) for _ in range(3000))
        pairs.append({'condition': cond, 'reference': ref, 'task': task, 'paired_n': len(pr),
                      'first_delta_pp': 100 * statistics.mean(dl), 'bootstrap_95ci_pp': [100 * boot[75], 100 * boot[2924]],
                      'success_delta_pp': 100 * statistics.mean(int(b['success']) - int(a['success']) for a, b in pr)})
out = {'groups': groups, 'paired_comparisons': pairs,
       'adoption_rule': 'box or coords >= +15 pp first-approach over plain text, and no drop without the cue (text input)',
       'scope': 'pre-event simulation; competition-like objects and scenes; no artifact goes to the venue'}
t = R / 'summary.tmp'; t.write_text(json.dumps(out, indent=1)); t.replace(R / 'summary.json')
