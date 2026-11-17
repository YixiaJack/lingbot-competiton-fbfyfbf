"""Tables for the sweeping-harness experiment: per harness (same checkpoint, same scenes) and the offline judgement test.
Usage: summarize.py --root R [--run sweep_s22]
"""
import argparse, json
from pathlib import Path
import numpy as np
from prompts import SUBTASK_OF_SKILL

STAGES = ['grasp_dustpan', 'grasp_brush', 'sweep', 'carry', 'pour', 'home']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--run', default='sweep_s22')
    a = p.parse_args()
    out = {'harness': {}, 'judge': {}}
    by_h = {}
    for d in sorted((a.root / 'results').glob(f'{a.run}_*')):
        rs = [json.loads(f.read_text()) for f in sorted(d.glob('*/result.json'))]
        if rs:
            by_h[d.name[len(a.run) + 1:]] = {r['item']: r for r in rs}
    common = sorted(set.intersection(*[set(v) for v in by_h.values()])) if by_h else []
    for h, rs in by_h.items():
        items = sorted(rs)
        sel = [rs[i] for i in items]
        row = {'n': len(sel), 'score': float(np.mean([r['score'] for r in sel])),
               'success_rate': float(np.mean([r['success'] for r in sel])),
               'process_points': float(np.mean([r['process_points'] for r in sel])),
               'stage_rates': {s: float(np.mean([r['stages'][s] for r in sel])) for s in STAGES},
               'episode_s': float(np.mean([r['episode_s'] for r in sel])),
               'ended_by': {k: sum(r['ended_by'] == k for r in sel) for k in ('harness', 'time', 'error')},
               'errors': sum(bool(r['error']) for r in sel)}
        if common:
            row['score_on_common_items'] = float(np.mean([rs[i]['score'] for i in common]))
        hj = [json.loads((a.root / 'results' / f'{a.run}_{h}' / f'{i:04d}' / 'harness.json').read_text()) for i in items]
        if hj and 'n_calls' in hj[0]:
            pt = sum(x['prompt_tokens'] for x in hj); ch = sum(x['cache_hit_tokens'] for x in hj)
            lat = [x['latency_mean_s'] for x in hj if x['latency_mean_s']]
            row.update(llm_calls_per_episode=float(np.mean([x['n_calls'] for x in hj])),
                       llm_invalid=int(sum(x['n_invalid'] for x in hj)),
                       llm_latency_mean_s=float(np.mean(lat)) if lat else None,
                       llm_cache_hit_rate=ch / pt if pt else None,
                       distinct_prompts=len({q['prompt'] for x in hj for q in x['prompts']}))
        out['harness'][h] = row
    out['common_items'] = len(common)
    for d in sorted((a.root / 'judge').glob('*')):
        rs = [json.loads(f.read_text()) for f in sorted(d.glob('*.json'))]
        if not rs:
            continue
        rows = [r for x in rs for r in x['rows']]
        conf = {}
        for r in rows:
            conf.setdefault(r['target'], {}).setdefault(str(r['answer']), 0)
            conf[r['target']][str(r['answer'])] += 1
        # same yardstick for both versions: v1's skill answers mapped to v2's five subtasks
        coarse = lambda x: SUBTASK_OF_SKILL.get(x, x)
        sub = [coarse(r['answer']) == coarse(r['label']) for r in rows]
        sub_len = [coarse(r['answer']) in (coarse(r['label']), coarse(r['target'])) or r['lenient'] for r in rows]
        out['judge'][d.name] = {'demos': len(rs), 'queries': len(rows),
                                'subtask_exact': float(np.mean(sub)), 'subtask_lenient': float(np.mean(sub_len)),
                                'exact': float(np.mean([r['exact'] for r in rows])),
                                'lenient': float(np.mean([r['lenient'] for r in rows])),
                                'keyword_ok': float(np.mean([r['keyword_ok'] for r in rows])) if rows[0]['keyword_ok'] is not None else None,
                                'latency_mean_s': float(np.mean([r['latency_s'] for r in rows])),
                                'cache_hit_rate': float(sum(r['cache_hit_tokens'] or 0 for r in rows) / max(1, sum(r['prompt_tokens'] or 0 for r in rows))),
                                'confusion': conf}
    (a.root / 'summary.json').write_text(json.dumps(out, indent=1))
    print(f"{'harness':10s} {'n':>3s} {'score':>6s} {'SR':>5s} {'PSR':>5s} " + ' '.join(f'{s[:7]:>7s}' for s in STAGES) + f" {'time':>6s}")
    for h, r in out['harness'].items():
        print(f"{h:10s} {r['n']:3d} {r['score']:6.1f} {r['success_rate']:5.2f} {r['process_points']:5.1f} "
              + ' '.join(f"{r['stage_rates'][s]:7.2f}" for s in STAGES) + f" {r['episode_s']:6.1f}")
    for k, j in out['judge'].items():
        print(f"judge {k}: own-label exact {j['exact']:.2f} lenient {j['lenient']:.2f} | subtask exact {j['subtask_exact']:.2f} "
              f"lenient {j['subtask_lenient']:.2f} | keyword {j['keyword_ok']} latency {j['latency_mean_s']:.2f}s cache {j['cache_hit_rate']:.2f}")


if __name__ == '__main__':
    main()
