"""Label every task-3 demo of the official real-robot dataset with the fixed four subtasks (real_split.py).

Output JSON, one record per demo: episode index, demo id (T3_train001 ...), dustpan arm, the boundaries in frames
(30 fps) and the four (subtask, start, end, instruction) segments, plus a summary block. The LeRobot conversion that
fine-tunes the VLA on subtask instructions reads this file; the offline judgement test (judge_real.py) uses it as truth.
Usage: segment_real.py --data <.../snapshots/<rev>/v1> --out real_t3_subtasks.json
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import real_split as R


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    ids = {}
    for line in open(a.data / 'meta/labels.jsonl'):
        d = json.loads(line)
        ids[d['episode_index']] = next(f['value'] for f in d['labels'][0]['custom_fields'] if f['name'] == 'id')
    eps = sorted(e for e, i in ids.items() if i.startswith('T3_'))
    cols = ['episode_index', 'frame_index', 'observation.state.effector.position', 'observation.state.ee.position']
    per = {e: [] for e in eps}
    for f in sorted(glob.glob(str(a.data / 'data/*/*.parquet'))):
        t = pq.read_table(f, columns=cols)
        ep = t.column('episode_index').to_numpy()
        keep = np.isin(ep, eps)
        if not keep.any():
            continue
        fi = t.column('frame_index').to_numpy()[keep]
        gr = np.array(t.column(cols[2]).to_pylist(), dtype=float)[keep]
        ee = np.array(t.column(cols[3]).to_pylist(), dtype=float)[keep]
        for e in np.unique(ep[keep]):
            m = ep[keep] == e
            per[int(e)].append((fi[m], gr[m], ee[m]))
    recs, failed = [], []
    for e in eps:
        fi = np.concatenate([x[0] for x in per[e]]); o = np.argsort(fi)
        gr = np.concatenate([x[1] for x in per[e]])[o]; ee = np.concatenate([x[2] for x in per[e]])[o]
        try:
            s = R.segment(gr, ee)
        except ValueError as ex:
            failed.append({'episode_index': e, 'id': ids[e], 'error': str(ex)}); continue
        bounds = [0, s['b1'], s['b2'], s['b3'], s['T']]
        segs = [{'subtask': k, 'start': bounds[i], 'end': bounds[i + 1], 'instruction': R.instruction(k, s['pan_arm'])}
                for i, k in enumerate(R.SUBTASK_IDS)]
        recs.append({'episode_index': e, 'id': ids[e], **s, 'segments': segs})
    dur = {k: [(r['segments'][i]['end'] - r['segments'][i]['start']) / R.FPS for r in recs] for i, k in enumerate(R.SUBTASK_IDS)}
    summary = {'episodes': len(recs), 'failed': failed,
               'pan_arm': {s: sum(r['pan_arm'] == s for r in recs) for s in ('left', 'right')},
               'hold_pan_range': [min(r['hold_pan'] for r in recs), max(r['hold_pan'] for r in recs)],
               'hold_brush_range': [min(r['hold_brush'] for r in recs), max(r['hold_brush'] for r in recs)],
               'duration_s': {k: {'p5': round(float(np.percentile(v, 5)), 2), 'median': round(float(np.median(v)), 2),
                                  'p95': round(float(np.percentile(v, 95)), 2)} for k, v in dur.items()},
               'subtasks': R.SUBTASKS}
    a.out.write_text(json.dumps({'summary': summary, 'episodes': recs}, ensure_ascii=False, indent=1))
    print(json.dumps({k: summary[k] for k in ('episodes', 'failed', 'pan_arm', 'hold_pan_range', 'hold_brush_range', 'duration_s')}))


if __name__ == '__main__':
    main()
