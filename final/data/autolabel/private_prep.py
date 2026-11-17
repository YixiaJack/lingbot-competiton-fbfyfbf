"""Prepare our collected (good) dataset for VLA training, off the training machine.

- task of every episode from its instruction (task 1 plate template, task 2 organizer, task 3 dustpan/paper balls)
- task 3: the fixed 4-subtask split (sweep_harness/code/real_split.py); kept only if it splits and pours exactly once
- task 1: the boxgrasp grasp rule (first close of the gripper COMMAND after it opened) is kept for box cues only if the
  demo has one acting arm, the measured gripper holds something 0.5-1 s after that close, and the arm then lifts > 3 cm
- frame 0 of every task-1 demo saved as JPEG for the detector (runs on the inference machine)
Usage: private_prep.py --data <dataset v1 dir> --out DIR
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import av
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sweep_harness/code'))
import real_split as R
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_boxes as TB


def task_of(instr):
    s = (instr or '').lower()
    if any(w in s for w in ('dustpan', 'paper ball', 'sweep', 'brush all')): return 3
    if any(w in s for w in ('compartment', 'organizer', 'level', 'tier', 'layer', 'shelf', 'section')): return 2
    if 'plate' in s: return 1
    return 0


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); (a.out / 'frame0').mkdir(exist_ok=True)
    labels = {}
    for line in open(a.data / 'meta/labels.jsonl'):
        d = json.loads(line); L = d['labels'][0]
        labels[d['episode_index']] = {'instruction': L['task_instruction'], 'result': L.get('result'), 'score': L.get('score')}
    eps = {e['episode_index']: e for e in pq.read_table(sorted((a.data / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
    t = pq.read_table(sorted((a.data / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'observation.state.effector.position',
                      'observation.state.ee.position', 'action.effector.position'])
    ep = t.column('episode_index').to_numpy(); fi = t.column('frame_index').to_numpy()
    G = np.array(t.column('observation.state.effector.position').to_pylist()); E = np.array(t.column('observation.state.ee.position').to_pylist())
    C = np.array(t.column('action.effector.position').to_pylist())
    out = {'source': str(a.data), 'episodes': {}}
    cam = 'observation.images.camera_top'
    for e in sorted(eps):
        lb = labels[e]; k = task_of(lb['instruction']); rec = {'task': k, **lb, 'length': eps[e]['length']}
        m = ep == e; o = np.argsort(fi[m]); g, ee, c = G[m][o], E[m][o], C[m][o]
        if k == 3:
            try:
                s = R.segment(g, ee)
                pan = 0 if s['pan_arm'] == 'left' else 1
                x = ee[:, 7 * pan]; x0 = np.median(x[s['b1']:s['b2']])
                runs = [(u, v) for u, v in R.runs(x >= x0 + 0.5 * (s['x_pour'] - x0)) if v - u >= 15]
                merged = []
                for u, v in runs:
                    if merged and u - merged[-1][1] < 30: merged[-1] = (merged[-1][0], v)
                    else: merged.append((u, v))
                pours = sum(1 for u, v in merged if u < s['release'])
                rec.update(split=s, pours=pours, use_t3_split=(pours == 1))
                if pours == 1:
                    b = [0, s['b1'], s['b2'], s['b3'], s['T']]
                    rec['segments'] = [{'subtask': st, 'start': b[i], 'end': b[i + 1], 'instruction': R.instruction(st, s['pan_arm'])}
                                       for i, st in enumerate(R.SUBTASK_IDS)]
            except ValueError as ex:
                rec.update(split_error=str(ex), use_t3_split=False)
        elif k == 1:
            gf = TB.grasp_frame(c)
            arms = [x for x in (0, 1) if (c[:, x] > TB.CMD_OPEN).any()]
            ok = False
            if gf is not None and len(arms) == 1:
                x = arms[0]
                hold = float(np.median(g[gf + 15:gf + 30, x]) - np.median(g[:15, x]))
                lift = float(ee[gf:gf + 60, 7 * x + 2].max() - ee[gf, 7 * x + 2])
                ok = hold > 0.02 and lift > 0.03
                rec.update(hold=round(hold, 3), lift=round(lift, 3))
            rec.update(grasp_frame=gf, grasp_rule_ok=ok, parsed=TB.parse(lb['instruction']))
            m0 = eps[e]
            path = a.data / f"videos/{cam}/chunk-{m0[f'videos/{cam}/chunk_index']:03d}/file-{m0[f'videos/{cam}/file_index']:03d}.mp4"
            t0 = m0[f'videos/{cam}/from_timestamp']
            with av.open(str(path)) as ic:
                st = ic.streams.video[0]; ic.seek(int(max(t0 - 0.1, 0) / st.time_base), stream=st, backward=True)
                for fr in ic.decode(st):
                    if float(fr.pts * st.time_base) + 1e-3 >= t0:
                        fr.to_image().save(a.out / 'frame0' / f'ep{e:04d}.jpg', quality=95); break
        out['episodes'][str(e)] = rec
    E_ = out['episodes'].values()
    summ = {'episodes': len(out['episodes']), 'tasks': {t_: sum(r['task'] == t_ for r in E_) for t_ in (0, 1, 2, 3)},
            'results': {str(x): sum(str(r['result']) == str(x) for r in E_) for x in sorted({str(r['result']) for r in E_})},
            't3_split_ok': sum(r.get('use_t3_split', False) for r in E_ if r['task'] == 3),
            't1_grasp_rule_ok': sum(r.get('grasp_rule_ok', False) for r in E_ if r['task'] == 1),
            't1_parsed': sum(r.get('parsed') is not None for r in E_ if r['task'] == 1),
            't1_unparsed_instructions': sorted({r['instruction'] for r in E_ if r['task'] == 1 and r.get('parsed') is None})}
    out['summary'] = summ
    (a.out / 'private_prep.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
