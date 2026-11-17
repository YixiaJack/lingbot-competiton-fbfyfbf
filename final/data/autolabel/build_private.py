"""Plain and task-3 subtask datasets from our collected (good) dataset, built off the training machine.

priv_t12:    task-1 and task-2 demos, whole episodes, instructions rewritten to the official phrasing (private_instructions.py)
priv_t3_sub: task-3 demos that split into the fixed 4 subtasks and pour once, one sub-episode per subtask
Videos are not copied: videos/ is a symlink to be re-pointed on the training machine (meta/ and data/ are uploaded).
Usage: build_private.py --src <our dataset v1> --prep private_prep.json --instructions priv_instructions.json --out DIR
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sweep_harness/code'))
import derive_dataset as DD


def main():
    p = argparse.ArgumentParser()
    for k in ('src', 'prep', 'instructions', 'out'):
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args()
    P = json.loads(a.prep.read_text())['episodes']; I = json.loads(a.instructions.read_text())['episodes']
    info, eps, ids = DD.load_src(a.src); fps = info['fps']
    t12 = [{'src_ep': int(k), 'start': 0, 'end': eps[int(k)]['length'], 'task': I[k]['instruction'], 'orig_instruction': I[k]['original']}
           for k in sorted(P, key=int) if P[k]['task'] in (1, 2)]
    print('priv_t12', json.dumps(DD.build(a.src, a.out / 'priv_t12', t12, fps)))
    t3 = [{'src_ep': int(k), 'start': s['start'], 'end': s['end'], 'task': s['instruction'], 'subtask': s['subtask']}
          for k in sorted(P, key=int) if P[k]['task'] == 3 and P[k].get('use_t3_split') for s in P[k]['segments']]
    print('priv_t3_sub', json.dumps(DD.build(a.src, a.out / 'priv_t3_sub', t3, fps)))


if __name__ == '__main__':
    main()
