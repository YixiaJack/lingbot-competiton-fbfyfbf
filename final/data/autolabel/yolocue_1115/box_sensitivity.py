"""Does the deployed policy follow the box cue? (5080, own process; the service must have no model loaded.)
Scenes: official task-1 episodes, frame 0 (all three cameras + state from public647), whose instruction parses, whose
yolocue cue decision is 'ok' on the v5 frame-0 detections (the same boxes the training cue was drawn from), and which
have a confident other task-1 object on the OTHER half of the image (left arm takes the left half, right arm the right).
Per scene, 5 conditions x 2 seeds, the first chunk only:
  C0 original instruction, no box      C1 original + box on target      C2 original + box on the distractor
  C3 generic prompt + box on target    C4 generic prompt + box on the distractor
'arm' = the arm whose joints move more over the chunk. Following the box = arm on the boxed object's side.
usage: cd ~/workspace/lingbot/inference && source env.sh &&
  $LINGBOT_PYTHON_ENV/bin/python box_sensitivity.py <hf_ckpt> [--n 24] [--out result.json]"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from utils.arguments import LingBotInferConfig
from utils.data_define import ModelInput
import policy_adapter.policy_adapter as PA

sys.path.insert(0, str(Path.home() / 'yc_test'))
import yolocue_bridge as YC  # noqa: E402  (has cue_prompts; draw/select/parse identical to the live file)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('--n', type=int, default=24); ap.add_argument('--out', default='box_sensitivity.json')
a = ap.parse_args()
H = Path.home()
labels = {}
for line in open(H / 'locany/data/merged/meta/labels.jsonl'):
    r = json.loads(line); labels[r['episode_index']] = r['labels'][0]['task_instruction']
dets = json.loads((H / 'locany/yolo/runs/v5/dets_first.json').read_text())
ds = LeRobotDataset('local/public647', root=str(H / 'locany/data/public647'), video_backend='pyav')
T1_OBJ = set(YC.T1_OBJ)
side = lambda b: 'left' if (b[0] + b[2]) / 2 < 320 else 'right'

scenes = []
for e in range(ds.meta.total_episodes):
    instr = labels.get(e, '')
    p = YC.parse(instr)
    if not p or p[0] != 1:
        continue
    task_pub = ds.meta.episodes[e]['tasks'][0]
    if ' '.join(task_pub.split()) != ' '.join(instr.split()):
        continue                                       # merged labels and public647 must be the same episode
    d = dets.get(f'first/{e:04d}', [])
    tb, db, why = YC.select(d, *p)
    if why != 'ok':
        continue
    others = [r for r in d if r[0] in T1_OBJ and r[0] != p[1] and r[5] >= 0.5 and side(r[1:5]) != side(tb)]
    if not others:
        continue
    xb = max(others, key=lambda r: abs((r[1] + r[3]) / 2 - (tb[0] + tb[2]) / 2))
    scenes.append({'ep': e, 'instr': instr, 'target': p[1], 'tb': tb, 'db': db, 'distractor': xb[0], 'xb': [round(v, 1) for v in xb[1:5]]})
    if len(scenes) >= a.n:
        break
print(f'{len(scenes)} scenes')

cfg = LingBotInferConfig.from_yaml(H / 'workspace/lingbot/inference/config/inference.yaml', overrides={'policy.model_path': a.model})
pol = PA.PolicyFactory.create(cfg.policy)
pol.init(cfg.robot, 'Pick up the carrot and place it on the purple plate.')


def frame0(e):
    it = ds[int(ds.meta.episodes[e]['dataset_from_index'])]
    img = lambda k: (it[k].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    arm, g = it['observation.state.arm.position'].numpy(), it['observation.state.effector.position'].numpy()
    f = {f'observation.images.{c}': img(f'observation.images.{c}') for c in ('camera_top', 'camera_wrist_left', 'camera_wrist_right')}
    f.update({'observation.state.left_arm.position': arm[:6], 'observation.state.right_arm.position': arm[6:12],
              'observation.state.left_gripper.position': g[:1], 'observation.state.right_gripper.position': g[1:2]})
    return ModelInput(sequence=1, captured_at=time.monotonic(), features=f)


def arm_of(chunk):
    acts = chunk.actions
    mv = {}
    for s in ('left', 'right'):
        q = np.asarray(acts[f'action.{s}_arm.position'], float)
        mv[s] = float(np.abs(q - q[0]).sum())
    return max(mv, key=mv.get), mv


CONDS = ['C0', 'C1', 'C2', 'C3', 'C4']
rows = []
t0 = time.time()
for sc in scenes:
    mi = frame0(sc['ep'])
    named, generic = YC.cue_prompts(sc['instr'])
    raw = mi.features['observation.images.camera_top']
    variants = {'C0': (sc['instr'], raw), 'C1': (sc['instr'], YC.draw(raw, sc['tb'], sc['db'], False)),
                'C2': (sc['instr'], YC.draw(raw, sc['xb'], sc['db'], False)),
                'C3': (generic, YC.draw(raw, sc['tb'], sc['db'], False)), 'C4': (generic, YC.draw(raw, sc['xb'], sc['db'], False))}
    res = {}
    for c in CONDS:
        text, img = variants[c]
        arms = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            obs = pol.preprocess(mi, run_inference=True)
            obs['task'] = text
            obs['observation.images.camera_top'] = img
            arm, mv = arm_of(pol.postprocess(pol.infer(obs)))
            arms.append(arm)
        res[c] = arms
    rows.append({**sc, 'arms': res, 'target_side': side(sc['tb']), 'distractor_side': side(sc['xb'])})
    print(sc['ep'], sc['target'], side(sc['tb']), '| distractor', sc['distractor'], side(sc['xb']), '|',
          ' '.join(f"{c}:{''.join(x[0] for x in res[c])}" for c in CONDS), flush=True)

summ = {}
for c in CONDS:
    want = 'distractor_side' if c in ('C2', 'C4') else 'target_side'
    hits = [x == r[want] for r in rows for x in r['arms'][c]]
    summ[c] = round(float(np.mean(hits)), 3) if hits else None
out = {'model': a.model, 'scenes': len(rows), 'seeds': 2, 'seconds': round(time.time() - t0, 1),
       'rate': summ, 'meaning': {'C0': 'arm on target side, no box', 'C1': 'arm on target side, box on target',
                                 'C2': 'arm on DISTRACTOR side, original text + box on distractor',
                                 'C3': 'arm on target side, generic text + box on target',
                                 'C4': 'arm on DISTRACTOR side, generic text + box on distractor'}, 'rows': rows}
Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
print('RATES', json.dumps(summ), f"({len(rows)} scenes x 2 seeds, {out['seconds']} s)")
