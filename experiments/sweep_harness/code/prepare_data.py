"""LeRobot datasets from the sweeping demonstrations, same format as the prelim conversion (state = 14-d RoboTwin joint
vector, action = next frame's state, fps 50, three cameras).
  skills  every skill segment of every demo is its own episode; prompt = canonical (40%) or a scene-filled variant
          (prompts.sample_prompt), fixed per segment. The VLA learns each step and to stop at its end.
  full    every demo is one episode with the full task instruction (the pure-VLA baseline's training signal).
One model is trained on both, so all harnesses and the baseline share the same weights.
Items: lists/train_items.txt, written once by the controller (val items held out for the LLM judgement test). Also writes timing.json: the timed
harness's step durations (median expert duration x 1.3).
Sharded for speed (video encoding is ~45 s per demo): --variant skills_K / full_K converts every NSHARDS-th training
item starting at K into lerobot/skills_K / full_K. Shard K of both variants holds the same demos, so with the uniform
dataset sampling of train.sh (each listed dataset drawn with equal probability) the two variants get half the batches
each. Shard 0 of skills writes timing.json.
Usage: prepare_data.py --root R --variant skills_K|full_K
"""
import os
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from skills import FULL_INSTRUCTION, SKILL_IDS, relabel, segments
from prompts import sample_prompt

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
MOTORS = [f'joint_{i}' for i in range(14)]
DT = 15 / 250
NSHARDS = 8


def load(d):   # decompress every array once (indexing an NpzFile decompresses the whole array on every access)
    with np.load(d / 'episode.npz') as z:
        return {k: z[k] for k in z.files}


def episodes(data, meta, i, variant):
    """(prompt or (skill, start, rng), frame indices) for each LeRobot episode of demo i. Skill segments use
    skills.relabel (pour ends ~1 s after both balls are in the basket)."""
    n = meta['frames'] - 1   # the last frame has no next state
    if variant == 'full':
        return [(FULL_INSTRUCTION, list(range(n)))]
    rng = np.random.default_rng(1000 * i + 7)
    out = []
    for skill, s0, s1 in segments(relabel(data['skill'], data['flags'])):
        idx = list(range(s0, min(s1, n)))
        if len(idx) >= 5:
            out.append(((skill, s0, rng), idx))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', required=True, help='skills_K or full_K')
    a = p.parse_args()
    variant, shard = a.variant.rsplit('_', 1); shard = int(shard)
    assert variant in ('skills', 'full') and 0 <= shard < NSHARDS
    items = [int(x) for x in (a.root / 'lists/train_items.txt').read_text().split()][shard::NSHARDS]   # fixed by the controller
    output = a.root / 'lerobot' / a.variant
    if (output / 'READY').exists():
        return
    if output.exists():
        output.rename(output.with_name(output.name + f'.interrupted-{time.time_ns()}'))
    first = load(a.root / 'episodes' / f'{items[0]:04d}'); h, w = first['cam_high'].shape[1:3]
    features = {'observation.state': {'dtype': 'float32', 'shape': (14,), 'names': [MOTORS]},
                'action': {'dtype': 'float32', 'shape': (14,), 'names': [MOTORS]},
                **{f'observation.images.{c}': {'dtype': 'video', 'shape': (h, w, 3), 'names': ['height', 'width', 'channels']}
                   for c in CAMERAS}}
    ds = LeRobotDataset.create(repo_id=f'sweep/{a.variant}', root=output, fps=50, robot_type='aloha', features=features,
                               use_videos=True, image_writer_processes=2, image_writer_threads=2)
    index, prompts = [], {}
    for i in items:
      data = load(a.root / 'episodes' / f'{i:04d}')
      meta = json.loads((a.root / 'episodes' / f'{i:04d}' / 'complete.json').read_text())
      for pr, idx in episodes(data, meta, i, variant):
          if isinstance(pr, tuple):
              skill, start, rng = pr
              balls = {'red': data['balls_xy'][start][0].tolist(), 'white': data['balls_xy'][start][1].tolist()}
              prompt = sample_prompt(skill, rng, balls)
          else:
              skill, prompt = 'full', pr
          for j in idx:
              ds.add_frame({'observation.state': data['state'][j], 'action': data['state'][j + 1], 'task': prompt,
                            **{f'observation.images.{c}': data[c][j] for c in CAMERAS}})
          ds.save_episode()
          index.append({'item': i, 'skill': skill, 'prompt': prompt, 'frames': len(idx), 'first': idx[0]})
          prompts[prompt] = prompts.get(prompt, 0) + 1
    ds.finalize()
    (output / 'episodes_index.json').write_text(json.dumps(index, indent=1))
    (output / 'prompt_counts.json').write_text(json.dumps(dict(sorted(prompts.items(), key=lambda x: -x[1])), indent=1))
    validate(output, a.root, index)
    if variant == 'skills' and shard == 0:   # timed harness: median duration of each skill over the training demos, x 1.3
        dur = {k: [e['frames'] * DT for e in index if e['skill'] == k] for k in SKILL_IDS}
        timing = {k: round(1.3 * float(np.median(v)), 2) if v else 8.0 for k, v in dur.items()}
        (a.root / 'timing.json').write_text(json.dumps(timing, indent=1))
    (output / 'READY').write_text('complete\n')


def validate(output, root, index):
    """Read back through LingBot's own dataset class: counts, state/action alignment, prompt, image error."""
    sys.path.insert(0, os.path.expandvars('$SCRATCH/lingbot/dev/lingbot-vla-v2-dino'))
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset as LBDataset
    ds = LBDataset(str(output), delta_timestamps={'action': [i / 50 for i in range(50)]}, video_backend='torchcodec')
    assert ds.num_episodes == len(index) and len(ds) == sum(e['frames'] for e in index)
    checked, offset = [], np.cumsum([0] + [e['frames'] for e in index])
    for ep in sorted({0, len(index) // 2, len(index) - 1}):
        e = index[ep]; src = load(root / 'episodes' / f"{e['item']:04d}")
        for k in (0, e['frames'] - 1):
            item = ds[int(offset[ep] + k)]; j = e['first'] + k
            np.testing.assert_allclose(item['observation.state'].numpy(), src['state'][j], atol=1e-6)
            np.testing.assert_allclose(item['action'][0].numpy(), src['state'][j + 1], atol=1e-6)
            assert item['task'] == e['prompt'], (item['task'], e['prompt'])
            err = float(np.abs(item['observation.images.cam_high'].permute(1, 2, 0).numpy() / 255. - src['cam_high'][j] / 255.).mean())
            assert err < .03, err
            if k == e['frames'] - 1:
                assert item['action_is_pad'][1:].all()
            checked.append({'episode': ep, 'frame': k, 'image_mae': err})
    (output / 'validation.json').write_text(json.dumps({'episodes': ds.num_episodes, 'frames': len(ds), 'checked': checked}, indent=1))


if __name__ == '__main__':
    main()
