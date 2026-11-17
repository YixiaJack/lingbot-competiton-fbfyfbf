"""Build one scene and save what it looks like: head-camera frame zero, the same frame with the labelled boxes, and the
layout (every object's and plate's position). No expert, no policy. Layout rules: SCENE.md.

  bash runtime.sh sim python show_scene.py --task 1 --target banana --dest pink --n_objects 4 --n_plates 3 --seed 7 --out OUT
  bash runtime.sh sim python show_scene.py --task 2 --target lipstick --dest top --seed 3 --out OUT
  bash runtime.sh sim python show_scene.py --task 3 --seed 5 --light --out OUT
Without --target / --dest a random one is drawn from the seed. --light: RoboTwin's random lighting (half the detector
images use it). Needs the RoboTwin env (runtime.sh sim) and REAL_ASSETS pointing at the processed assets, if they are
not in ../assets.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scene_real import RealTask, config, boxes, T1_OBJECTS, COSMETICS, PLATE_COLORS, LEVELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, choices=[1, 2, 3], required=True)
    p.add_argument('--target'); p.add_argument('--dest')
    p.add_argument('--template', type=int, default=0, help='instruction template 0-4')
    p.add_argument('--n_objects', type=int, default=4); p.add_argument('--n_plates', type=int, default=3)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--light', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.task == 1:
        spec = {'task': 1, 'target': a.target or str(rng.choice(T1_OBJECTS)), 'dest': a.dest or str(rng.choice(PLATE_COLORS)),
                'template': a.template, 'n_objects': a.n_objects, 'n_plates': a.n_plates}
    elif a.task == 2:
        spec = {'task': 2, 'target': a.target or str(rng.choice(COSMETICS)), 'dest': a.dest or str(rng.choice(LEVELS)),
                'template': a.template, 'n_objects': a.n_objects}
    else:
        spec = {'task': 3}
    cfg = config(os.environ['ROBOTWIN_DIR'])
    if a.light:
        cfg['domain_randomization'] = {**cfg.get('domain_randomization', {}), 'random_light': True, 'crazy_random_light_rate': 0}
    env = RealTask()
    try:
        env.setup_demo(spec=spec, seed=a.seed, now_ep_num=0, **cfg)
        img = env.get_obs()['observation']['head_camera']['rgb']
        bx = boxes(env, strict=False)
        layout = {
            'spec': spec, 'seed': a.seed, 'instruction': env.get_instruction(),
            'plates': [{'color': q['color'], 'xy': [round(float(v), 4) for v in q['xy']]} for q in env.plates],
            'objects': [{'key': k, 'xyz': [round(float(v), 4) for v in c.get_pose().p]}
                        for k, c in zip(env.candidate_keys, env.candidates)],
            'organizer': {lv: {'x': [round(float(v), 4) for v in d['x']], 'y': [round(float(v), 4) for v in d['y']],
                               'floor_z': round(float(d['floor']), 4)} for lv, d in env.levels.items()},
            'task3_objects': [{'key': k, 'xyz': [round(float(v), 4) for v in c.get_pose().p]}
                              for k, c in zip(env.extra_keys, env.extras)],
            'boxes': bx,
        }
    finally:
        try: env.close_env(clear_cache=True)
        except Exception: pass
    a.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(a.out / 'frame0.png')
    im = Image.fromarray(img).convert('RGB'); dr = ImageDraw.Draw(im)
    for b in bx:
        col = ((255, 255, 255) if b['cls'] == spec.get('target') else
               (255, 140, 0) if b['cls'] in (f"plate_{spec.get('dest')}", f"organizer_{spec.get('dest')}") else (90, 200, 255))
        dr.rectangle(b['box'], outline=col, width=1); dr.text((b['box'][0] + 2, max(0, b['box'][1] - 11)), b['cls'], fill=col)
    im.save(a.out / 'frame0_boxes.png')
    (a.out / 'layout.json').write_text(json.dumps(layout, indent=1, ensure_ascii=False))
    print(layout['instruction'], '|', len(bx), 'boxes ->', a.out)


if __name__ == '__main__':
    main()
