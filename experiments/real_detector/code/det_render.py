"""Detector images: frame zero of random competition-like scenes (tasks 1-3), YOLO labels from segmentation.

Item i renders PER_ITEM scenes. Items < N_TRAIN_ITEMS -> train split, the rest -> val (disjoint seed range).
Mix per item: 11 task-1, 6 task-2, 3 task-3 scenes. Half of the scenes use RoboTwin's random lighting.
Usage: det_render.py --root R --item i
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image
from scene_real import RealTask, config, boxes, CLASSES, T1_OBJECTS, COSMETICS, PLATE_COLORS, LEVELS, TEMPLATES

from catalog import DET_TRAIN_ITEMS as N_TRAIN_ITEMS, DET_ITEMS
PER_ITEM = 20   # 1800 train + 400 val images (items 90-109)
CLS = {c: i for i, c in enumerate(CLASSES)}


def spec_for(rng, task):
    if task == 1:
        return {'task': 1, 'target': str(rng.choice(T1_OBJECTS)), 'dest': str(rng.choice(PLATE_COLORS)),
                'template': int(rng.integers(5)), 'n_objects': int(rng.integers(3, 7)), 'n_plates': int(rng.integers(2, 5))}
    if task == 2:
        return {'task': 2, 'target': str(rng.choice(COSMETICS)), 'dest': str(rng.choice(LEVELS)),
                'template': int(rng.integers(5)), 'n_objects': int(rng.integers(3, 7))}
    return {'task': 3}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--item', type=int, required=True)
    a = p.parse_args()
    if a.item >= DET_ITEMS:
        return
    split = 'train' if a.item < N_TRAIN_ITEMS else 'val'
    D = a.root / 'det'
    for sub in ('images', 'labels', 'meta'):
        (D / sub / split).mkdir(parents=True, exist_ok=True)
    done_flag = D / 'items' / f'{a.item:04d}.done'
    if done_flag.exists():
        return
    rng = np.random.default_rng(10_000 + a.item)
    base_cfg = config(os.environ['ROBOTWIN_DIR'])
    tasks = [1] * 11 + [2] * 6 + [3] * 3
    for j, task in enumerate(tasks):
        name = f'{a.item:04d}_{j:02d}_t{task}'
        if (D / 'meta' / split / f'{name}.json').exists():
            continue
        cfg = json.loads(json.dumps(base_cfg))
        light = bool(j % 2)
        cfg['domain_randomization'] = {**cfg.get('domain_randomization', {}), 'random_light': light, 'crazy_random_light_rate': 0}
        for attempt in range(6):
            spec = spec_for(rng, task)
            seed = (1_000_000 if split == 'val' else 0) + a.item * 1000 + j * 10 + attempt
            env = RealTask()
            try:
                env.setup_demo(spec=spec, seed=seed, now_ep_num=j, **cfg)
                img = env.get_obs()['observation']['head_camera']['rgb']
                bx = boxes(env, strict=False)
                instr = env.get_instruction()
            except Exception as e:
                print(name, 'attempt', attempt, repr(e)[:160], flush=True)
                continue
            finally:
                try: env.close_env(clear_cache=True)
                except Exception: pass
            H, W = img.shape[:2]
            Image.fromarray(img).save(D / 'images' / split / f'{name}.png')
            rows = [f"{CLS[b['cls']]} {(b['box'][0] + b['box'][2]) / 2 / W:.6f} {(b['box'][1] + b['box'][3]) / 2 / H:.6f} "
                    f"{(b['box'][2] - b['box'][0]) / W:.6f} {(b['box'][3] - b['box'][1]) / H:.6f}" for b in bx]
            (D / 'labels' / split / f'{name}.txt').write_text('\n'.join(rows) + '\n')
            (D / 'meta' / split / f'{name}.json').write_text(json.dumps(
                {'spec': spec, 'seed': seed, 'instruction': instr, 'random_light': light, 'boxes': bx}))
            break
    done_flag.parent.mkdir(parents=True, exist_ok=True)
    done_flag.write_text('ok\n')


if __name__ == '__main__':
    main()
