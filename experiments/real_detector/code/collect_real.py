"""One successful expert demonstration per item, tasks 1 and 2, in competition-like scenes.

Items 0..N1-1 are task 1, the rest task 2. Each item fixes the target, destination and instruction template (cycled so
every object is a target and every plate colour / level a destination); the layout comes from the seed. A seed is kept
only if every task object is fully visible in frame zero (the rules promise this) and the expert succeeds.
Saved per item: episode.npz (14-d joint state, both TCPs, three cameras per frame), frame0.png, complete.json with the
oracle cue boxes (target, and the destination plate the expert used / the named level).
Usage: collect_real.py --root R --item i
"""
import argparse, json, os, traceback
from pathlib import Path
import numpy as np
from PIL import Image
from scene_real import RealTask, config, boxes, T1_OBJECTS, COSMETICS, PLATE_COLORS, LEVELS
from catalog import N1, N2


def spec_for(item):
    if item < N1:
        return {'task': 1, 'target': T1_OBJECTS[item % len(T1_OBJECTS)], 'dest': PLATE_COLORS[(item // 2) % 5],
                'template': item % 5, 'n_objects': 3 + item % 3, 'n_plates': 2 + item % 2}
    j = item - N1
    return {'task': 2, 'target': COSMETICS[j % len(COSMETICS)], 'dest': LEVELS[(j // 3) % 3],
            'template': j % 5, 'n_objects': 3 + j % 3}


def swapped(spec):
    """Same item with the next object of the list as target: used only when the scripted expert cannot do the named
    one in any seed (11/12: cherry, 2.4 cm with a stem, 1 success in ~200 seeds). Recorded in complete.json."""
    s = dict(spec); objs = T1_OBJECTS if s['task'] == 1 else COSMETICS
    s['target_swapped_from'] = s['target']; s['target'] = objs[(objs.index(s['target']) + 1) % len(objs)]
    return s


def oracle_boxes(env, gt):
    s = env.spec
    tb = gt[env.candidate_keys.index(s['target'])]['box']
    if s['task'] == 1:
        db = gt[len(env.candidates) + env.plates.index(env.dest_plate())]['box']
    else:
        db = next(b['box'] for b in gt if b['cls'] == f"organizer_{s['dest']}")
    return tb, db


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--item', type=int, required=True)
    a = p.parse_args()
    out = a.root / 'episodes' / f'{a.item:04d}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    cfg = config(os.environ['ROBOTWIN_DIR'])
    errors = []
    for attempt in range(24):   # 12 seeds with the named target, then 12 with the next object (seeds are fixed, so a
        spec = spec_for(a.item) if attempt < 12 else swapped(spec_for(a.item))   # rerun would only repeat failures)
        seed = 200_000 + a.item * 50 + attempt
        env = RealTask()
        try:
            env.setup_demo(spec=spec, seed=seed, now_ep_num=a.item, **cfg)
            obs = env.get_obs()
            gt = boxes(env, strict=True)
            tb, db = oracle_boxes(env, gt)
            Image.fromarray(obs['observation']['head_camera']['rgb']).save(out / 'frame0.png')
            positions = np.array([c.get_pose().p for c in env.candidates])
            env.save_data = True
            env._take_picture()
            env.play_once()
            if not env.plan_success or not env.check_success() or len(env.frames) < 10:
                raise RuntimeError(f'expert failed: plan={env.plan_success} where={env.where()} frames={len(env.frames)}')
            arrays = {k: np.stack([f[k] for f in env.frames]) for k in env.frames[0]}
            # validate the first-approach proxy on the expert itself (the proxy experiment did this; the expert must
            # read as correct, otherwise the metric, not the policy, is measured)
            d0 = np.linalg.norm(arrays['ee'][0][:, None, :] - positions[None, :, :], axis=-1)
            first = None
            for step, ee in enumerate(arrays['ee']):
                dist = np.linalg.norm(ee[:, None, :] - positions[None, :, :], axis=-1)
                cand = np.argwhere((dist < .10) & ((d0 - dist) > .04) & (ee[:, 2, None] < .90))
                if len(cand):
                    arm, obj = min(cand, key=lambda c: dist[tuple(c)])
                    first = {'object': env.candidate_keys[int(obj)], 'frame': step, 'distance_m': float(dist[arm, obj])}
                    break
            tmp = out / 'episode.tmp.npz'; np.savez_compressed(tmp, **arrays); tmp.replace(out / 'episode.npz')
            meta = {'seed': seed, 'spec': spec, 'instruction': env.get_instruction(), 'candidates': env.candidate_keys,
                    'candidate_positions': positions.round(4).tolist(), 'boxes': gt, 'target_box': tb, 'dest_box': db,
                    'frames': len(env.frames), 'scene_version': 'real_v2', 'expert_first_approach': first,
                    'expert_first_approach_correct': first is not None and first['object'] == spec['target']}
            (out / 'complete.json').write_text(json.dumps(meta, indent=1))
            print(json.dumps({k: meta[k] for k in ('seed', 'instruction', 'frames')}), flush=True)
            return
        except Exception as exc:
            traceback.print_exc()
            errors.append({'seed': seed, 'error': repr(exc)[:300]})
            (out / 'errors.json').write_text(json.dumps(errors, indent=1))
        finally:
            try: env.close_env(clear_cache=True)
            except Exception: pass
    raise RuntimeError(f'item {a.item}: no successful demonstration in 24 seeds (named target and next object)')


if __name__ == '__main__':
    main()
