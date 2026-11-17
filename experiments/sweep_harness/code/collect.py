"""One successful expert demonstration per item (the whole task, every skill labelled per frame).
Kept only if every object is visible at frame zero, every planned move succeeded, both balls end in the basket, and
both tools are down with the arms home. Saved per item: episode.npz (14-d joint state, three cameras, skill label,
compact simulator flags and ball positions per frame) and complete.json.
Usage: collect.py --root R --item i
"""
import argparse, json, os, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import sweep_scene as S

FLAG_NAMES = ['in_pan_red', 'in_pan_white', 'in_basket_red', 'in_basket_white', 'pan_held_left', 'brush_held_right',
              'pan_lifted', 'pan_over_basket']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--item', type=int, required=True)
    a = p.parse_args()
    out = a.root / 'episodes' / f'{a.item:04d}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    cfg = S.config(os.environ['ROBOTWIN_DIR'])
    errors = []
    for attempt in range(16):
        seed = 200_000 + a.item * 50 + attempt
        env = S.SweepTask()
        try:
            env.setup_demo(seed=seed, now_ep_num=a.item, **cfg)
            ok, which = S.visible(env)
            if not ok:
                raise RuntimeError(f'{which} not fully visible')
            env.save_data = True
            env._take_picture()
            info = env.play_once()
            f = env.progress()
            if not (env.plan_success and f['all_in_basket'] and f['tools_down'] and f['grippers_open'] and f['arms_home']):
                raise RuntimeError(f"expert failed: {info['info']} {getattr(env, 'first_fail', None)} basket={f['in_basket']} "
                                   f"down={f['tools_down']} home={f['arms_home']}")
            arrays = {k: np.stack([fr[k] for fr in env.frames]) for k in ('state', 'cam_high', 'cam_left_wrist', 'cam_right_wrist', 'flags', 'balls_xy')}
            arrays['skill'] = np.array([fr['skill'] or 'grasp_dustpan' for fr in env.frames])
            Image.fromarray(arrays['cam_high'][0]).save(out / 'frame0.png')
            tmp = out / 'episode.tmp.npz'; np.savez_compressed(tmp, **arrays); tmp.replace(out / 'episode.npz')
            segs, start = [], 0
            for j in range(1, len(arrays['skill']) + 1):
                if j == len(arrays['skill']) or arrays['skill'][j] != arrays['skill'][start]:
                    segs.append({'skill': str(arrays['skill'][start]), 'start': start, 'end': j}); start = j
            meta = {'seed': seed, 'layout': {k: (np.round(v, 4).tolist() if not isinstance(v, dict) else
                                                 {c: np.round(q, 4).tolist() for c, q in v.items()}) for k, v in env.layout.items()},
                    'frames': len(env.frames), 'segments': segs, 'flag_names': FLAG_NAMES, 'scene_version': 'sweep_v1'}
            (out / 'complete.json').write_text(json.dumps(meta, indent=1))
            print(json.dumps({'item': a.item, 'seed': seed, 'frames': len(env.frames), 'skills': [s['skill'] for s in segs]}), flush=True)
            return
        except Exception as exc:
            traceback.print_exc()
            errors.append({'seed': seed, 'error': repr(exc)[:400]})
            (out / 'errors.json').write_text(json.dumps(errors, indent=1))
        finally:
            try: env.close_env(clear_cache=True)
            except Exception: pass
    raise RuntimeError(f'item {a.item}: no successful demonstration in 16 seeds')


if __name__ == '__main__':
    main()
