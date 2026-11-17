import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scene import TargetCue, config, boxes, TRAIN_COMBOS, tcp_positions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--item', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    out = args.root / 'episodes' / f'{args.item:04d}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists() and json.loads((out / 'complete.json').read_text()).get('metric_version') == 2:
        return
    target, destination = TRAIN_COMBOS[args.item % len(TRAIN_COMBOS)]
    cfg = config(os.environ['ROBOTWIN_DIR'])
    errors = []
    for attempt in range(8):
        env = TargetCue()
        seed = 1000 + args.item * 20 + attempt
        try:
            env.setup_demo(target=target, destination=destination, seed=seed, now_ep_num=args.item, **cfg)
            obs = env.get_obs()
            b = boxes(env)
            Image.fromarray(obs['observation']['head_camera']['rgb']).save(out / 'frame0.png')
            meta = {'seed': seed, 'target': target, 'destination': destination, 'boxes': b,
                    'instruction': env.get_instruction(), 'split': 'train', 'proxy': 'colored_blocks_and_plates',
                    'scene_version':2, 'metric_version':2}
            positions=np.array([obj.get_pose().p for obj in env.blocks])
            ee0=tcp_positions(env)
            d0=np.linalg.norm(ee0[:,None,:]-positions[None,:,:],axis=-1)
            (out / 'scene.json').write_text(json.dumps(meta, indent=2))
            env.save_data = True
            env._take_picture()
            env.play_once()
            if not env.plan_success or not env.check_success() or len(env.frames) < 10:
                raise RuntimeError(f'expert failed: plan={env.plan_success}, frames={len(env.frames)}')
            arrays = {k: np.stack([f[k] for f in env.frames]) for k in env.frames[0]}
            first=None
            for step,ee in enumerate(arrays['ee']):
                dist=np.linalg.norm(ee[:,None,:]-positions[None,:,:],axis=-1)
                candidates=np.argwhere((dist<.10)&((d0-dist)>.04)&(ee[:,2,None]<.90))
                if len(candidates):
                    arm,obj=min(candidates,key=lambda x:dist[tuple(x)])
                    first={'object':int(obj),'frame':step,'distance_m':float(dist[arm,obj])}
                    break
            meta['expert_first_approach']=first
            meta['expert_first_approach_correct']=first is not None and first['object']==target
            tmp = out / 'episode.tmp.npz'
            np.savez_compressed(tmp, **arrays)
            tmp.replace(out / 'episode.npz')
            meta['frames'] = len(env.frames)
            (out / 'complete.json').write_text(json.dumps(meta, indent=2))
            print(json.dumps(meta), flush=True)
            return
        except Exception as exc:
            import traceback
            traceback.print_exc()
            errors.append({'seed': seed, 'error': repr(exc)})
            (out / 'errors.json').write_text(json.dumps(errors, indent=2))
        finally:
            try:
                env.close_env(clear_cache=True)
            except Exception:
                pass
    raise RuntimeError(f'Failed 8 attempts for item {args.item}; see {out}')


if __name__ == '__main__':
    main()
