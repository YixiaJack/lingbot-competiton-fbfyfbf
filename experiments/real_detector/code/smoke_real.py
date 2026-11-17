"""Smoke test: build scenes for all three tasks, save frame zero with boxes, run the task 1/2 expert."""
import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scene_real import RealTask, config, boxes, T1_OBJECTS, COSMETICS, PLATE_COLORS, LEVELS

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
plan = [(1, int(sys.argv[2]) if len(sys.argv) > 2 else 12), (2, int(sys.argv[3]) if len(sys.argv) > 3 else 9), (3, 3)]
cfg = config(os.environ['ROBOTWIN_DIR'])
res = []
for task, n in plan:
    for i in range(n):
        spec = ({'task': 1, 'target': T1_OBJECTS[(i * 7) % len(T1_OBJECTS)], 'dest': PLATE_COLORS[i % 5], 'template': i % 5}
                if task == 1 else {'task': 2, 'target': COSMETICS[i % 10], 'dest': LEVELS[i % 3], 'template': i % 5}
                if task == 2 else {'task': 3})
        seed, t0, env, r = 5000 + task * 100 + i, time.time(), RealTask(), {}
        try:
            env.setup_demo(spec=spec, seed=seed, now_ep_num=i, **cfg)
            img = env.get_obs()['observation']['head_camera']['rgb']
            Image.fromarray(img).save(out / f't{task}_{i:02d}_frame0.png')
            try:
                bx, berr = boxes(env), None
            except ValueError as e:
                bx, berr = [], str(e)
            im = Image.fromarray(img).convert('RGB'); d = ImageDraw.Draw(im)
            for b in bx:
                d.rectangle(b['box'], outline=(255, 255, 255)); d.text((b['box'][0], b['box'][1] - 9), b['cls'][:10], fill=(255, 255, 0))
            im.resize((640, 480)).save(out / f't{task}_{i:02d}_boxes.png')
            r = {'task': task, 'seed': seed, 'spec': spec, 'instruction': env.get_instruction(), 'box_error': berr,
                 'boxes': bx, 'keys': env.candidate_keys + env.extra_keys}
            if task in (1, 2):
                env.play_once()
                r.update(plan=bool(env.plan_success), success=bool(env.check_success()), where=env.where(),
                         target_final=np.round(env.target.get_pose().p, 3).tolist())
                Image.fromarray(env.get_obs()['observation']['head_camera']['rgb']).save(out / f't{task}_{i:02d}_final.png')
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {'task': task, 'seed': seed, 'spec': spec, 'error': repr(e)[:400]}
        finally:
            try: env.close_env(clear_cache=True)
            except Exception: pass
        r['sec'] = round(time.time() - t0, 1); res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'boxes'}), flush=True)
(out / 'smoke.json').write_text(json.dumps(res, indent=1))
