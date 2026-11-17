"""Stage-by-stage expert diagnosis for task 1/2 scenes: which step loses the plan, under which variant.
Variants (comma list in argv[3]): base, noyaw (objects not rotated), free (place constrain='free'),
short (pre_grasp_dis 0.05).
Usage: debug_expert.py OUT N VARIANTS TASK
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import scene_real as S
from envs.utils import ArmTag

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[2]); variants = sys.argv[3].split(','); task = int(sys.argv[4]) if len(sys.argv) > 4 else 1
cfg = S.config(os.environ['ROBOTWIN_DIR'])
orig_load = S.load_asset


def staged(env, v):
    s, obj = env.spec, env.target
    p0 = np.array(obj.get_pose().p)
    arm = ArmTag('left' if p0[0] < 0 else 'right')
    log = {'arm': str(arm), 'obj': p0.round(3).tolist()}
    pre = 0.05 if 'short' in v else 0.09
    env.move(env.grasp_actor(obj, arm_tag=arm, pre_grasp_dis=pre)); log['grasp'] = bool(env.plan_success)
    env.move(env.move_by_displacement(arm_tag=arm, z=0.10)); log['lift'] = bool(env.plan_success)
    log['lifted_z'] = round(float(obj.get_pose().p[2]), 3)
    if s['task'] == 1:
        px, py = env.dest_plate()['xy']; tz = S.TABLE_TOP + 0.008
    else:
        lv = env.levels[s['dest']]; px, py = sum(lv['x']) / 2, sum(lv['y']) / 2; tz = lv['floor'] + 0.004
        env.move(env.move_by_displacement(arm_tag=arm, z=0.06)); log['lift2'] = bool(env.plan_success)
    log['dest'] = [round(px, 3), round(py, 3), round(tz, 3)]
    env.move(env.place_actor(obj, target_pose=[px, py, tz, 0, 1, 0, 0], arm_tag=arm, functional_point_id=0,
                             pre_dis=0.09, dis=0.02, constrain='free' if 'free' in v else 'align'))
    log['place'] = bool(env.plan_success)
    env.move(env.move_by_displacement(arm_tag=arm, z=0.08)); log['retreat'] = bool(env.plan_success)
    log['success'] = bool(env.check_success()); log['where'] = env.where()
    log['final'] = np.round(obj.get_pose().p, 3).tolist()
    return log


res = []
for v in variants:
    if 'noyaw' in v:
        S.load_asset = lambda scene, key, xy, yaw, name=None, static=False: orig_load(scene, key, xy, 0.0, name, static)
    else:
        S.load_asset = orig_load
    for i in range(N):
        tgt = (S.T1_OBJECTS if task == 1 else S.COSMETICS)[(i * 5) % (36 if task == 1 else 10)]
        spec = ({'task': 1, 'target': tgt, 'dest': S.PLATE_COLORS[i % 5], 'template': 0} if task == 1 else
                {'task': 2, 'target': tgt, 'dest': S.LEVELS[i % 3], 'template': 0})
        env, t0 = S.RealTask(), time.time()
        try:
            env.setup_demo(spec=spec, seed=7000 + i, now_ep_num=i, **cfg)
            r = {'variant': v, 'i': i, 'target': tgt, 'dest': spec['dest'], **staged(env, v)}
            if i < 4:
                Image.fromarray(env.get_obs()['observation']['head_camera']['rgb']).save(out / f'{v}_{task}_{i}_final.png')
        except Exception as e:
            r = {'variant': v, 'i': i, 'target': tgt, 'error': repr(e)[:200]}
        finally:
            try: env.close_env(clear_cache=True)
            except Exception: pass
        r['sec'] = round(time.time() - t0, 1); res.append(r); print(json.dumps(r), flush=True)
(out / f'debug_t{task}.json').write_text(json.dumps(res, indent=1))
