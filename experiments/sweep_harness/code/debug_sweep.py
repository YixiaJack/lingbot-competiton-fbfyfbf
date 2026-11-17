"""Debug the sweeping scene and expert, stage by stage.
  axis: which top-down contact yaw grasps each handle (grasp, lift 5 cm, did the tool come along?)
  full: run the expert on seeds, save head-camera video + per-stage flags
Usage: debug_sweep.py OUT MODE [SEEDS] [PAN_CP BRUSH_CP]
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import sweep_scene as S
from envs.utils import ArmTag

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
mode = sys.argv[2]
seeds = [int(s) for s in sys.argv[3].split(',')] if len(sys.argv) > 3 else [1]
if len(sys.argv) > 5:
    S.SweepTask.PAN_CP, S.SweepTask.BRUSH_CP = int(sys.argv[4]), int(sys.argv[5])
cfg = S.config(os.environ['ROBOTWIN_DIR'])


def make(seed, save=False):
    env = S.SweepTask()
    env.setup_demo(seed=seed, now_ep_num=seed, **cfg)
    env.save_data = save
    return env


def video(frames, path, key='cam_high', every=2):
    import cv2
    h, w = frames[0][key].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (w * 3, h))
    for f in frames[::every]:
        vw.write(np.concatenate([f['cam_high'], f['cam_left_wrist'], f['cam_right_wrist']], 1)[..., ::-1])
    vw.release()


res = []
if mode == 'axis':
    for tool, arm in (('pan', 'left'), ('brush', 'right')):
        for cp in range(4):
            env, r = None, {'tool': tool, 'cp': cp}
            try:
                env = make(seeds[0])
                actor = getattr(env, tool)
                z0 = float(actor.get_pose().p[2])
                env.move(env.grasp_actor(actor, arm_tag=ArmTag(arm), pre_grasp_dis=0.08, contact_point_id=cp))
                r['grasp_plan'] = bool(env.plan_success)
                r['contacts_closed'] = env.gripper_contacts()
                r['gripper'] = [round(env.robot.get_left_gripper_val(), 3), round(env.robot.get_right_gripper_val(), 3)]
                env.move(env.move_by_displacement(ArmTag(arm), z=0.05))
                r['lift_plan'] = bool(env.plan_success)
                r['dz'] = round(float(actor.get_pose().p[2]) - z0, 4)
                r['tilt_deg'] = round(float(np.degrees(np.arccos(np.clip(S.mat(actor.get_pose())[2, 2], -1, 1)))), 1)
                Image.fromarray(env.get_obs()['observation']['head_camera']['rgb']).save(out / f'axis_{tool}_{cp}.png')
            except Exception as e:
                r['error'] = repr(e)[:300]; traceback.print_exc()
            finally:
                if env: env.close_env(clear_cache=True)
            print(json.dumps(r), flush=True); res.append(r)
    names = set()
else:
    for seed in seeds:
        env, r, t0 = None, {'seed': seed}, time.time()
        try:
            env = make(seed, save=True)
            obs = env.get_obs()
            Image.fromarray(obs['observation']['head_camera']['rgb']).save(out / f'seed{seed}_frame0.png')
            r['visible'] = S.visible(env)
            r['arm_bases'] = [np.round(env.robot.left_entity.get_root_pose().p, 3).tolist(), np.round(env.robot.right_entity.get_root_pose().p, 3).tolist()]
            r['layout'] = {k: (np.round(v, 3).tolist() if not isinstance(v, dict) else {c: np.round(p, 3).tolist() for c, p in v.items()})
                           for k, v in env.layout.items()}
            env._take_picture()
            env.debug_moves = []
            info = env.play_once()
            r['moves'] = env.debug_moves
            r['info'] = info['info']; r['plan_success'] = bool(env.plan_success); r['first_fail'] = getattr(env, 'first_fail', None)
            f = env.progress()
            r['final'] = {k: f[k] for k in ('in_pan', 'in_basket', 'tools_down', 'grippers_open', 'arms_home', 'pan_xyz', 'balls_xyz')}
            r['success'] = bool(env.check_success())
            r['frames'] = len(env.frames)
            r['skills'] = env.skill_log
            if env.frames:
                video(env.frames, out / f'seed{seed}.mp4')
                Image.fromarray(env.frames[-1]['cam_high']).save(out / f'seed{seed}_last.png')
        except Exception as e:
            r['error'] = repr(e)[:400]; traceback.print_exc()
        finally:
            if env: env.close_env(clear_cache=True)
        r['sec'] = round(time.time() - t0, 1)
        print(json.dumps(r), flush=True); res.append(r)
(out / f'{mode}.json').write_text(json.dumps(res, indent=1))
