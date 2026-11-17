"""Planner reachability of the carry and pour poses (no motion executed), after the left arm has grasped the dustpan.
Also tries the two handle grasps on several seeds. Usage: probe_reach.py OUT SEEDS
"""
import json, os, sys
from pathlib import Path
import numpy as np
import transforms3d as t3d
import sweep_scene as S
from envs.utils import ArmTag

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
seeds = [int(s) for s in sys.argv[2].split(',')]
cfg = S.config(os.environ['ROBOTWIN_DIR'])
res = []
for seed in seeds:
    env = S.SweepTask(); env.setup_demo(seed=seed, now_ep_num=seed, **cfg)
    r = {'seed': seed, 'layout': {k: np.round(v, 3).tolist() for k, v in env.layout.items() if k != 'balls'}}
    for tool, arm, cp in (('pan', 'left', env.PAN_CP), ('brush', 'right', env.BRUSH_CP)):
        try:
            pre, pose = env.choose_grasp_pose(getattr(env, tool), arm_tag=ArmTag(arm), pre_dis=0.08, contact_point_id=cp)
            r[f'{tool}_grasp'] = pre is not None
        except Exception as e:
            r[f'{tool}_grasp'] = repr(e)[:120]
    if r['pan_grasp'] is not True:
        env.close_env(clear_cache=True); print(json.dumps(r), flush=True); res.append(r); continue
    env.move(env.grasp_actor(env.pan, arm_tag=ArmTag('left'), pre_grasp_dis=0.08, contact_point_id=env.PAN_CP))
    T_pan = env.held(env.pan, 'left')
    grip_local = (np.linalg.inv(S.mat(env.pan_e.get_pose())) @ np.append(env.robot.get_left_tcp_pose()[:3], 1))[:3]
    g0 = np.array(env.robot.get_left_tcp_pose()[:3])
    plan = lambda m: env.robot.left_plan_path(env.tool_target('left', T_pan, m))['status'] == 'Success'
    r['lift'] = {f'tilt{d}': plan(env.pan_pose(g0 + [0, 0, 0.10], np.deg2rad(d), grip_local)) for d in (0, -8, -15, 8, 15)}
    mouth = np.array([0, -S.PAN_D / 2, 0.01])
    pour = {}
    for roll in (60, 80, 95, 110):
        Rp = t3d.euler.euler2mat(0, 0, env.layout['pan'][2]) @ t3d.axangles.axangle2mat([1, 0, 0], np.deg2rad(roll))
        off = Rp @ (mouth - grip_local)
        for dz in (0.03, 0.05, 0.08):
            g = np.array([env.basket_xy[0] - off[0], env.basket_xy[1] - off[1], S.TABLE_TOP + S.BASKET_H + dz - off[2]])
            pour[f'r{roll}_dz{dz}'] = [plan(env.pan_pose(g, np.deg2rad(roll), grip_local)), np.round(g, 3).tolist()]
            pour[f'r{roll}_dz{dz}_level'] = plan(env.pan_pose(g, 0.0, grip_local))
    r['pour'] = pour
    env.close_env(clear_cache=True)
    print(json.dumps(r), flush=True); res.append(r)
(out / 'probe.json').write_text(json.dumps(res, indent=1))
