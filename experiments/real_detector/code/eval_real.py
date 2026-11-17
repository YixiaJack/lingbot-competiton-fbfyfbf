"""Paired BF16 rollout in competition-like scenes: first-approach proxy (the recognition score) + task success.

Items 0-29 task 1, 30-59 task 2, new seeds and layouts. A seed is used only if every task object is fully visible and
the scripted expert can solve it (checked in a throwaway env with the same seed, as RoboTwin's own evaluation does);
the policy then gets a fresh env with that seed, so all conditions see the same initial scene (scene_fingerprint).
Cue sources: oracle (simulator boxes), yolo (trained detector, CPU, frame zero), jitter (both boxes shifted by up to
10% of their size), wrong10 (items with item % 10 == 0 point at another candidate).
boxgrasp drops the target box once a gripper has closed: GraspLatch on the state at each inference call and on every
executed action (action = next state), the same rule prepare_data_real.py applies to the recorded demo states.
Usage: eval_real.py --root R --run RUN --mode text|box|boxgrasp|coords --source oracle|yolo|jitter|wrong10 --item I --port P
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scene_real import RealTask, config, boxes, tcp_positions, T1_OBJECTS, COSMETICS, PLATE_COLORS, LEVELS
from cues_real import apply_cue, parse_instruction, select_boxes, run_detector, GraspLatch
from collect_real import oracle_boxes, swapped
sys.path.insert(0, os.environ['LINGBOT_DIR'])
from deploy.websocket_client_policy import WebsocketClientPolicy


def spec_eval(item):
    if item < 60:   # 11/12: all 60 evaluation scenes are task 1 (task-2 expert not working)
        return {'task': 1, 'target': T1_OBJECTS[(item * 7 + 3) % len(T1_OBJECTS)], 'dest': PLATE_COLORS[(item * 3 + 1) % 5],
                'template': (item + 2) % 5, 'n_objects': 3 + item % 3, 'n_plates': 2 + (item // 3) % 2}
    raise ValueError(item)
    j = item - 30
    return {'task': 2, 'target': COSMETICS[(j * 3 + 1) % len(COSMETICS)], 'dest': LEVELS[(j + 1) % 3],
            'template': (j + 1) % 5, 'n_objects': 3 + j % 3}


def feasible_seed(spec, item, cfg):
    """20 seeds with the named target, then 20 with the next object (collect_real.swapped). Returns seed, rejects, spec."""
    rejects, base = [], spec
    for attempt in range(40):
        spec = base if attempt < 20 else swapped(base)
        seed = 500_000 + item * 50 + attempt
        env = RealTask()
        try:
            env.setup_demo(spec=spec, seed=seed, now_ep_num=item, **cfg)
            boxes(env, strict=True)
            env.play_once()
            if env.plan_success and env.check_success():
                return seed, rejects, spec
            rejects.append({'seed': seed, 'reason': f'expert failed ({env.where()})'})
        except Exception as e:
            rejects.append({'seed': seed, 'reason': repr(e)[:160]})
        finally:
            try: env.close_env(clear_cache=True)
            except Exception: pass
    raise RuntimeError(f'no feasible scene for item {item}: {rejects}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--run', required=True)
    p.add_argument('--mode', choices=['text', 'box', 'boxgrasp', 'coords'], required=True)
    p.add_argument('--source', choices=['oracle', 'yolo', 'jitter', 'wrong10'], default='oracle')
    p.add_argument('--item', type=int, required=True); p.add_argument('--port', type=int, default=0)
    p.add_argument('--seed-only', action='store_true', help='pick and store the feasible seed of this item, nothing else')
    p.add_argument('--tag', default='', help='suffix for the result folder, e.g. _mem15 for the 16 GB check')
    a = p.parse_args()
    spec = spec_eval(a.item)
    cfg = config(os.environ['ROBOTWIN_DIR'], eval_mode=True)
    # The expert check is not reproducible between runs (prelim, 10/27), so the seed is chosen once, in its own phase,
    # and every condition reads it: same scene everywhere.
    seed_file = a.root / 'eval_seeds' / f'{a.item:04d}.json'
    if a.seed_only:
        if not seed_file.exists():
            seed, rejects, spec = feasible_seed(spec, a.item, cfg)
            seed_file.parent.mkdir(exist_ok=True)
            tmp = seed_file.with_suffix('.tmp'); tmp.write_text(json.dumps({'seed': seed, 'rejects': rejects, 'spec': spec}))
            tmp.replace(seed_file)
        return
    out = a.root / 'results' / f'{a.run}_{a.mode}_{a.source}{a.tag}' / f'{a.item:04d}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        return
    d = json.loads(seed_file.read_text())   # raises if the seed phase has not run: never pick a seed here
    seed, rejects = d['seed'], d['rejects']
    spec = d.get('spec', spec)   # the target may have been swapped when the seed was chosen
    env = RealTask(); env.setup_demo(spec=spec, seed=seed, now_ep_num=a.item, **cfg)
    obs = env.get_obs(); img0 = obs['observation']['head_camera']['rgb']
    gt = boxes(env, strict=True); instr = env.get_instruction()
    tb, db = oracle_boxes(env, gt)
    detector_ms, reason = None, a.source
    if a.source == 'yolo':
        from ultralytics import YOLO
        det = YOLO(str(a.root / 'det' / 'runs' / 'yolo11s' / 'weights' / 'best.pt'))
        s = time.perf_counter(); dets = run_detector(det, img0, det.names); detector_ms = 1000 * (time.perf_counter() - s)
        tb, db, reason = select_boxes(dets, parse_instruction(instr))
    elif a.source == 'jitter':
        rng = np.random.default_rng(seed + 777)
        def jit(b):
            x1, y1, x2, y2 = b; d = rng.uniform(-.1, .1, 2) * [x2 - x1, y2 - y1]
            return [max(0, x1 + d[0]), max(0, y1 + d[1]), min(320, x2 + d[0]), min(240, y2 + d[1])]
        tb, db = jit(tb), jit(db)
    elif a.source == 'wrong10' and a.item % 10 == 0:
        others = [g['box'] for k, g in zip(env.candidate_keys, gt) if k != spec['target']]
        tb = others[0]
    shown, prompt = apply_cue(img0, instr, tb, db, a.mode)
    Image.fromarray(img0).save(out / 'frame0.png'); Image.fromarray(shown).save(out / 'cue.png')
    policy = WebsocketClientPolicy(port=a.port)
    policy.infer(dict(reset=True, robo_name='robotwin_clean', path_to_pi_model=None, rng_seed=seed + 17))
    pos0 = np.array([c.get_pose().p for c in env.candidates])
    scene = {'objects': pos0.round(4).tolist(), 'keys': env.candidate_keys,
             'plates': [[q['color'], *np.round(q['xy'], 4).tolist()] for q in env.plates],
             'joints': np.asarray(obs['joint_action']['vector']).round(4).tolist()}
    fingerprint = hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()
    ee0 = tcp_positions(env); d0 = np.linalg.norm(ee0[:, None, :] - pos0[None, :, :], axis=-1)
    first, trace, lat, mem, frames, t0 = None, [], [], [], [], time.time()
    latch, grasp_step = GraspLatch(obs['joint_action']['vector']), None   # tracked in every mode, only boxgrasp draws by it
    while env.take_action_cnt < env.step_lim and not env.eval_success:
        o = env.get_obs()
        latch.update(o['joint_action']['vector'])
        high, _ = apply_cue(o['observation']['head_camera']['rgb'], instr, tb, db, a.mode, grasped=latch.on)
        ret = policy.infer({'observation.images.cam_high': high,
                            'observation.images.cam_left_wrist': o['observation']['left_camera']['rgb'],
                            'observation.images.cam_right_wrist': o['observation']['right_camera']['rgb'],
                            'observation.state': o['joint_action']['vector'], 'task': prompt})
        lat.append(ret['server_timing']['infer_ms']); mem.append(ret['server_timing'])
        act = np.asarray(ret['action']); act = act[None] if act.ndim == 1 else act
        for x in act:
            env.take_action(x)
            if latch.update(x) and grasp_step is None:
                grasp_step = env.take_action_cnt
            ee = tcp_positions(env); d = np.linalg.norm(ee[:, None, :] - pos0[None, :, :], axis=-1)
            cand = np.argwhere((d < .10) & ((d0 - d) > .04) & (ee[:, 2, None] < .90))
            if first is None and len(cand):
                arm, obj = min(cand, key=lambda c: d[tuple(c)])
                first = {'object': env.candidate_keys[int(obj)], 'arm': ('left', 'right')[arm], 'step': env.take_action_cnt,
                         'distance_m': float(d[arm, obj])}
            trace.append({'step': env.take_action_cnt, 'ee': ee.round(4).tolist()})
            if env.take_action_cnt % 10 == 0:
                frames.append(env.get_obs()['observation']['head_camera']['rgb'])
            if env.eval_success or env.take_action_cnt >= env.step_lim:
                break
    import cv2
    if frames:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(out / 'rollout.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
        for f in frames: vw.write(f[..., ::-1])
        vw.release()
    res = {'run': a.run, 'mode': a.mode, 'source': a.source + a.tag, 'item': a.item, 'mem_cap_gb': os.environ.get('LINGBOT_MEM_CAP_GB'), 'task': spec['task'], 'seed': seed,
           'seed_rejects': rejects, 'spec': spec, 'instruction': instr, 'prompt': prompt, 'cue_reason': reason,
           'boxes_used': [tb, db], 'cue_applied': a.mode != 'text' and tb is not None and db is not None,
           'gripper_closed_step': grasp_step,   # boxgrasp: the target box is gone from the next inference call on
           'first_approach': first, 'first_approach_correct': first is not None and first['object'] == spec['target'],
           'success': bool(env.eval_success or env.check_success()), 'where': env.where(), 'steps': env.take_action_cnt,
           'seconds': time.time() - t0, 'detector_ms': detector_ms, 'latency_ms': lat, 'memory': mem, 'precision': 'BF16',
           'scene': scene, 'scene_fingerprint': fingerprint,
           'metric_definition': 'first gripper TCP within 10cm of a candidate start, >4cm progress, TCP z<0.90m; none = incorrect'}
    (out / 'trace.json').write_text(json.dumps(trace))
    tmp = out / 'result.tmp'; tmp.write_text(json.dumps(res, indent=1)); tmp.replace(out / 'result.json')
    env.close_env(clear_cache=True)
    print(json.dumps({k: res[k] for k in ('run', 'mode', 'source', 'item', 'task', 'first_approach_correct', 'success', 'where')}), flush=True)


if __name__ == '__main__':
    main()
