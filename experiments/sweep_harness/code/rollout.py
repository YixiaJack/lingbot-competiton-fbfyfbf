"""One evaluation episode: SweepTask scene + LingBot server + one harness, scored with the official task-3 rubric.

Scenes: eval item i uses the seed chosen by --seed-only (the scripted expert solves it and every object is visible),
held out from the demonstration seeds. Every harness sees the same scene and the same VLA noise seed.
Usage:
  rollout.py --root R --item I --seed-only
  rollout.py --root R --run RUN --harness none|timed|oracle|v1|v2 --item I --port P [--exec 50] [--model deepseek-flash]
"""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import sweep_scene as S
from harness import Clock, Rubric, make_scheduler, LIMIT_S


def feasible_seed(item, cfg):
    rejects = []
    for attempt in range(30):
        seed = 500_000 + item * 50 + attempt
        env = S.SweepTask()
        try:
            env.setup_demo(seed=seed, now_ep_num=item, **cfg)
            ok, which = S.visible(env)
            if not ok:
                raise RuntimeError(f'{which} not fully visible')
            env.play_once()
            if env.plan_success and env.check_success():
                return seed, rejects
            rejects.append({'seed': seed, 'reason': f'expert failed: {getattr(env, "first_fail", None)}'})
        except Exception as e:
            rejects.append({'seed': seed, 'reason': repr(e)[:200]})
        finally:
            try: env.close_env(clear_cache=True)
            except Exception: pass
    raise RuntimeError(f'no feasible scene for item {item}')


def overlay(rgb, lines):
    im = Image.fromarray(rgb).resize((640, 480)); d = ImageDraw.Draw(im)
    for i, s in enumerate(lines):
        d.rectangle([0, 18 * i, 640, 18 * i + 17], fill=(0, 0, 0)); d.text((4, 18 * i + 3), s[:100], fill=(255, 255, 255))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--item', type=int, required=True)
    p.add_argument('--seed-only', action='store_true'); p.add_argument('--run', default='')
    p.add_argument('--harness', choices=['none', 'timed', 'oracle', 'v1', 'v2'], default='none')
    p.add_argument('--port', type=int, default=0); p.add_argument('--exec', type=int, default=50)
    p.add_argument('--model', default='deepseek-flash'); p.add_argument('--thinking', type=int, default=0)
    p.add_argument('--tag', default=''); p.add_argument('--limit', type=float, default=LIMIT_S, help='episode limit (s); smoke tests only')
    a = p.parse_args()
    cfg = S.config(os.environ['ROBOTWIN_DIR'], eval_mode=True)
    seed_file = a.root / 'eval_seeds' / f'{a.item:04d}.json'
    if a.seed_only:
        if not seed_file.exists():
            seed, rejects = feasible_seed(a.item, cfg)
            seed_file.parent.mkdir(exist_ok=True)
            tmp = seed_file.with_suffix('.tmp'); tmp.write_text(json.dumps({'seed': seed, 'rejects': rejects})); tmp.replace(seed_file)
        return
    out = a.root / 'results' / f'{a.run}_{a.harness}{a.tag}' / f'{a.item:04d}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        return
    seed = json.loads(seed_file.read_text())['seed']   # raises if the seed phase has not run
    sys.path.insert(0, os.environ['LINGBOT_DIR'])
    from deploy.websocket_client_policy import WebsocketClientPolicy
    timing = json.loads((a.root / 'timing.json').read_text()) if a.harness == 'timed' else None

    env = S.SweepTask(); env.setup_demo(seed=seed, now_ep_num=a.item, **cfg)
    obs = env.get_obs()
    Image.fromarray(obs['observation']['head_camera']['rgb']).save(out / 'frame0.png')
    scene = {k: (np.round(v, 4).tolist() if not isinstance(v, dict) else {c: np.round(q, 4).tolist() for c, q in v.items()})
             for k, v in env.layout.items()}
    policy = WebsocketClientPolicy(port=a.port)
    policy.infer(dict(reset=True, robo_name='robotwin_clean', path_to_pi_model=None, rng_seed=seed + 17))
    # warm-up call before the clock starts: the first call compiles the model (minutes); its action is discarded
    policy.infer({'observation.images.cam_high': obs['observation']['head_camera']['rgb'],
                  'observation.images.cam_left_wrist': obs['observation']['left_camera']['rgb'],
                  'observation.images.cam_right_wrist': obs['observation']['right_camera']['rgb'],
                  'observation.state': obs['joint_action']['vector'], 'task': 'warm up'})
    policy.infer(dict(reset=True, robo_name='robotwin_clean', path_to_pi_model=None, rng_seed=seed + 17))
    clock, rubric = Clock(limit=a.limit), Rubric()
    sched = make_scheduler(a.harness, timing=timing, model=a.model, thinking=bool(a.thinking))
    frames, prompts_used, vla_ms, err, t0 = [], [], [], None, time.time()
    try:
        sched.start(obs, clock, env)
        rubric.update(env.progress(), clock.now)
        while not clock.expired and not sched.done:
            o = env.get_obs()
            sched.on_chunk(o, clock, env)
            if sched.done:
                break
            prompt = sched.prompt()
            prompts_used.append({'t': round(clock.now, 2), 'prompt': prompt})
            ret = policy.infer({'observation.images.cam_high': o['observation']['head_camera']['rgb'],
                                'observation.images.cam_left_wrist': o['observation']['left_camera']['rgb'],
                                'observation.images.cam_right_wrist': o['observation']['right_camera']['rgb'],
                                'observation.state': o['joint_action']['vector'], 'task': prompt})
            ms = ret['server_timing']['infer_ms']; vla_ms.append(ms); clock.block(ms / 1000)
            act = np.asarray(ret['action']); act = act[None] if act.ndim == 1 else act
            for i, x in enumerate(act[:a.exec]):
                env.take_action(x); clock.tick()
                if clock.actions % 5 == 0:
                    rubric.update(env.progress(), clock.now)
                if clock.actions % 8 == 0:
                    frames.append(overlay(env.get_obs()['observation']['head_camera']['rgb'],
                                          [f'{a.harness}  t={clock.now:5.1f}s', prompt]))
                if clock.expired:
                    break
    except Exception:
        err = traceback.format_exc()[-2000:]
    finally:
        sched.close()
    f = env.progress()
    rubric.update(f, clock.now)
    score = rubric.final(f, clock.now, limit=a.limit)
    if frames:
        import cv2
        vw = cv2.VideoWriter(str(out / 'rollout.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
        for fr in frames: vw.write(fr[..., ::-1])
        vw.release()
    Image.fromarray(env.get_obs()['observation']['head_camera']['rgb']).save(out / 'last.png')
    res = {'run': a.run, 'harness': a.harness + a.tag, 'item': a.item, 'seed': seed, 'model': a.model if a.harness in ('v1', 'v2') else None,
           **score, 'episode_s': round(clock.now, 2), 'actions': clock.actions, 'blocked_s': round(clock.blocked, 2),
           'ended_by': 'harness' if sched.done else ('time' if clock.expired else 'error'),
           'vla_ms_mean': float(np.mean(vla_ms)) if vla_ms else None, 'n_chunks': len(vla_ms),
           'final_state': {k: f[k] for k in ('in_pan', 'in_basket', 'tools_down', 'grippers_open', 'arms_home', 'balls_xyz', 'pan_xyz')},
           'scene': scene, 'scene_fingerprint': hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()[:16],
           'wall_s': round(time.time() - t0, 1), 'error': err}
    (out / 'harness.json').write_text(json.dumps({'prompts': prompts_used, **sched.summary()}, indent=1, default=str))
    tmp = out / 'result.tmp'; tmp.write_text(json.dumps(res, indent=1)); tmp.replace(out / 'result.json')
    env.close_env(clear_cache=True)
    print(json.dumps({k: res[k] for k in ('harness', 'item', 'score', 'success', 'process_points', 'episode_s', 'ended_by')}), flush=True)


if __name__ == '__main__':
    main()
