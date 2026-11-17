"""Plug-in check on the official LingBot-VLA 2.0 6B base (the model the final requires), BF16 as on the RTX 5080.

Builds competition-like task 1/2 scenes, gets the two boxes (oracle, or the trained detector on CPU), applies the cue,
and drives the robot through the stock LingBot websocket server for a few action chunks. Checks that the whole chain
runs and records detector time, per-call inference latency and peak GPU memory. The base model is not fine-tuned on
these scenes, so task success is not the point here.
Usage: integ_6b.py --root R --port P --source oracle|yolo --mode box|coords|text --episodes N
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scene_real import RealTask, config, boxes, CLASSES, T1_OBJECTS, COSMETICS, PLATE_COLORS, LEVELS
from cues_real import parse_instruction, apply_cue, select_boxes, run_detector
sys.path.insert(0, os.environ['LINGBOT_DIR'])
from deploy.websocket_client_policy import WebsocketClientPolicy


def oracle_boxes(env, gt):
    """True target box and destination box. In strict mode gt follows labeled_entities order:
    candidates, plates, organizer levels; the destination plate is the named-colour plate nearest the target."""
    s = env.spec
    tb = gt[env.candidate_keys.index(s['target'])]['box']
    if s['task'] == 1:
        db = gt[len(env.candidates) + env.plates.index(env.dest_plate())]['box']
    else:
        db = next(b['box'] for b in gt if b['cls'] == f"organizer_{s['dest']}")
    return tb, db


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--port', type=int, required=True)
    p.add_argument('--source', default='oracle'); p.add_argument('--mode', default='box')
    p.add_argument('--episodes', type=int, default=4); p.add_argument('--chunks', type=int, default=4)
    a = p.parse_args()
    out = a.root / 'integ' / f'{a.source}_{a.mode}'; out.mkdir(parents=True, exist_ok=True)
    detector = None
    if a.source == 'yolo':
        from ultralytics import YOLO
        detector = YOLO(str(a.root / 'det' / 'runs' / 'yolo11s' / 'weights' / 'best.pt'))
    cfg = config(os.environ['ROBOTWIN_DIR'], eval_mode=True)
    policy = WebsocketClientPolicy(port=a.port)
    rows = []
    for i in range(a.episodes):
        task = 1 if i % 2 == 0 else 2
        spec = ({'task': 1, 'target': T1_OBJECTS[(i * 11) % 36], 'dest': PLATE_COLORS[i % 5], 'template': i % 5} if task == 1
                else {'task': 2, 'target': COSMETICS[(i * 3) % 10], 'dest': LEVELS[i % 3], 'template': i % 5})
        gt, why = None, []
        for attempt in range(20):
            env = RealTask()
            try:   # RoboTwin raises UnStableError when an object settles badly (e.g. a star fruit tipping over)
                env.setup_demo(spec=spec, seed=90_000 + i * 100 + attempt, now_ep_num=i, **cfg)
                gt = boxes(env, strict=True); break
            except Exception as e:
                why.append(repr(e)[:120])
                try: env.close_env(clear_cache=True)
                except Exception: pass
        if gt is None:
            raise RuntimeError(f'episode {i}: no fully visible scene in 20 seeds: {why[:5]}')
        obs = env.get_obs(); img0 = obs['observation']['head_camera']['rgb']
        instr = env.get_instruction(); parsed = parse_instruction(instr)
        det_ms = None
        if detector is not None:
            s = time.perf_counter(); dets = run_detector(detector, img0, detector.names); det_ms = 1000 * (time.perf_counter() - s)
            tb, db, reason = select_boxes(dets, parsed)
        else:
            tb, db = oracle_boxes(env, gt)
            reason = 'oracle'
        shown, prompt = apply_cue(img0, instr, tb, db, a.mode)
        Image.fromarray(shown).save(out / f'{i}_cue.png')
        policy.infer(dict(reset=True, robo_name='robotwin_clean', path_to_pi_model=None, rng_seed=1000 + i))
        timing, shapes = [], []
        for c in range(a.chunks):
            o = env.get_obs()
            high, _ = apply_cue(o['observation']['head_camera']['rgb'], instr, tb, db, a.mode)
            ret = policy.infer({'observation.images.cam_high': high,
                                'observation.images.cam_left_wrist': o['observation']['left_camera']['rgb'],
                                'observation.images.cam_right_wrist': o['observation']['right_camera']['rgb'],
                                'observation.state': o['joint_action']['vector'], 'task': prompt})
            timing.append(ret['server_timing']); act = np.asarray(ret['action']); shapes.append(list(act.shape))
            for x in (act if act.ndim == 2 else act[None]):
                env.take_action(x)
        rows.append({'episode': i, 'task': task, 'instruction': instr, 'parsed': parsed, 'cue_reason': reason,
                     'cue_applied': tb is not None and db is not None and a.mode != 'text', 'prompt': prompt,
                     'detector_ms': det_ms, 'action_shapes': shapes,
                     'infer_ms': [t['infer_ms'] for t in timing], 'peak_allocated_gib': max(t['peak_allocated_gib'] for t in timing),
                     'peak_reserved_gib': max(t['peak_reserved_gib'] for t in timing)})
        env.close_env(clear_cache=True)
        print(json.dumps(rows[-1]), flush=True)
    warm = [x for r in rows for x in r['infer_ms'][1:]]
    res = {'model': 'official lingbot-vla-v2-6b base', 'precision': 'BF16', 'source': a.source, 'mode': a.mode,
           'episodes': rows, 'warm_infer_ms_median': float(np.median(warm)) if warm else None,
           'peak_reserved_gib_max': max(r['peak_reserved_gib'] for r in rows),
           'note': 'measured on an HPC GPU, not the RTX 5080; memory is the PyTorch peak inside the server process'}
    (out / 'result.json').write_text(json.dumps(res, indent=1))
    (a.root / 'integ' / 'INTEG_OK').write_text(f'{a.source}_{a.mode}\n')


if __name__ == '__main__':
    main()
