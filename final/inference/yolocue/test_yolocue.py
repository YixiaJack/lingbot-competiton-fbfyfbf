"""Offline checks of yolocue on the 5080 with the inference env's python; the running service is not touched.

  1. detector vs ultralytics: every frame-0 image the training boxes were computed on (~/locany/label/first0) against
     the run's dets_first.json (same weights, ultralytics predict conf 0.01 on the GPU);
  2. cue decision: parse + select on our detections vs on dets_first.json, for every episode whose instruction parses;
  3. drawing vs derive_box_dataset.draw, pixel for pixel;
  4. a fake session through YoloCue: begin, preprocess (cue drawn, mailbox frame untouched), postprocess (open, then
     close -> white box gone from the next frame), close (policy methods restored); shadow mode leaves the image raw;
     task 3 / unknown objects / wrong frame size give no cue;
  5. detection latency.
usage: python -I test_yolocue.py [--run v5] [--n 0] [--labels ~/locany/data/merged/meta/labels.jsonl]
"""
import argparse
import json
import logging
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yolocue_bridge as YC  # noqa: E402

LOG = logging.getLogger('yolocue-test')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


def ref_draw(rgb, tb, db, grasped):
    """experiments/autolabel/derive_box_dataset.py draw(), verbatim."""
    from PIL import Image, ImageDraw
    WHITE, ORANGE, LINE = (255, 255, 255), (255, 128, 0), 5
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    for box, col, skip in ((tb, WHITE, grasped), (db, ORANGE, False)):
        if skip:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        d.rectangle([x1, y1, x2 - 1, y2 - 1], outline=col, width=LINE)
    return np.asarray(im)


def load_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def compare(mine, ref, floor=0.1):
    """Rows with conf >= floor: every ref row needs a same-class row of ours (and vice versa). -> (ok, dconf, dxy)"""
    A = [r for r in mine if r[5] >= floor]; B = [r for r in ref if r[5] >= floor]
    dconf = dxy = 0.0; ok = True
    for X, Y in ((B, A), (A, B)):
        for r in X:
            cand = [q for q in Y if q[0] == r[0]]
            if not cand:
                if r[5] >= floor + 0.01:   # rows right at the floor may sit on either side of it
                    ok = False
                continue
            q = max(cand, key=lambda q: iou(r[1:5], q[1:5]))
            dconf = max(dconf, abs(q[5] - r[5])); dxy = max(dxy, max(abs(u - v) for u, v in zip(q[1:5], r[1:5])))
    return ok and dconf <= 0.01 and dxy <= 1.0, dconf, dxy


class FakePolicy:
    def __init__(self, task):
        self._task = task

    def preprocess(self, model_input, *, run_inference, rtc_prefix_len=None, prev_action_chunk=None):
        if not run_inference:
            return None
        return {'task': self._task, YC.CAM: np.asarray(model_input.features[YC.CAM]).copy()}

    def postprocess(self, model_output):
        return types.SimpleNamespace(actions={k: np.asarray(v, np.float32).reshape(-1, 1) for k, v in model_output.items()})


def fake_session(frame, task, mode, log_dir):
    mi = types.SimpleNamespace(features={YC.CAM: frame})
    server = types.SimpleNamespace(_robot_runtime=types.SimpleNamespace(mailbox=types.SimpleNamespace(get=lambda: mi)))
    session = types.SimpleNamespace(session_id=f'test-{mode}-{time.time_ns()}', task=task)
    pol = FakePolicy(task)
    cue = YC.YoloCue(server=server, session=session, policy=pol, mode=mode, logger=LOG, log_dir=log_dir)
    return cue, pol, mi


def chunk(left, right):
    return {'action.left_gripper.position': left, 'action.right_gripper.position': right}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v5'); ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--labels', default=str(Path.home() / 'locany/data/merged/meta/labels.jsonl'))
    ap.add_argument('--device', help='override config.json (e.g. cuda to compare with the GPU-made dets_first)')
    ap.add_argument('--json', help='also write the result here')
    a = ap.parse_args()
    out, fails = {}, []
    cfg = YC.load_config()
    if a.device:
        cfg['device'] = a.device
    t = time.perf_counter(); det = YC.detector(cfg); out['load_s'] = round(time.perf_counter() - t, 2)
    ref = json.loads((Path.home() / f'locany/yolo/runs/{a.run}/dets_first.json').read_text())
    img_dir = Path.home() / 'locany/label/first0/images'
    eps = sorted(int(p.name[2:]) for p in img_dir.iterdir())[:a.n or None]

    # 1. detections
    mine, ms, exact, close_, worst = {}, [], 0, 0, {'dconf': 0.0, 'dxy': 0.0}
    bad = []
    for e in eps:
        rgb = load_rgb(img_dir / f'ep{e:03d}/0000.jpg')
        t = time.perf_counter(); d = det(rgb); ms.append((time.perf_counter() - t) * 1000)
        mine[e] = d; r = ref.get(f'first/{e:04d}', [])
        if d == r:
            exact += 1; close_ += 1; continue
        ok, dc, dx = compare(d, r)
        worst['dconf'] = max(worst['dconf'], dc); worst['dxy'] = max(worst['dxy'], dx)
        close_ += ok
        if not ok:
            bad.append(e)
    out['1_detections'] = {'frames': len(eps), 'identical': exact, 'within_tol': close_, 'worst': worst, 'off_frames': bad[:20]}
    if close_ < len(eps):
        fails.append('detections differ from dets_first beyond tolerance')

    # 2. cue decisions (the rule actually used for training boxes)
    ins = {}
    for line in open(a.labels):
        r = json.loads(line); ins[r['episode_index']] = r['labels'][0]['task_instruction']
    agree, total, diff, reasons, subpx, px_moved = 0, 0, [], {}, [], 0
    for e in eps:
        pr = YC.parse(ins.get(e, ''))
        if pr is None:
            continue
        total += 1
        x = YC.select(ref.get(f'first/{e:04d}', []), *pr); y = YC.select(mine[e], *pr)
        reasons[y[2]] = reasons.get(y[2], 0) + 1
        if x == y:
            agree += 1
        elif x[2] == y[2] and x[2] == 'ok' and max(abs(u - v) for bx, by in zip(x[:2], y[:2]) for u, v in zip(bx, by)) <= 1.0:
            subpx.append(max(abs(u - v) for bx, by in zip(x[:2], y[:2]) for u, v in zip(bx, by)))
            px_moved += any(int(round(u)) != int(round(v)) for bx, by in zip(x[:2], y[:2]) for u, v in zip(bx, by))
        else:
            diff.append({'ep': e, 'instr': ins.get(e), 'ref': x, 'ours': y})
    out['2_decisions'] = {'episodes_parsed': total, 'identical': agree, 'same_cue_subpixel': len(subpx),
                          'subpixel_max': max(subpx, default=0), 'drawn_line_moves_1px': px_moved,
                          'different_decision': len(diff), 'reasons': reasons, 'differences': diff[:12], 'labels': len(ins)}
    if diff:
        fails.append(f'{len(diff)} cue decisions differ')

    # 3. drawing
    rng = np.random.default_rng(0); frame0 = load_rgb(img_dir / f'ep{eps[0]:03d}/0000.jpg'); same = 0
    for _ in range(200):
        x1, y1 = rng.uniform(0, 600), rng.uniform(0, 440)
        tb = [round(v, 1) for v in (x1, y1, x1 + rng.uniform(5, 120), y1 + rng.uniform(5, 120))]
        db = [round(v, 1) for v in (rng.uniform(0, 400), rng.uniform(0, 300), rng.uniform(400, 640), rng.uniform(300, 480))]
        g = bool(rng.integers(2))
        same += np.array_equal(YC.draw(frame0, tb, db, g), ref_draw(frame0, tb, db, g))
    out['3_drawing'] = {'cases': 200, 'pixel_identical': int(same)}
    if same < 200:
        fails.append('drawing differs from derive_box_dataset.draw')

    # 4. fake sessions
    log_dir = HERE / 'test_out'
    pick = next((e for e in eps if YC.parse(ins.get(e, '')) and YC.parse(ins[e])[0] == 1
                 and YC.select(mine[e], *YC.parse(ins[e]))[2] == 'ok'), None)
    frame = load_rgb(img_dir / f'ep{pick:03d}/0000.jpg'); raw = frame.copy(); task = ins[pick]
    cue, pol, mi = fake_session(frame, task, 'on', log_dir)
    s = {'episode': pick, 'task': task, 'wanted': cue.wanted()}
    cue.begin(); tb, db = cue.boxes; s['boxes'] = cue.boxes
    s['no_inference_passes_none'] = pol.preprocess(mi, run_inference=False) is None
    o1 = pol.preprocess(mi, run_inference=True)
    s['cue_drawn'] = bool(np.array_equal(o1[YC.CAM], ref_draw(raw, tb, db, False)))
    s['mailbox_frame_untouched'] = bool(np.array_equal(frame, raw))
    pol.postprocess(chunk([0.0] * 25, [0.0] * 12 + [0.9] * 13))
    s['after_open_no_grasp'] = cue.grasp is None
    o2 = pol.preprocess(mi, run_inference=True)
    s['white_still_there'] = bool(np.array_equal(o2[YC.CAM], ref_draw(raw, tb, db, False)))
    pol.postprocess(chunk([0.0] * 25, [0.9] * 10 + [0.1] * 15))
    s['grasp'] = cue.grasp
    o3 = pol.preprocess(mi, run_inference=True)
    s['white_gone_orange_kept'] = bool(np.array_equal(o3[YC.CAM], ref_draw(raw, tb, db, True)))
    cue.close()
    s['methods_restored'] = 'preprocess' not in pol.__dict__ and 'postprocess' not in pol.__dict__
    o4 = pol.preprocess(mi, run_inference=True)
    s['raw_after_close'] = bool(np.array_equal(o4[YC.CAM], raw))
    cue2, pol2, mi2 = fake_session(frame, task, 'shadow', log_dir)
    cue2.begin(); o5 = pol2.preprocess(mi2, run_inference=True); cue2.close()
    s['shadow_keeps_raw'] = bool(np.array_equal(o5[YC.CAM], raw)) and cue2.boxes is not None
    s['task3_no_cue'] = YC.parse('Using both arms, brush all tabletop paper balls into the dustpan, then pour them into the basket.') is None
    s['unknown_object_no_cue'] = YC.parse('Put the apple on the blue plate.') is None
    s['task2_parses'] = YC.parse('Store the pink makeup puff in the middle compartment.') == (2, 'powder_puff', 'organizer_middle')
    small = np.zeros((360, 640, 3), np.uint8)
    cue3, pol3, mi3 = fake_session(small, task, 'on', log_dir)
    cue3.begin(); o6 = pol3.preprocess(mi3, run_inference=True); cue3.close()
    s['wrong_size_raw'] = cue3.boxes is None and bool(np.array_equal(o6[YC.CAM], small)) and 'detector error' in str(cue3.reason)
    out['4_session'] = s
    for k, v in s.items():
        if v is False:
            fails.append(f'session check failed: {k}')

    # 5. latency (CPU unless config.json says otherwise)
    out['5_latency_ms'] = {'device': cfg['device'], 'median': round(float(np.median(ms)), 1),
                           'p95': round(float(np.percentile(ms, 95)), 1), 'max': round(float(np.max(ms)), 1)}
    out['fails'] = fails
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
