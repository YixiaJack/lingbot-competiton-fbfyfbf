"""Example figures for the README (figures/): YOLO raw output vs the two boxes LingBot gets (val renders of tasks 1-3),
and box vs boxgrasp on demonstration 0000 at the frames LingBot is queried (every 50 steps). CPU only.
Usage (on the cluster, model env): REAL_ASSETS=$R/assets python make_figures.py <code dir> <output dir>
"""
import sys, json, glob, os
R = os.path.expandvars('$SCRATCH/lingbot/final_real_20261112')
CODE, OUT = sys.argv[1], sys.argv[2]
sys.path[:0] = [CODE, os.path.expandvars('$SCRATCH/lingbot/final_yolo_20261111/deps')]
os.environ['YOLO_CONFIG_DIR'] = f'{R}/yolo_settings'
import numpy as np, cv2
from PIL import Image
from ultralytics import YOLO
from cues_real import parse_instruction, select_boxes, run_detector, apply_cue, grasp_flags
D = f'{R}/det'; K = 2   # upscale for viewing (nearest: pixels stay as the model sees them)
model = YOLO(f'{D}/runs/yolo11s/weights/best.pt')
PAL = [(0, 200, 0), (0, 160, 255), (220, 0, 220), (0, 220, 220), (255, 60, 60), (120, 80, 255), (180, 180, 0), (0, 120, 60), (200, 0, 90), (90, 160, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def up(img): return cv2.resize(img, (img.shape[1] * K, img.shape[0] * K), interpolation=cv2.INTER_NEAREST)

def header(panel, text, h=34):
    bar = np.full((h, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, h - 11), FONT, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])

def row(panels, caption_lines):
    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    body = panels[0]
    for p in panels[1:]: body = np.hstack([body, gap, p])
    cap = np.full((26 * len(caption_lines) + 10, body.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(caption_lines):
        cv2.putText(cap, t, (8, 26 * (i + 1)), FONT, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([body, cap])

def draw_dets(img, dets):
    out = up(img).copy()
    for n, (k, c, b) in enumerate(sorted(dets, key=lambda d: -d[1])):
        col = PAL[n % len(PAL)]; x1, y1, x2, y2 = [int(round(v * K)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), col, 2)
        lab = f'{k} {c:.2f}'; (tw, th), _ = cv2.getTextSize(lab, FONT, 0.42, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), col, -1)
        cv2.putText(out, lab, (x1 + 2, ty), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def pick(task, cond):
    for mp in sorted(glob.glob(f'{D}/meta/val/*.json')):
        it, j = int(os.path.basename(mp)[:4]), int(os.path.basename(mp)[5:7])
        if it >= 110 or j % 2: continue          # even j: fixed lighting
        m = json.load(open(mp))
        if m['spec']['task'] == task and not any(b['truncated'] for b in m['boxes']) and cond(m):
            return os.path.basename(mp)[:-5], m

picks = {1: pick(1, lambda m: m['spec']['n_objects'] >= 5 and m['spec']['n_plates'] >= 3),
         2: pick(2, lambda m: m['spec']['n_objects'] >= 5), 3: pick(3, lambda m: True)}
info = {}
for task, (stem, m) in picks.items():
    img = np.asarray(Image.open(f'{D}/images/val/{stem}.png').convert('RGB'))
    dets = run_detector(model, img, model.names)
    parsed = parse_instruction(m['instruction'])
    tb, db, reason = select_boxes(dets, parsed)
    vla, _ = apply_cue(img, m['instruction'], tb, db, 'box')
    n_vla = 2 if tb is not None else 0
    p1 = header(up(img), 'frame 0, head camera (320x240)')
    p2 = header(draw_dets(img, dets), f'YOLO output: {len(dets)} boxes (conf >= 0.05)')
    p3 = header(up(vla), f'LingBot input: {n_vla} boxes' + ('' if n_vla else ' (plain image)'))
    sel = (f'selected: target = {parsed[1]} (white), destination = {parsed[2]} (orange)' if n_vla
           else f'selected: none, "{reason}" -> plain instruction, plain image')
    fig = row([p1, p2, p3], [f'task {task} | instruction: "{m["instruction"]}"', sel])
    f = f'{OUT}/yolo_to_lingbot_task{task}.png'
    cv2.imwrite(f, cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
    info[task] = {'image': stem, 'instruction': m['instruction'], 'gt': [b['cls'] for b in m['boxes']],
                  'dets': [(k, round(c, 2)) for k, c, b in sorted(dets, key=lambda d: -d[1])], 'reason': reason}

# box vs boxgrasp on one demonstration, at the frames where LingBot is queried (every 50 steps)
ep = f'{R}/episodes/0000'
with np.load(f'{ep}/episode.npz') as z: cam, st = z['cam_high'], z['state']
meta = json.load(open(f'{ep}/complete.json')); g = grasp_flags(st)
frames = [0, 50, 100, 150]; first = g.index(True)
rows = []
for mode in ('box', 'boxgrasp'):
    ps = []
    for t in frames:
        im, _ = apply_cue(cam[t], meta['instruction'], meta['target_box'], meta['dest_box'], mode, grasped=g[t])
        state_txt = '' if not g[t] else (' (gripper closed)' if min(st[t, 6], st[t, 13]) < 0.5 else ' (released)')
        ps.append(header(up(im), f'{mode}, frame {t}{state_txt}'))
    rows.append(row(ps, [f'{mode}: ' + ('frame-0 boxes on every frame (s22_box)' if mode == 'box'
                         else f'target box dropped from frame {first}, when the gripper closes; destination box kept (s22_boxgrasp)')]))
fig = np.vstack([rows[0], np.full((10, rows[0].shape[1], 3), 255, np.uint8), rows[1]])
cv2.imwrite(f'{OUT}/box_vs_boxgrasp_episode0000.png', cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
info['episode'] = {'instruction': meta['instruction'], 'T': len(st), 'grasp_frame': first,
                   'gripper_at_frames': {t: round(float(min(st[t, 6], st[t, 13])), 2) for t in frames}}
print(json.dumps(info, ensure_ascii=False))
