"""yolocue inside lingbot_infer (5080 inference machine): the YOLO box cue the VLA was trained with, drawn online on the
head-camera image the model gets in task-1/2 sessions.

Training (experiments/autolabel: t1_boxes.py, t2_boxes.py, own_boxes_v5.py, derive_box_dataset.py) drew the "boxgrasp"
cue on camera_top of half of the task-1/2 demos (the other half of the mix stays plain):
  - YOLO on frame 0 (ultralytics predict: conf 0.01, NMS IoU 0.5 = the 8.4 default, input 480x640 unpadded), boxes
    rounded to 0.1 px and conf to 0.001 as in dets_first.json;
  - the instruction gives target class + destination; target = best box of its class with conf >= 0.5 and 0.15 ahead
    of a second box of that class; destination = task 1 the plate of the named colour (conf >= 0.5) nearest the
    target, task 2 the most confident box (conf >= 0.5) of the named organizer level; anything missing -> no cue;
    (added 11/15, not in the v3/v5 training boxes) task 1: target box IoU > 0.5 with the destination -> no cue;
  - white box (255,255,255) = target until a gripper COMMAND closes (above 0.5 once, then below 0.2), orange box
    (255,128,0) = destination for the whole episode; PIL rectangle [x1, y1, x2-1, y2-1], width 5, at 640x480.
Online:
  - session start (robot at home, like frame 0): YOLO on the newest head-camera frame; without a cue the next
    inferences retry until retry_s after the start or the first grasp;
  - policy.preprocess is wrapped: the cue is drawn on the copy of camera_top that goes to the model (robotharness and
    the takeover trace keep reading the raw frames from the mailbox);
  - policy.postprocess is wrapped: every chunk's gripper commands run the grasp rule; the scheduler is synchronous (a
    chunk is fully executed before the next frame is taken), so a close inside chunk n drops the white box from the
    frame of chunk n+1 on.
No cue, i.e. the raw image the VLA also trained on, when the instruction is not a task-1/2 template, when a box is
missing or ambiguous, for frames that are not 480x640 uint8, or on any error.
Layout on the 5080 (~/yolocue): mode (off | shadow: detect and log, image unchanged | on; missing = off), config.json,
weights/<name>.torchscript + <name>.names.json (export_ts.py). Log: <runtime.log_dir>/yolocue/<time>_<session>.jsonl
plus <time>_<session>_cue.jpg (first frame with the cue) / _grasp.jpg (first frame after the grasp) / _nocue.jpg.
"""
import json
import re
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CAM = 'observation.images.camera_top'
H, W = 480, 640
WHITE, ORANGE, LINE = (255, 255, 255), (255, 128, 0), 5
CMD_OPEN, CMD_CLOSED = 0.5, 0.2
GRIPPER_KEYS = ('action.left_gripper.position', 'action.right_gripper.position')
DEFAULTS = {'weights': 'weights/v5.torchscript', 'names': 'weights/v5.names.json', 'device': 'cpu',
            'det_conf': 0.01, 'det_iou': 0.5, 'max_det': 300, 'conf': 0.5, 'margin': 0.15, 'retry_s': 3.0}

# instruction -> (task, target class, destination class). Templates: field_prompt_policy.PATTERNS (official 647);
# phrases: autolabel/build_yolo.T1_PHRASE / T2_PHRASE plus private_instructions aliases.
T1_TEMPLATES = [r'Put the (.+) on the (\w+) plate\.', r'Pick up the (.+) and place it on the (\w+) plate\.',
                r'Move the (.+) onto the (\w+) plate\.', r'Place the (.+) on the (\w+) plate\.',
                r'Could you put the (.+) on the (\w+) plate\?']
T2_TEMPLATES = [r'Put the (.+) in the (\w+) compartment\.', r'Place the (.+) inside the (\w+) level of the organizer\.',
                r'Store the (.+) in the (\w+) compartment\.', r'Take the (.+) and put it in the (\w+) section\.',
                r'Organize the (.+) into the (\w+) compartment\.']
T1_LOOSE = r'(?:pick up|put|place|move|could you put) the (.+?) (?:and place it )?(?:on|onto) the ([a-z]+) plate[.?]?'
T2_LOOSE = (r'(?:put|place|store|organize|take) the (.+?) (?:and put it )?(?:in|into|inside) the (top|middle|bottom) '
            r'(?:compartment|section|level)(?: of the organizer)?[.?]?')
T1_OBJ = ['avocado', 'banana', 'bell_pepper', 'carrot', 'cucumber', 'dragon_fruit', 'eggplant', 'green_apple', 'kiwi',
          'lemon', 'mango', 'mangosteen', 'mushroom', 'onion', 'orange', 'papaya', 'passion_fruit', 'peach', 'pear',
          'persimmon', 'pineapple', 'pomegranate', 'potato', 'pumpkin', 'red_apple', 'strawberry', 'tomato',
          'watermelon', 'wax_apple', 'grapes', 'cherry', 'tangerine', 'red_chili', 'red_delicious', 'starfruit', 'green_chili']
T1_PHRASE = {c.replace('_', ' '): c for c in T1_OBJ}
T1_PHRASE.update({'bunch of grapes': 'grapes', 'cherries': 'cherry', 'mandarin': 'tangerine', 'red chili pepper': 'red_chili',
                  'red delicious apple': 'red_delicious', 'kiwi fruit': 'kiwi', 'green chili pepper': 'green_chili',
                  'watermellon': 'watermelon'})
T2_PHRASE = {'beige lip gloss tube': 'lip_gloss', 'beige lotion pump bottle': 'lotion_bottle',
             'black lipstick with a clear cap': 'lipstick', 'dark-colored round contour compact': 'contour',
             'large black and gold makeup brush': 'brush_large', 'light-colored round highlighter compact': 'highlighter',
             'pink makeup puff': 'powder_puff', 'rectangular multicolor eyeshadow palette': 'eyeshadow_palette',
             'small black and gold makeup brush': 'brush_small', 'black round glitter compact': 'glitter_compact',
             'dark-colored round highlighter compact': 'contour'}
COLORS, LEVELS = ('white', 'yellow', 'pink', 'purple', 'blue'), ('top', 'middle', 'bottom')
_PATS = [(1, [re.compile(t, re.I) for t in T1_TEMPLATES + [T1_LOOSE]], T1_PHRASE, COLORS, 'plate_'),
         (2, [re.compile(t, re.I) for t in T2_TEMPLATES + [T2_LOOSE]], T2_PHRASE, LEVELS, 'organizer_')]


def parse(text):
    """(task, target class, destination class), or None (task 3, unknown object or phrasing)."""
    s = ' '.join(str(text or '').split())
    for task, pats, table, dests, prefix in _PATS:
        for p in pats:
            m = p.fullmatch(s)
            if m and m.group(1).lower() in table and m.group(2).lower() in dests:
                return task, table[m.group(1).lower()], prefix + m.group(2).lower()
    return None


def _ctr(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def select(dets, task, tcls, dcls, conf=0.5, margin=0.15, rgb=None):
    """t1_boxes.select (task 1) / t2_boxes (task 2) on rows [cls, x1, y1, x2, y2, conf]."""
    t = sorted([(d[5], d[1:5]) for d in dets if d[0] == tcls], key=lambda x: -x[0])
    if not t or t[0][0] < conf:
        return None, None, 'target not detected'
    if len(t) > 1 and t[0][0] - t[1][0] < margin:
        return None, None, 'target ambiguous'
    tb = t[0][1]
    d = [(r[5], r[1:5]) for r in dets if r[0] == dcls and r[5] >= conf]
    if not d and task == 1 and rgb is not None:
        d = _plate_colour_fix(rgb, dets, dcls, conf)
    if not d:
        return None, None, 'destination not detected' if task == 1 else 'level not detected'
    if task == 1:
        tc = _ctr(tb)
        db = min(d, key=lambda x: (_ctr(x[1])[0] - tc[0]) ** 2 + (_ctr(x[1])[1] - tc[1]) ** 2)[1]
        if _iou(tb, db) > 0.5:   # 11/15: the "target" is the destination plate itself (T1_train158: cherry box = blue plate)
            return None, None, 'target overlaps destination'
    else:
        db = max(d, key=lambda z: z[0])[1]
    return [round(v, 1) for v in tb], [round(v, 1) for v in db], 'ok'


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


# median (a*, b*) of the inner part of each plate colour on the station (OpenCV 8-bit Lab), 86 plates of the 71
# corrected team episodes, sd 1-2 per channel (11/15). Used only to rescue a destination YOLO read as another colour.
STATION_PLATE_AB = {'plate_blue': (130.8, 121.3), 'plate_pink': (165.3, 145.2), 'plate_purple': (145.2, 130.2),
                    'plate_white': (138.4, 137.2), 'plate_yellow': (147.2, 166.8)}


def plate_ab(rgb, b):
    import cv2
    x0, y0, x1, y1 = (int(v) for v in b)
    crop = np.ascontiguousarray(np.asarray(rgb)[max(y0, 0):y1, max(x0, 0):x1])
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (max(int(w * .32), 1), max(int(h * .32), 1)), 0, 0, 360, 1, -1)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)[m > 0]
    return np.median(lab[:, 1:], 0)


def _plate_colour_fix(rgb, dets, dcls, conf, max_dist=10.0, yolo_wrong=12.0):
    """The named plate colour is missing from YOLO's output: assign colours to the confident plate boxes by their
    measured (a*, b*) (one plate per colour, minimum total distance; no scene has two plates of one colour, 0/97).
    The plate that gets the named colour is used when it is close to that colour (<= max_dist) and YOLO's own colour
    for it is clearly off (>= yolo_wrong away), e.g. 11/15 01:13 the purple plate read as a second pink plate."""
    import itertools
    plates = [r for r in dets if r[0].startswith('plate_') and r[5] >= conf]
    ab = [plate_ab(rgb, r[1:5]) for r in plates]
    keep = [(r, a) for r, a in zip(plates, ab) if a is not None]
    if not keep or len(keep) > 5:
        return []
    cols = list(STATION_PLATE_AB)
    dist = lambda a, c: float(np.hypot(a[0] - STATION_PLATE_AB[c][0], a[1] - STATION_PLATE_AB[c][1]))
    best = min(itertools.permutations(cols, len(keep)), key=lambda perm: sum(dist(a, c) for (r, a), c in zip(keep, perm)))
    for (r, a), c in zip(keep, best):
        if c == dcls and dist(a, c) <= max_dist and dist(a, r[0]) >= yolo_wrong:
            return [(r[5], r[1:5])]
    return []


def draw(rgb, tb, db, grasped):
    """derive_box_dataset.draw, unchanged (returns a writable copy)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for box, col, skip in ((tb, WHITE, grasped), (db, ORANGE, False)):
        if skip:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        d.rectangle([x1, y1, x2 - 1, y2 - 1], outline=col, width=LINE)
    return np.array(im)


class Detector:
    """TorchScript export of the ultralytics model + ultralytics 8.4 predict pre/post-processing for 480x640 frames."""

    def __init__(self, ts_path, names, device='cpu', conf=0.01, iou=0.5, max_det=300):
        import torch
        self.torch, self.names, self.conf, self.iou, self.max_det = torch, list(names), conf, iou, max_det
        self.device = torch.device(device)
        self.model = torch.jit.load(str(ts_path), map_location=self.device).eval()
        with torch.inference_mode():   # TorchScript optimises on the first calls
            for _ in range(2):
                self.model(torch.zeros(1, 3, H, W, device=self.device))

    def __call__(self, rgb):
        import torchvision
        torch = self.torch
        rgb = np.asarray(rgb)
        if rgb.shape != (H, W, 3) or rgb.dtype != np.uint8:
            raise ValueError(f'camera_top is {rgb.shape} {rgb.dtype}; the cue was trained on {H}x{W} uint8 frames')
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(self.device).float()
        x /= 255
        with torch.inference_mode():
            p = self.model(x)
            p = (p[0] if isinstance(p, (list, tuple)) else p).float()[0]          # (4 + nc, anchors)
            nc = p.shape[0] - 4
            p = p.T[p[4:4 + nc].amax(0) > self.conf]
            xy, wh = p[:, :2], p[:, 2:4] / 2
            box = torch.cat((xy - wh, xy + wh), 1)
            score, cls = p[:, 4:4 + nc].max(1, keepdim=True)
            keep = score.view(-1) > self.conf
            box, score, cls = box[keep], score[keep].view(-1), cls[keep].view(-1)
            if box.shape[0] > 30000:
                o = score.argsort(descending=True)[:30000]
                box, score, cls = box[o], score[o], cls[o]
            i = torchvision.ops.nms(box + cls[:, None].float() * 7680, score, self.iou)[:self.max_det]
            box, score, cls = box[i], score[i], cls[i]
            allc = p[:, 4:4 + nc][keep][i] if box.shape[0] <= 30000 else None
            box[:, 0::2] = box[:, 0::2].clamp(0, W)
            box[:, 1::2] = box[:, 1::2].clamp(0, H)
        out = [[self.names[int(c)], *[round(v, 1) for v in b], round(s, 3)] for b, c, s in zip(box.tolist(), cls.tolist(), score.tolist())]
        # per-colour scores of the plate boxes, for the one-plate-per-colour fix in select()
        self.plate_scores = {}
        if allc is not None:
            pid = {n: k for k, n in enumerate(self.names) if n.startswith('plate_')}
            for r, sc in zip(out, allc.tolist()):
                if r[0].startswith('plate_'):
                    self.plate_scores[tuple(r[1:5])] = {n: round(sc[k], 4) for n, k in pid.items()}
        return out


_LOCK = threading.Lock()
_DET = {}


def load_config():
    cfg = dict(DEFAULTS)
    f = ROOT / 'config.json'
    if f.exists():
        cfg.update(json.loads(f.read_text()))
    return cfg


def read_mode():
    f = ROOT / 'mode'
    m = f.read_text().strip() if f.exists() else 'off'
    return m if m in ('off', 'shadow', 'on') else 'off'


def detector(cfg):
    key = (cfg['weights'], cfg['device'], cfg['det_conf'], cfg['det_iou'], cfg['max_det'])
    with _LOCK:
        if key not in _DET:
            names = json.loads((ROOT / cfg['names']).read_text())['names']
            _DET.clear()
            _DET[key] = Detector(ROOT / cfg['weights'], names, cfg['device'], cfg['det_conf'], cfg['det_iou'], cfg['max_det'])
        return _DET[key]


def preload(logger=None):
    """Load the detector in the background when the service starts (skipped while the mode is off)."""
    def run():
        try:
            if read_mode() != 'off':
                t = time.perf_counter()
                detector(load_config())
                if logger is not None:
                    logger.info('yolocue: detector loaded in %.1f s', time.perf_counter() - t)
        except Exception:
            if logger is not None:
                logger.exception('yolocue: detector preload failed')
    threading.Thread(target=run, name='yolocue-preload', daemon=True).start()


def uninstall(policy):
    for k in ('preprocess', 'postprocess'):
        policy.__dict__.pop(k, None)


class YoloCue:
    def __init__(self, server, session, policy, mode, logger, log_dir):
        self.server, self.session, self.policy, self.mode, self.logger = server, session, policy, mode, logger
        self.cfg = load_config()
        self.task = getattr(policy, '_task', None) or getattr(session, 'task', None) or ''
        self.sid = str(getattr(session, 'session_id', ''))
        self.parsed = parse(self.task)
        self.dir = Path(log_dir) / 'yolocue'
        self.base = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.sid[:8]}"
        self.t0 = time.monotonic()
        self.boxes, self.reason, self.grasp, self.opened = None, None, None, [False, False]
        self.n_obs = self.n_drawn = self.n_chunks = self.n_detect = 0
        self.saved, self.closed = set(), False

    @property
    def state(self):
        return 'cue' if self.boxes is not None else f'no cue yet ({self.reason})'

    def wanted(self):
        return self.parsed is not None

    def _log(self, **rec):
        rec = {'wall_time': round(time.time(), 3), 't': round(time.monotonic() - self.t0, 3), **rec}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / f'{self.base}.jsonl', 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except Exception:
            self.logger.exception('yolocue: log write failed')

    def _save(self, tag, rgb, dets=None):
        if tag in self.saved:
            return
        self.saved.add(tag)
        try:
            from PIL import Image, ImageDraw
            im = Image.fromarray(np.asarray(rgb))
            if dets:
                d = ImageDraw.Draw(im)
                for r in dets:
                    d.rectangle(r[1:5], outline=(255, 0, 0), width=1)
                    d.text((r[1] + 2, r[2] + 1), f'{r[0]} {r[5]:.2f}', fill=(255, 0, 0))
            self.dir.mkdir(parents=True, exist_ok=True)
            im.save(self.dir / f'{self.base}_{tag}.jpg', quality=90)
        except Exception:
            self.logger.exception('yolocue: image save failed')

    def begin(self):
        """Blocking, before the control loop starts: the first chunk already sees the cue."""
        self._log(event='start', mode=self.mode, task=self.task, session_id=self.sid, parsed=self.parsed,
                  weights=self.cfg['weights'], device=self.cfg['device'])
        rt = self.server._robot_runtime
        mi = rt.mailbox.get() if rt is not None else None
        if mi is not None:
            self._detect(np.asarray(mi.features[CAM]), 'start')
        else:
            self.reason = 'no frame at start'
            self._log(event='detect', why='start', reason=self.reason)
        self._install()

    def _detect(self, rgb, why):
        t = time.perf_counter()
        try:
            dets = detector(self.cfg)(rgb)
        except Exception as e:
            self.reason = f'detector error: {e!r}'
            self._log(event='detect', why=why, reason=self.reason)
            self.logger.exception('yolocue: detection failed')
            return
        self.n_detect += 1
        tb, db, self.reason = select(dets, *self.parsed, conf=self.cfg['conf'], margin=self.cfg['margin'], rgb=rgb)
        shown = [r for r in dets if r[5] >= 0.25 or (r[0] in self.parsed[1:] and r[5] >= 0.05)]
        self._log(event='detect', why=why, ms=round((time.perf_counter() - t) * 1000, 1), reason=self.reason,
                  target_box=tb, dest_box=db, dets=shown[:40])
        if self.reason == 'ok':
            self.boxes = (tb, db)
            self._save('cue', draw(rgb, tb, db, False))
        else:
            self._save('nocue', rgb, shown)
            self._save('nocue_raw', rgb)

    def _install(self):
        uninstall(self.policy)
        orig_pre, orig_post = self.policy.preprocess, self.policy.postprocess

        def preprocess(model_input, *args, **kwargs):
            out = orig_pre(model_input, *args, **kwargs)
            if out is not None and not self.closed:
                try:
                    self._on_obs(out)
                except Exception:
                    self.logger.exception('yolocue: drawing failed; this frame goes raw')
            return out

        def postprocess(model_output, *args, **kwargs):
            chunk = orig_post(model_output, *args, **kwargs)
            if not self.closed:
                try:
                    self._on_chunk(chunk)
                except Exception:
                    self.logger.exception('yolocue: grasp tracking failed')
            return chunk

        self.policy.preprocess, self.policy.postprocess = preprocess, postprocess

    def _on_obs(self, out):
        self.n_obs += 1
        img = out.get(CAM)
        if not isinstance(img, np.ndarray):
            return
        if self.boxes is None and self.grasp is None and time.monotonic() - self.t0 <= self.cfg['retry_s']:
            self._detect(img, 'retry')
        if self.boxes is None:
            return
        grasped = self.grasp is not None
        if self.mode == 'on' or (grasped and 'grasp' not in self.saved):
            cued = draw(img, *self.boxes, grasped)
            if grasped:
                self._save('grasp', cued)
            if self.mode == 'on':
                out[CAM] = cued
                self.n_drawn += 1

    def _on_chunk(self, chunk):
        self.n_chunks += 1
        if self.grasp is not None:
            return
        acts = getattr(chunk, 'actions', None) or {}
        g = [np.asarray(acts[k], dtype=float).reshape(-1) for k in GRIPPER_KEYS]
        for j in range(min(len(g[0]), len(g[1]))):
            for a in (0, 1):
                v = g[a][j]
                if not self.opened[a] and v > CMD_OPEN:
                    self.opened[a] = True
                elif self.opened[a] and v < CMD_CLOSED:
                    self.grasp = {'chunk': self.n_chunks, 'step': j, 'arm': ('left', 'right')[a]}
                    self._log(event='grasp', **self.grasp)
                    return

    def close(self, why='stop'):
        if self.closed:
            return
        self.closed = True
        uninstall(self.policy)
        self._log(event='end', why=why, reason=self.reason, cue=self.boxes, grasp=self.grasp, frames=self.n_obs,
                  drawn=self.n_drawn, chunks=self.n_chunks, detections=self.n_detect)
