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
            'det_conf': 0.01, 'det_iou': 0.5, 'max_det': 300, 'conf': 0.5, 'margin': 0.15, 'retry_s': 3.0,
            # 11/15: cue_prompt -> while the cue is shown the VLA gets cue_prompts()[0] (only for checkpoints trained
            # on the *_named box datasets); regrasp -> white box back when the object was released off the destination
            'cue_prompt': False, 'regrasp': True, 'regrasp_tasks': [1], 'release_wait_s': 1.5, 'release_window_s': 5.0,
            'max_regrasp': 2}

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


# 11/15: box-referencing prompts. *_named box datasets use CUE_NAMED (the object's name stays, so a dropped object can
# still be found by name), *_gen use CUE_GENERIC (only the boxes say what and where, which forces the model to read
# them). Online, CUE_NAMED replaces the instruction while the cue is drawn (config cue_prompt).
CUE_NAMED = {1: 'Pick up the {o} in the white box and place it on the {d} plate in the orange box.',
             2: 'Put the {o} in the white box into the {d} compartment in the orange box.'}
CUE_GENERIC = {1: 'Pick up the object in the white box and place it on the plate in the orange box.',
               2: 'Put the item in the white box into the compartment in the orange box.'}
_PHRASE_FIX = {'watermellon': 'watermelon'}


def parse_phrases(text):
    """(task, object phrase as written, destination word) with the same patterns as parse(), or None."""
    s = ' '.join(str(text or '').split())
    for task, pats, table, dests, prefix in _PATS:
        for p in pats:
            m = p.fullmatch(s)
            if m and m.group(1).lower() in table and m.group(2).lower() in dests:
                o = m.group(1).lower()
                return task, _PHRASE_FIX.get(o, o), m.group(2).lower()
    return None


def cue_prompts(text):
    """(named prompt, generic prompt) for a task-1/2 instruction, or None."""
    p = parse_phrases(text)
    if p is None:
        return None
    task, o, d = p
    return CUE_NAMED[task].format(o=o, d=d), CUE_GENERIC[task]


def _inside(b, box, pad=0.15):
    """centre of b inside box grown by pad of its size on every side"""
    cx, cy = _ctr(b)
    w, h = box[2] - box[0], box[3] - box[1]
    return box[0] - pad * w <= cx <= box[2] + pad * w and box[1] - pad * h <= cy <= box[3] + pad * h


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
        self.prompts = cue_prompts(self.task)
        self.release_t, self.placed, self.n_regrasp = None, False, 0
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
            self.dets0, self.tb0 = dets, tb      # cue frame: objects that were there from the start
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
        if self.release_t is not None:
            self._check_release(img)
        grasped = self.grasp is not None
        if self.mode == 'on' or (grasped and 'grasp' not in self.saved):
            cued = draw(img, *self.boxes, grasped)
            if grasped:
                self._save('grasp', cued)
            if self.mode == 'on':
                out[CAM] = cued
                self.n_drawn += 1
                if self.cfg['cue_prompt'] and self.prompts is not None and 'task' in out:
                    out['task'] = self.prompts[0]

    def _check_release(self, img):
        """Regrasp insurance. After the grasping arm's gripper command opened again (release_t, from _on_chunk), look
        for the target once the release has happened: on the table off the destination -> the grasp missed or the object
        dropped: white box back on it and the grasp rule re-armed; on the destination -> placed, stop checking."""
        dt = time.monotonic() - self.release_t
        if dt < self.cfg['release_wait_s']:
            return
        if dt > self.cfg['release_wait_s'] + self.cfg['release_window_s']:
            self._log(event='release_unresolved', after_s=round(dt, 2))
            self.release_t = None
            return
        try:
            dets = detector(self.cfg)(img)
        except Exception:
            self.logger.exception('yolocue: release check detection failed')
            return
        self.n_detect += 1
        tcls, tb_old, db = self.parsed[1], self.boxes[0], self.boxes[1]
        # any instance of the target class already on a plate / in the organizer -> placed
        if any(self._on_destination(d[1:5], db, dets) for d in dets if d[0] == tcls and d[5] >= 0.25):
            self.placed, self.release_t = True, None
            self._log(event='placed', after_s=round(dt, 2))
            return
        # a regrasp target must be an object that moved: a target-class box sitting where another object already was
        # in the cue frame is a look-alike that never moved. On the official last frames all 5 false regrasps were such
        # look-alikes (red apple read as passion fruit, lotion bottle as lip gloss, contour as highlighter, ...).
        static = [d[1:5] for d in getattr(self, 'dets0', []) if d[5] >= 0.25 and _iou(d[1:5], self.tb0) < 0.5]
        t = [d[1:5] for d in sorted(dets, key=lambda d: -d[5]) if d[0] == tcls and d[5] >= self.cfg['conf']
             and not any(_iou(d[1:5], b) > 0.5 for b in static)]
        if not t:
            return                                     # occluded, out of view, or only look-alikes: next frame
        tb = [round(v, 1) for v in t[0]]
        self.n_regrasp += 1
        self.boxes, self.grasp, self.opened, self.release_t = (tb, db), None, [False, False], None
        self._log(event='regrasp', n=self.n_regrasp, after_s=round(dt, 2), target_box=tb, previous_box=tb_old)
        self._save(f'regrasp{self.n_regrasp}', draw(img, tb, db, False))

    def _on_destination(self, tb, db, dets):
        """Placed somewhere sensible = no regrasp. Official demos end with 3/172 task-1 and 13/126 task-2 objects
        (all placed correctly) outside the plain dest box: an object can rest on a plate rim beyond the plate box, and
        one seen through the acrylic organizer often spans the neighbouring level. So task 1: inside ANY detected plate
        (or the dest box); task 2: inside the union of all detected organizer levels and the dest box. Both grown 15%."""
        if self.parsed[0] == 1:
            plates = [r[1:5] for r in dets if r[0].startswith('plate_') and r[5] >= 0.25] + [db]
            return any(_inside(tb, b) for b in plates)
        org = [r[1:5] for r in dets if r[0].startswith('organizer_') and r[5] >= 0.25] + [db]
        u = [min(b[0] for b in org), min(b[1] for b in org), max(b[2] for b in org), max(b[3] for b in org)]
        return _inside(tb, u)

    def _on_chunk(self, chunk):
        self.n_chunks += 1
        acts = getattr(chunk, 'actions', None) or {}
        g = [np.asarray(acts[k], dtype=float).reshape(-1) for k in GRIPPER_KEYS]
        if self.grasp is not None and "arm" in self.grasp:   # 11/15 fix: after a DAgger handback grasp = {"resume": ...}, nothing to track
            # release = the grasping arm's command opens AFTER a closed step of the same chunk. Each chunk is in time
            # order from its request, but with RTC its first steps repeat the previous chunk's tail, which can still
            # hold the opening before the grasp; a closed -> open order inside one chunk is a real (planned) reopen.
            a = ('left', 'right').index(self.grasp['arm'])
            seq = g[a]
            shut = np.flatnonzero(seq < CMD_CLOSED)
            if (self.cfg['regrasp'] and self.parsed[0] in self.cfg['regrasp_tasks'] and self.boxes is not None
                    and not self.placed and self.release_t is None and self.n_regrasp < self.cfg['max_regrasp']
                    and len(shut) and np.any(seq[shut[0]:] > CMD_OPEN)):
                self.release_t = time.monotonic()
                self._log(event='release', chunk=self.n_chunks, arm=self.grasp['arm'])
            return
        for j in range(min(len(g[0]), len(g[1]))):
            for a in (0, 1):
                v = g[a][j]
                if not self.opened[a] and v > CMD_OPEN:
                    self.opened[a] = True
                elif self.opened[a] and v < CMD_CLOSED:
                    self.grasp = {'chunk': self.n_chunks, 'step': j, 'arm': ('left', 'right')[a]}
                    self._log(event='grasp', **self.grasp)
                    return

    def on_resume(self):
        """DAgger handback (the server calls this when a paused session resumes). During the pause a human may have
        moved, grasped or placed the target, so the cue state is rebuilt from the current frame with the start rules:
        target on a plate / in the organizer -> placed, no white box; target visible elsewhere -> fresh cue and the grasp
        rule re-armed; target not visible (most likely in a gripper) -> treated as grasped, orange box only."""
        self.release_t, self.opened = None, [False, False]
        rt = self.server._robot_runtime
        mi = rt.mailbox.get() if rt is not None else None
        if mi is None or self.parsed is None:
            self._log(event='resume', decision='no frame'); return
        rgb = np.asarray(mi.features[CAM])
        try:
            dets = detector(self.cfg)(rgb)
        except Exception:
            self.logger.exception('yolocue: resume detection failed; the cue keeps its old state'); return
        self.n_detect += 1
        tb, db_new, why = select(dets, *self.parsed, conf=self.cfg['conf'], margin=self.cfg['margin'], rgb=rgb)
        db = self.boxes[1] if self.boxes is not None else db_new
        tcls = self.parsed[1]
        if db is not None and any(self._on_destination(d[1:5], db, dets) for d in dets if d[0] == tcls and d[5] >= 0.25):
            self.placed = True
            if self.boxes is not None:
                self.grasp = {'resume': 'placed'}
            decision = 'placed'
        elif why == 'ok':
            self.boxes, self.grasp, self.placed, self.reason = (tb, db_new), None, False, 'ok'
            self.dets0, self.tb0 = dets, tb
            self._save('resume_cue', draw(rgb, tb, db_new, False))
            decision = 'fresh cue'
        elif self.boxes is not None:
            self.grasp = {'resume': 'target not visible'}
            decision = 'grasped (target not visible)'
        else:
            decision = f'no cue ({why})'
        self._log(event='resume', decision=decision, target_box=tb, dest_box=db)

    def close(self, why='stop'):
        if self.closed:
            return
        self.closed = True
        uninstall(self.policy)
        self._log(event='end', why=why, reason=self.reason, cue=self.boxes, grasp=self.grasp, frames=self.n_obs,
                  drawn=self.n_drawn, chunks=self.n_chunks, detections=self.n_detect, regrasps=self.n_regrasp,
                  placed=self.placed, cue_prompt=bool(self.cfg['cue_prompt'] and self.prompts))
