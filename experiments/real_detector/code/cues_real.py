"""Target cues for tasks 1 and 2: rule-based instruction parsing (no language model), cue rendering, detector selection.

Same cue format as the proxy experiment: on the head camera, a white box around the target and an orange box around the
destination, drawn from frame zero and kept for the whole episode; or the two boxes as 0-1000 xyxy text after the
instruction. Missing either box means no cue at all (plain image and plain instruction).
boxgrasp (11/14): like box, but the white target box is dropped from the first frame on which a gripper has closed
(GraspLatch) and never comes back; the orange destination box stays, plates and the organizer never move. Once the
object is lifted the frame-zero target box frames empty table, and after a drop it points where the object no longer is.
"""
import re

from catalog import NAME, PLATE_COLORS, LEVELS, TEMPLATES

# Gripper entries of the 14-d state / action (6 joints + gripper per arm). "Closed" is the field page's definition
# (field_build.gripper_events: more than half the gripper's range away from where it started), made causal with a fixed
# range so the policy side can decide it online. RoboTwin's gripper runs 1 (open) to 0 (closed); a real robot needs its
# own GRIP_RANGE. 200 task-1 demos (11/14): the acting gripper crosses at frame 54-84, the other one never moves.
GRIPS, GRIP_RANGE = (6, 13), 1.0


class GraspLatch:
    """on = some gripper has closed at or before this state. Latched: it never turns back off."""

    def __init__(self, state0):
        self.g0 = [float(state0[i]) for i in GRIPS]
        self.on = False

    def update(self, state):
        self.on = self.on or any(abs(float(state[i]) - g) > 0.5 * GRIP_RANGE for i, g in zip(GRIPS, self.g0))
        return self.on


def grasp_flags(states):
    """GraspLatch over a recorded episode: one flag per frame (training-data conversion)."""
    latch = GraspLatch(states[0])
    return [latch.update(s) for s in states]

_OBJ = {v.lower(): k for k, v in NAME.items() if not k.startswith('plate_')}
_PATTERNS = []
for _task, _tpls in TEMPLATES.items():
    for _t in _tpls:
        _p = re.escape(_t).replace(r'\{o\}', '(?P<o>.+?)').replace(r'\{d\}', '(?P<d>[a-z]+)')
        _PATTERNS.append((_task, re.compile(_p, re.I)))


def parse_instruction(text):
    """-> (task, target class, destination class) or None. Destination: plate_<colour> or organizer_<level>."""
    for task, pat in _PATTERNS:
        m = pat.fullmatch(text.strip())
        if not m:
            continue
        o, d = m['o'].lower(), m['d'].lower()
        if o not in _OBJ:
            continue
        if task == 1 and d in PLATE_COLORS:
            return 1, _OBJ[o], f'plate_{d}'
        if task == 2 and d in LEVELS:
            return 2, _OBJ[o], f'organizer_{d}'
    return None


def apply_cue(image, instruction, target_box, dest_box, mode, grasped=False):
    """grasped: GraspLatch state of this frame; only the boxgrasp mode reads it."""
    if mode == 'text' or target_box is None or dest_box is None:
        return image, instruction
    if mode in ('box', 'boxgrasp'):
        import cv2
        out = image.copy()
        drawn = [(target_box, (255, 255, 255)), (dest_box, (255, 128, 0))]
        if mode == 'boxgrasp' and grasped:
            drawn = drawn[1:]
        for box, color in drawn:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        return out, instruction
    if mode == 'coords':
        h, w = image.shape[:2]
        norm = lambda b: ','.join(str(round(v / (w if i % 2 == 0 else h) * 1000)) for i, v in enumerate(b))
        return image, f'{instruction} Target box [0-1000]: ({norm(target_box)}). Destination box: ({norm(dest_box)}).'
    raise ValueError(mode)


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def select_boxes(dets, parsed, conf_target=0.5, conf_dest=0.5, margin=0.15):
    """dets: [(class name, confidence, xyxy)] from the detector on frame zero.
    Target: best box of the target class; needs conf >= conf_target and a lead of `margin` over a second box of the
    same class (each object appears once, two confident boxes mean the detector is confused).
    Destination: best box of the destination class with conf >= conf_dest; two plates can share a colour (either is
    correct), so pick the one nearest the target. Returns (target_box, dest_box, reason)."""
    if parsed is None:
        return None, None, 'instruction not parsed'
    _, tcls, dcls = parsed
    t = sorted([(c, b) for k, c, b in dets if k == tcls], key=lambda x: -x[0])
    if not t or t[0][0] < conf_target:
        return None, None, 'target not detected'
    if len(t) > 1 and t[0][0] - t[1][0] < margin:
        return None, None, 'target ambiguous'
    tb = t[0][1]
    d = [(c, b) for k, c, b in dets if k == dcls and c >= conf_dest]
    if not d:
        return None, None, 'destination not detected'
    if dcls.startswith('plate_'):
        tc = _center(tb)
        db = min(d, key=lambda x: (_center(x[1])[0] - tc[0]) ** 2 + (_center(x[1])[1] - tc[1]) ** 2)[1]
    else:
        db = max(d, key=lambda x: x[0])[1]
    return list(tb), list(db), 'ok'


def run_detector(detector, image_rgb, names):
    """CPU, frame zero only. RoboTwin gives RGB; the ultralytics numpy API expects BGR."""
    r = detector.predict(image_rgb[..., ::-1], device='cpu', conf=0.05, verbose=False)[0]
    return [(names[int(b.cls.item())], float(b.conf.item()), b.xyxy[0].cpu().tolist()) for b in r.boxes]
