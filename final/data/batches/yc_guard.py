"""Cross-class guard for the yolocue target choice (offline prototype, not installed):
after YC.select picks the target box, call it ambiguous when a box of another non-container class overlaps it (IoU > iou)
with confidence >= target_conf - margin. Motivated by the live 10:37 cue on a carrot for "red chili" (red_chili 0.83 / carrot 0.61)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path.home() / 'lingbot_v7work'))
import yolocue_bridge as YC
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
def iou(a, b):
    i = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])); return i / (area(a) + area(b) - i + 1e-9)
def select_guarded(dets, task, tcls, dcls, rgb=None, margin=0.25, ov=0.5):
    tb, db, why = YC.select(dets, task, tcls, dcls, rgb=rgb)
    if tb is None: return tb, db, why
    tconf = max((d[5] for d in dets if d[0] == tcls and iou(d[1:5], tb) > 0.9), default=0)
    rivals = [d for d in dets if d[0] != tcls and not d[0].startswith(('plate_', 'organizer_')) and iou(d[1:5], tb) > ov and d[5] >= tconf - margin]
    if rivals: return None, None, f'target ambiguous: {max(rivals, key=lambda d: d[5])[0]} {max(rivals, key=lambda d: d[5])[5]:.2f} on the {tcls} {tconf:.2f} box'
    return tb, db, why
