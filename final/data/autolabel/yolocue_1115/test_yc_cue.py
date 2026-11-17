"""Offline test of the 11/15 yolocue additions (cue prompts + regrasp insurance) with a scripted detector and a fake
session; no model, no robot, no GPU. usage: python -I test_yc_cue.py   (next to yolocue_bridge.py)"""
import logging
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolocue_bridge as YC  # noqa: E402

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger('t')
ok = lambda m: print('PASS', m)

# 1. prompts
cases = {
    'Pick up the carrot and place it on the purple plate.':
        'Pick up the carrot in the white box and place it on the purple plate in the orange box.',
    'Could you put the green chili pepper on the yellow plate?':
        'Pick up the green chili pepper in the white box and place it on the yellow plate in the orange box.',
    'Move the bunch of grapes onto the pink plate.':
        'Pick up the bunch of grapes in the white box and place it on the pink plate in the orange box.',
    'Put the watermellon on the blue plate.':
        'Pick up the watermelon in the white box and place it on the blue plate in the orange box.',
    'Place the black lipstick with a clear cap inside the top level of the organizer.':
        'Put the black lipstick with a clear cap in the white box into the top compartment in the orange box.',
    'Take the pink makeup puff and put it in the bottom section.':
        'Put the pink makeup puff in the white box into the bottom compartment in the orange box.',
}
for src, want in cases.items():
    got = YC.cue_prompts(src)
    assert got is not None and got[0] == want, (src, got)
    assert got[1] == YC.CUE_GENERIC[1 if 'plate' in src else 2]
assert YC.cue_prompts('Brush all paper balls off the tabletop into the dustpan.') is None
assert YC.cue_prompts('Put the spaceship on the white plate.') is None
ok(f'cue_prompts: {len(cases)} templates/aliases right, task 3 and unknown objects -> None')

# fake detector + session
SCRIPT = {'dets': []}


class FakeDet:
    plate_scores = {}

    def __call__(self, rgb):
        return list(SCRIPT['dets'])


YC.detector = lambda cfg: FakeDet()
CLOCK = {'t': 1000.0}
YC.time.monotonic = lambda: CLOCK['t']
FRAME = np.zeros((480, 640, 3), np.uint8)
CARROT_A, CARROT_B, PLATE = [430, 290, 550, 430], [250, 340, 360, 400], [170, 170, 310, 270]   # B: free table
DETS = [['carrot', *CARROT_A, 0.96], ['plate_purple', *PLATE, 0.93], ['red_chili', 130, 265, 255, 318, 0.95]]


class Policy:
    def __init__(self, task):
        self._task = task

    def preprocess(self, model_input, *, run_inference=True, rtc_prefix_len=None, prev_action_chunk=None):
        return {'task': self._task, YC.CAM: FRAME.copy()}

    def postprocess(self, model_output):
        return model_output


def session(task, cfg=None, mode='on'):
    mi = types.SimpleNamespace(features={YC.CAM: FRAME})
    server = types.SimpleNamespace(_robot_runtime=types.SimpleNamespace(mailbox=types.SimpleNamespace(get=lambda: mi)))
    sess = types.SimpleNamespace(session_id='test', task=task)
    pol = Policy(task)
    yc = YC.YoloCue(server, sess, pol, mode, LOG, '/tmp/yc_cue_test')
    yc.cfg.update(cfg or {})
    SCRIPT['dets'] = DETS
    yc.begin()
    return yc, pol


def chunk(right=None, left=None, n=44):
    l = np.zeros(n) if left is None else np.asarray(left, float)
    r = np.zeros(n) if right is None else np.asarray(right, float)
    return types.SimpleNamespace(actions={'action.left_gripper.position': l, 'action.right_gripper.position': r})


def white_at(img, box):
    x1, y1 = int(box[0]), int(box[1])
    return bool((img[y1 + 2, x1 + 20] == 255).all())


TASK = 'Pick up the carrot and place it on the purple plate.'
NAMED = YC.cue_prompts(TASK)[0]

# 2. prompt switch
yc, pol = session(TASK, {'cue_prompt': False})
o = pol.preprocess(None)
assert o['task'] == TASK and white_at(o[YC.CAM], CARROT_A)
yc, pol = session(TASK, {'cue_prompt': True})
o = pol.preprocess(None)
assert o['task'] == NAMED and white_at(o[YC.CAM], CARROT_A)
yc2, pol2 = session(TASK, {'cue_prompt': True}, mode='shadow')
assert pol2.preprocess(None)['task'] == TASK
ok('cue_prompt: off -> original instruction; on -> named prompt with the cue; shadow -> unchanged')

# 3. grasp, RTC-overlap chunk (open..close in its prefix) must not count as a release, then a real release -> regrasp
grasp_seq = np.r_[np.full(10, 0.9), np.full(34, 0.05)]                   # right gripper opens then closes
pol.postprocess(chunk(right=grasp_seq))
assert yc.grasp and yc.grasp['arm'] == 'right'
o = pol.preprocess(None)
assert not white_at(o[YC.CAM], CARROT_A) and o['task'] == NAMED          # white box gone after the grasp, name kept
pol.postprocess(chunk(right=np.r_[np.full(6, 0.9), np.full(38, 0.05)]))    # prefix still shows the opening
assert yc.release_t is None
pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))   # closed, then opens: release
assert yc.release_t == CLOCK['t']
CLOCK['t'] += 0.5
SCRIPT['dets'] = [['carrot', *CARROT_B, 0.9], ['plate_purple', *PLATE, 0.93]]
pol.preprocess(None)
assert yc.grasp is not None and yc.n_regrasp == 0                       # still inside release_wait_s
CLOCK['t'] += 1.5
o = pol.preprocess(None)
assert yc.n_regrasp == 1 and yc.grasp is None and yc.boxes[0] == CARROT_B and white_at(o[YC.CAM], CARROT_B)
ok('regrasp: RTC prefix opening ignored; release seen; after 1.5 s the carrot off the plate gets the white box back')

# 4. second grasp, release on the destination -> placed, no more checks
pol.postprocess(chunk(right=grasp_seq))
assert yc.grasp is not None
pol.postprocess(chunk(right=np.r_[np.full(20, 0.05), np.full(24, 0.9)]))
CLOCK['t'] += 2.0
SCRIPT['dets'] = [['carrot', 200, 190, 280, 250, 0.9], ['plate_purple', *PLATE, 0.93]]
pol.preprocess(None)
assert yc.placed and yc.n_regrasp == 1 and yc.release_t is None
pol.postprocess(chunk(right=np.r_[np.full(10, 0.05), np.full(34, 0.9)]))
assert yc.release_t is None
ok('placed: release with the carrot inside the destination box stops the checks')

# 4b. released on ANOTHER plate (not the named one) -> treated as placed, no white box back
yc, pol = session(TASK)
pol.postprocess(chunk(right=grasp_seq))
pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))
CLOCK['t'] += 2.0
SCRIPT['dets'] = [['carrot', 420, 200, 480, 240, 0.9], ['plate_blue', 395, 180, 543, 281, 0.97], ['plate_purple', *PLATE, 0.93]]
pol.preprocess(None)
assert yc.placed and yc.n_regrasp == 0
ok('task 1: an object resting on any detected plate is not regrasped')

# 4c. task 2: default regrasp_tasks [1] -> no watching; with [1, 2] an item seen across the neighbouring level is placed
T2 = 'Put the large black and gold makeup brush in the bottom compartment.'
ORG_B, ORG_M = [80, 200, 250, 254], [80, 140, 250, 199]
def t2_session(cfg):
    yc, pol = session(T2, cfg)
    SCRIPT['dets'] = [['brush_large', 103, 405, 213, 442, 0.9], ['organizer_bottom', *ORG_B, 0.9]]
    yc.boxes, yc.reason = None, None
    yc._detect(FRAME, 'test')
    return yc, pol
yc, pol = t2_session({})
assert yc.boxes is not None
pol.postprocess(chunk(right=grasp_seq)); pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))
assert yc.release_t is None
yc, pol = t2_session({'regrasp_tasks': [1, 2]})
pol.postprocess(chunk(right=grasp_seq)); pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))
assert yc.release_t is not None
CLOCK['t'] += 2.0
SCRIPT['dets'] = [['brush_large', 120, 150, 230, 190, 0.9], ['organizer_bottom', *ORG_B, 0.9], ['organizer_middle', *ORG_M, 0.9]]
pol.preprocess(None)
assert yc.placed and yc.n_regrasp == 0
ok('task 2: off by default; when on, an item seen across the neighbouring level counts as placed')

# 5. max_regrasp and the window
yc, pol = session(TASK, {'max_regrasp': 1, 'release_window_s': 3.0})
pol.postprocess(chunk(right=grasp_seq))
pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))
CLOCK['t'] += 2.0
SCRIPT['dets'] = [['carrot', *CARROT_B, 0.9]]
pol.preprocess(None)
assert yc.n_regrasp == 1
pol.postprocess(chunk(right=grasp_seq))
pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))
assert yc.release_t is None                                             # max_regrasp reached: no more watching
yc, pol = session(TASK, {'release_window_s': 3.0})
pol.postprocess(chunk(right=grasp_seq))
pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))
SCRIPT['dets'] = []                                                     # target never visible
for _ in range(6):
    CLOCK['t'] += 1.0
    pol.preprocess(None)
assert yc.release_t is None and yc.n_regrasp == 0
ok('limits: max_regrasp stops further checks; a release never resolved within the window is dropped')

# 5b. a target-class box where another object already stood in the cue frame (a look-alike that never moved) is no
#     regrasp target: here the red chili's spot is labelled 'carrot' after the release
yc, pol = session(TASK)
pol.postprocess(chunk(right=grasp_seq))
pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))
CLOCK['t'] += 2.0
SCRIPT['dets'] = [['carrot', 131, 266, 254, 317, 0.8], ['plate_purple', *PLATE, 0.93]]
pol.preprocess(None)
assert yc.n_regrasp == 0 and yc.release_t is not None
ok('look-alike that never moved (on the red chili spot) is not a regrasp target')

# 5c. DAgger handback: on_resume rebuilds the cue from the current frame
yc, pol = session(TASK)
pol.postprocess(chunk(right=grasp_seq))                                 # policy grasped ...
SCRIPT['dets'] = [['carrot', 200, 190, 280, 250, 0.9], ['plate_purple', *PLATE, 0.93]]
yc.on_resume()                                                          # ... the human placed it on the plate
o = pol.preprocess(None)
assert yc.placed and yc.grasp is not None and not white_at(o[YC.CAM], CARROT_A)
yc, pol = session(TASK)
pol.postprocess(chunk(right=grasp_seq))
SCRIPT['dets'] = [['carrot', *CARROT_B, 0.9], ['plate_purple', *PLATE, 0.93]]
yc.on_resume()                                                          # the human dropped it elsewhere on the table
o = pol.preprocess(None)
assert yc.grasp is None and yc.boxes[0] == CARROT_B and white_at(o[YC.CAM], CARROT_B) and not yc.placed
SCRIPT['dets'] = [['plate_purple', *PLATE, 0.93]]
yc.on_resume()                                                          # target not visible (in the gripper)
o = pol.preprocess(None)
assert yc.grasp is not None and not white_at(o[YC.CAM], CARROT_B)
ok('on_resume: placed -> no white box; visible on the table -> fresh cue; not visible -> treated as grasped')

# 6. regrasp off
yc, pol = session(TASK, {'regrasp': False})
pol.postprocess(chunk(right=grasp_seq))
pol.postprocess(chunk(right=np.r_[np.full(30, 0.05), np.full(14, 0.9)]))
assert yc.release_t is None
yc.close()
ok('regrasp: False disables the insurance; close() works')
print('ALL PASS')
