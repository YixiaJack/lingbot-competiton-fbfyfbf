"""Fixed four-subtask split of task 3 (sweeping), derived from the official real-robot demos (2026-11-14).

Every harness run uses this split: the LLM (DeepSeek-V4.1-Flash) only says which subtask the robot is in; the harness
turns that into the fixed instruction below and sends it to the VLA. The same function labels the training demos, so the
VLA is fine-tuned on exactly the strings the harness sends. No simulator imports; numpy only.

What the 216 official task-3 demos show (segment_real.py, docs in ../README.md):
  - Two mirrored layouts, 108 each: dustpan on the right (right arm takes it, brush on the left) or on the left.
  - The arm that grasps first always takes the dustpan. Holding the dustpan handle the gripper reads 0.318-0.338, the
    brush handle 0.218-0.249: no overlap, so the roles can be read off the grippers.
  - Each gripper opens exactly twice (216/216): once to grasp, once to put the tool down. Both tools are put down at
    the same moment, then both arms go home.
  - Median durations: grasp 6.7 s, sweep 3.9 s, lift-and-pour 5.2 s, put down and home 10.6 s (27 s in total).

Why four subtasks and where the cuts are:
  grasp  both tools picked up. The two grasps follow each other with nothing in between and always in the same order
         (dustpan first), so one instruction covers both; splitting them would add a 2-3 s subtask whose boundary is a
         gripper closing, which the overview camera shows badly (in the 11/14 sim test the LLM left the dustpan grasp
         early 12 times in 30).
  sweep  every ball inside the dustpan. Ends when the dustpan arm starts lifting.
  dump   lift over the basket and tip it. Lifting and tipping are one continuous reach of the dustpan arm with no pause;
         ends when the arm starts back from the pour pose, about 1 s after the balls have dropped (the same rule as the
         11/14 sim relabel: the LLM reads "balls in the basket" as the end of pouring).
  home   put both tools down, release, both arms home and still. One continuous motion, no decision inside it.
"""
import numpy as np

FPS = 30
FULL_INSTRUCTION = ('Brush all paper balls off the tabletop into the dustpan before pouring them into the basket. '
                    'Place both tools down; return both arms home and remain stationary.')

# instruction templates: {pan} is the side of the arm holding the dustpan, {brush} the other side ('left' / 'right')
SUBTASKS = [
    {'id': 'grasp', 'name': 'Grasp the tools',
     'instruction': 'Grasp the dustpan handle with the {pan} gripper, then grasp the brush handle with the {brush} gripper.',
     'goal': 'The arm on the dustpan side holds the dustpan by its handle and the other arm holds the brush by its '
             'handle (gripper readings: about 0.32 on the dustpan arm, about 0.23 on the brush arm).'},
    {'id': 'sweep', 'name': 'Sweep the paper balls into the dustpan',
     'instruction': 'Sweep all the paper balls into the dustpan with the brush.',
     'goal': 'Every paper ball that was on the table is inside the dustpan. If a ball is missed or rolls out, this '
             'subtask is not done.'},
    {'id': 'dump', 'name': 'Lift the dustpan over the basket and pour',
     'instruction': 'Lift the dustpan over the basket and pour the paper balls into the basket.',
     'goal': 'The dustpan has been lifted over the white basket and tipped, and every paper ball has fallen into the '
             'basket.'},
    {'id': 'home', 'name': 'Put the tools down and return home',
     'instruction': 'Put the dustpan and the brush down on the table, then return both arms home and stay still.',
     'goal': 'The dustpan and the brush lie on the table, both grippers have let go (they read 0 again), and both arms '
             'are back at their start pose and still. When this is done the whole task is complete.'},
]
SUBTASK_IDS = [s['id'] for s in SUBTASKS]
BY_ID = {s['id']: s for s in SUBTASKS}


def other(side):
    return 'right' if side == 'left' else 'left'


def instruction(subtask, pan_side):
    """The exact string the VLA gets for a subtask (training labels and harness use this same function)."""
    return BY_ID[subtask]['instruction'].format(pan=pan_side, brush=other(pan_side))


# ------------------------------------------------------------------------------------------- labelling the demos
def runs(mask):
    """[(start, end)] runs of True."""
    m = np.r_[False, np.asarray(mask, bool), False].astype(int); d = np.diff(m)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def segment(gr, ee, open_thr=0.38, min_open=5):
    """Boundaries of the four subtasks in one demo.

    gr: (T, 2) observation.state.effector.position (left, right). ee: (T, 14) observation.state.ee.position
    (left xyz + quat, right xyz + quat, each in its own arm frame; x points forward, away from the robot).
    open_thr sits between the highest holding reading (0.338) and the lowest grasp opening seen (0.44).
    Returns frame indices: grasp = [0, b1), sweep = [b1, b2), dump = [b2, b3), home = [b3, T).
    """
    gr, ee = np.asarray(gr, float), np.asarray(ee, float); T = len(gr)
    opens = [[(s, e) for s, e in runs(gr[:, a] > open_thr) if e - s >= min_open] for a in (0, 1)]
    if min(len(o) for o in opens) < 2:
        raise ValueError(f'expected two gripper openings per arm, got {[len(o) for o in opens]}')
    def settled(a, close_at):                               # gripper reaches its holding value after closing
        hold = float(np.median(gr[close_at + 10: close_at + 40, a]))
        j = close_at
        while j < T - 1 and abs(gr[j, a] - hold) > 0.02:
            j += 1
        return j, hold

    st = [settled(a, opens[a][0][1]) for a in (0, 1)]
    release = min(opens[0][-1][0], opens[1][-1][0])
    # The arm that grasps first takes the dustpan (216/216 official demos: it opens first and closes first). Two of our
    # own demos (11/14: ep80, ep95) open the brush gripper first but close it last; when BOTH the arm that finishes its
    # grasp first and the arm that reaches farthest forward during the pour disagree with the first-opening arm, they
    # win. Identical to the old rule on all 216 official demos. b1 = both tools held (the later settle).
    first_open = int(np.argmin([o[0][0] for o in opens]))
    first_held = int(np.argmin([s[0] for s in st]))
    both = max(s[0] for s in st)
    reach = [float(ee[both:release, 7 * a].max()) if release > both else -1.0 for a in (0, 1)]
    far = int(np.argmax(reach))
    pan = first_open if (first_held == first_open or far == first_open) else first_held
    brush = 1 - pan
    g_pan, hold_pan = st[pan]
    hold_brush = st[brush][1]
    b1 = both
    x, z = ee[:, 7 * pan], ee[:, 7 * pan + 2]
    xmax = float(x[b1:release].max())                       # the pour pose: farthest forward reach of the dustpan arm
    near = np.where(x[b1:release] >= xmax - 0.04)[0] + b1
    p_in, p_out = int(near[0]), int(near[-1])
    zref = float(np.percentile(z[b1:p_in], 30)) if p_in - b1 > 10 else float(z[b1])
    b2 = p_in
    while b2 > b1 and z[b2] > zref + 0.03:                  # back to where the lift started
        b2 -= 1
    b3 = p_out
    if not (0 < b1 < b2 < b3 < release < T):
        raise ValueError(f'boundaries out of order: {b1}, {b2}, {b3}, release {release}, T {T}')
    return {'pan_arm': 'left' if pan == 0 else 'right', 'b1': int(b1), 'b2': int(b2), 'b3': int(b3),
            'grasp_dustpan_done': int(g_pan), 'release': int(release), 'T': T,
            'hold_pan': round(hold_pan, 3), 'hold_brush': round(hold_brush, 3), 'x_pour': round(xmax, 3)}


def frame_labels(seg):
    """Per-frame subtask ids for a segment() result."""
    lab = np.empty(seg['T'], dtype=object)
    lab[:seg['b1']] = 'grasp'; lab[seg['b1']:seg['b2']] = 'sweep'; lab[seg['b2']:seg['b3']] = 'dump'; lab[seg['b3']:] = 'home'
    return lab


def roles_from_grippers(left, right):
    """Dustpan side read off the gripper readings once both tools are held, else None."""
    def what(g):
        return 'pan' if 0.29 <= g <= 0.36 else 'brush' if 0.19 <= g <= 0.27 else None
    l, r = what(left), what(right)
    if l == 'pan' and r == 'brush': return 'left'
    if r == 'pan' and l == 'brush': return 'right'
    return None
