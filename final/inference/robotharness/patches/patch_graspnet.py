"""Team patch 11/16 for ~/robotharness/code/harness.py, on top of patch_grasp_fused.py. Idempotent (marker GraspNet).
User request 11/16 ~02:55: a small network trained on our task-3 data decides whether each gripper holds a tool.

Model: ~/robotharness/graspnet/graspnet.torchscript (205k params). Input: one wrist-camera frame resized to 128x96 RGB;
output: P(a tool handle is held between this gripper's fingers). Trained on the 216 official task-3 demos + rollouts
105a2d09 episodes 93-171, with labels from the gripper trace (holding band after an opening vs open / stuck halfway /
closed empty). Held-out rollouts 172-190: 99.8% (3211 crops, 5 false holds, 1 miss); it also catches holds the readings
cannot show (brush at 0.07, dustpan rim grasp at 0.05). Runs on CPU in the harness thread.
- The user turn gets "Grasp check from the wrist cameras: left holds a handle, right holds nothing." and the log line
  gets "[net left p, right p]".
- An arm counts as holding when the network says so (p >= 0.5) AND its gripper is closed and steady (GripperTrack).
  This replaces the LLM's left_holds/right_holds in the rules (those stay in the reply, for the log).
- grasp -> sweep: both arms holding on 2 replies in a row (unchanged count). The LLM's own sweep answer waits while a
  gripper is open (as before) or while the network says an arm does not hold, at most WAIT_MAX replies in a row
  (then it is let through, so a network miss cannot stall the robot).
- Black or blank wrist frames (mean < 12 or std < 8), missing model or any error: everything falls back to the fused rules with the LLM's left_holds/right_holds.
usage: python3 patch_graspnet.py <harness.py>"""
import sys
F = sys.argv[1]
s = open(F).read()
if 'class GraspNet' in s:
    print('already patched'); sys.exit(0)
assert 'class GripperTrack' in s, 'apply patch_grasp_fused.py first'


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''# ---------------------------------------------------------------------------------------------------------- schedulers
def grippers(obs):''', '''GRASPNET_FILE = Path(os.environ.get('ROBOTHARNESS_GRASPNET',
                                   str(Path.home() / 'robotharness' / 'graspnet' / 'graspnet.torchscript')))
if GRASPNET_FILE.exists():
    _gw = ('The gripper words cannot show which tool is held: use the images for that.')
    assert FIXED_SYSTEM.count(_gw) == 1
    FIXED_SYSTEM = FIXED_SYSTEM.replace(_gw, _gw + ' Each update also gives a grasp check from a detector trained on '
                                        'the wrist cameras of this robot: for each gripper it says whether a tool handle '
                                        'is held between its fingers. It is reliable; use it together with the images.')


class GraspNet:
    """Per-arm grasp detector (see patch_graspnet.py). probs([left_rgb, right_rgb]) -> [p_left, p_right] or None."""
    net, torch, error = None, None, None

    @classmethod
    def load(cls):
        if cls.net is None and cls.error is None:
            try:
                import torch
                cls.torch = torch
                cls.net = torch.jit.load(str(GRASPNET_FILE), map_location='cpu').eval()
            except Exception as e:
                cls.error = repr(e)
        return cls.net

    @classmethod
    def probs(cls, imgs):
        net = cls.load()
        if net is None:
            return None
        try:
            x = np.stack([np.asarray(Image.fromarray(np.asarray(im, np.uint8)).convert('RGB').resize((128, 96), Image.BILINEAR))
                          for im in imgs])
            if any(xi.mean() < 12 or xi.std() < 8 for xi in x):
                return None   # black / blank frame (camera dropout): the network says 'held' on those, do not use it
            with cls.torch.inference_mode():
                t = cls.torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255)
                return [float(v) for v in cls.torch.sigmoid(net(t))]
        except Exception as e:
            cls.error = repr(e)
            return None


# ---------------------------------------------------------------------------------------------------------- schedulers
def grippers(obs):''')

rep('''        self.grip_q, self.n_grasp_seen, self.n_grasp_override = None, 0, 0
''', '''        self.grip_q, self.n_grasp_seen, self.n_grasp_override = None, 0, 0
        self.net_q, self.n_sweep_wait = None, 0
''')

rep('''        self.grip_q = {a: self.tracks[a].snapshot(clock.now) for a in ('left', 'right')}''',
    '''        self.grip_q = {a: self.tracks[a].snapshot(clock.now) for a in ('left', 'right')}
        p = GraspNet.probs([cams[1], cams[2]])
        self.net_q = None if p is None else {'left': p[0], 'right': p[1]}''')

rep('''        log = f'{text} [raw left {lg:.3f}, right {rg:.3f}]'   # the viewer and log readers parse "left <number>"''',
    '''        if self.net_q:
            hw = lambda a: 'holds a handle' if self.net_q[a] >= 0.5 else 'holds nothing'
            text += f" Grasp check from the wrist cameras: left {hw('left')}, right {hw('right')}."
        log = f'{text} [raw left {lg:.3f}, right {rg:.3f}]'   # the viewer and log readers parse "left <number>"
        if self.net_q:
            log += f" [net left {self.net_q['left']:.2f}, right {self.net_q['right']:.2f}]"''')

rep('''    def grasp_rule(self, d):
        """True when each arm is seen holding its own tool and its gripper is closed and steady (any reading)."""
        if self.pan_side not in ('left', 'right') or not self.grip_q:
            return False''', '''    WAIT_MAX = 4   # LLM sweep answers held back in a row while the network says an arm does not hold

    def arm_held(self, a):
        return bool(self.net_q) and self.net_q[a] >= 0.5 and self.grip_q[a]['state'] == 'closed'

    def grasp_rule(self, d):
        """True when each arm is seen holding its own tool and its gripper is closed and steady (any reading).
        With the grasp network available, 'seen holding' comes from the network instead of the LLM's fields."""
        if self.net_q and self.grip_q:
            return all(self.arm_held(a) for a in ('left', 'right'))
        if self.pan_side not in ('left', 'right') or not self.grip_q:
            return False''')

rep('''        if self.subtask == 'grasp' and sub == 'sweep' and self.grip_q:
            loose = [a for a in ('left', 'right') if self.grip_q[a]['state'] in ('open', 'opening', 'not_opened')]
            if loose:   # an open gripper holds nothing: wait (closing / stopped halfway do not block)
                sub = 'grasp'
                self.events.append({'t': round(now, 2), 'grasp_rule': f'stay in grasp: {"/".join(loose)} gripper '
                                    f'{self.grip_q[loose[0]]["state"]}', 'llm_subtask': 'sweep'})''',
    '''        if self.subtask == 'grasp' and sub == 'sweep' and self.grip_q:
            loose = [a for a in ('left', 'right') if self.grip_q[a]['state'] in ('open', 'opening', 'not_opened')]
            unheld = [a for a in ('left', 'right') if self.net_q and not self.arm_held(a)]
            if loose:   # an open gripper holds nothing: wait (closing / stopped halfway do not block)
                sub = 'grasp'
                self.events.append({'t': round(now, 2), 'grasp_rule': f'stay in grasp: {"/".join(loose)} gripper '
                                    f'{self.grip_q[loose[0]]["state"]}', 'llm_subtask': 'sweep'})
            elif unheld and self.n_sweep_wait < self.WAIT_MAX:
                self.n_sweep_wait += 1
                sub = 'grasp'
                self.events.append({'t': round(now, 2), 'grasp_rule': f'stay in grasp: grasp check says {"/".join(unheld)} '
                                    f'not holding ({self.n_sweep_wait}/{self.WAIT_MAX})', 'llm_subtask': 'sweep'})
        if d['subtask'] != 'sweep' or self.subtask != 'grasp':
            self.n_sweep_wait = 0''')

open(F, 'w').write(s)
print('patched')
