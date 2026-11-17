"""Team patch 11/16 for ~/robotharness/code/harness.py, on top of patch_graspnet.py. Idempotent (marker NUDGE_FILE).
User request 11/16 ~03:45: keep the grasp network, but help the brush arm when it gets stuck.

Seen on the 5080 (13500, 03:09-03:36): the brush arm reaches the handle and the VLA commands close (chunk command
0.03-0.2), but the gripper stays open or half open for 8-31 s (fingertips on the table or on top of the handle). With the
grasp network the harness keeps the grasp instruction until the brush is really held, so nothing moves the arm out.
Earlier harnesses often sent sweep too early; the VLA then kept grasping the brush under the sweep instruction while
moving the arm.

Nudge: while the subtask is grasp, one arm holds (network + gripper closed and steady) and the other arm has opened
but the network says it does not hold for STUCK_S (usually the brush arm; replay ep173 had the dustpan arm stuck), the instruction is switched to sweep for NUDGE_S, then back
to grasp. At most NUDGE_MAX per session, NUDGE_GAP_S apart. Each nudge is recorded in the scheduler events ('nudge').
Switch file ~/robotharness/nudge, read at session start: 'off' disables it; missing or anything else = on.
usage: python3 patch_nudge.py <harness.py>"""
import sys
F = sys.argv[1]
s = open(F).read()
if 'NUDGE_FILE' in s:
    print('already patched'); sys.exit(0)
assert 'class GraspNet' in s, 'apply patch_graspnet.py first'


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''class GraspNet:''', '''NUDGE_FILE = Path.home() / 'robotharness' / 'nudge'   # team 11/16: 'off' disables the stuck-brush nudge


def nudge_on():
    try:
        return NUDGE_FILE.read_text().strip() != 'off'
    except OSError:
        return True


class GraspNet:''')

rep('''        self.net_q, self.n_sweep_wait = None, 0
''', '''        self.net_q, self.n_sweep_wait = None, 0
        self.nudge_enabled = nudge_on()
        self.stuck_since, self.nudge_until, self.n_nudge, self.last_nudge = None, None, 0, -1e9
''')

rep('''    WAIT_MAX = 4   # LLM sweep answers held back in a row while the network says an arm does not hold
''', '''    WAIT_MAX = 4   # LLM sweep answers held back in a row while the network says an arm does not hold
    STUCK_S, NUDGE_S, NUDGE_GAP_S, NUDGE_MAX = 8.0, 2.4, 10.0, 2

    def prompt(self):
        if self.nudge_until is not None and self.pan_side in ('left', 'right'):
            return RS.instruction('sweep', self.pan_side)
        return self.current

    def nudge_check(self, now):
        """Stuck brush arm -> a short sweep instruction (see patch_nudge.py)."""
        if self.nudge_until is not None:
            if now >= self.nudge_until:
                self.nudge_until, self.stuck_since = None, now
                self.events.append({'t': round(now, 2), 'nudge': 'end, back to grasp'})
            return
        if not self.nudge_enabled or self.subtask != 'grasp' or not self.net_q or self.pan_side not in ('left', 'right'):
            self.stuck_since = None
            return
        held = {a: self.net_q[a] >= 0.5 and self.tracks[a].snapshot(now)['state'] == 'closed' for a in ('left', 'right')}
        late = [a for a in ('left', 'right') if not held[a] and held[RS.other(a)]
                and self.tracks[a].snapshot(now)['state'] != 'not_opened']
        stuck = bool(late)
        brush = late[0] if late else RS.other(self.pan_side)
        if not stuck:
            self.stuck_since = None
            return
        if self.stuck_since is None:
            self.stuck_since = now
        if (now - self.stuck_since >= self.STUCK_S and self.n_nudge < self.NUDGE_MAX
                and now - self.last_nudge >= self.NUDGE_GAP_S):
            self.nudge_until, self.last_nudge = now + self.NUDGE_S, now
            self.n_nudge += 1
            self.events.append({'t': round(now, 2), 'nudge': f'{brush} arm stuck {now - self.stuck_since:.1f} s '
                                f'(grasp check {self.net_q[brush]:.2f}, gripper {self.tracks[brush].snapshot(now)["state"]}): '
                                f'sweep for {self.NUDGE_S} s ({self.n_nudge}/{self.NUDGE_MAX})'})
''')

rep('''        self.track(obs, clock.now)   # team 11/16: gripper motion history for GripperTrack
''', '''        self.track(obs, clock.now)   # team 11/16: gripper motion history for GripperTrack
        self.nudge_check(clock.now)
''')

rep('''                 n_grasp_override=self.n_grasp_override)
''', '''                 n_grasp_override=self.n_grasp_override, n_nudge=self.n_nudge)
''')

open(F, 'w').write(s)
print('patched')
