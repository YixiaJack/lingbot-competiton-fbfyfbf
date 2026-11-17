"""Team patch 11/16 for ~/robotharness/code/harness.py (Fixed harness on the 5080, on top of the ball-description and
no-lift patches). Idempotent (marker GripperTrack). User request 11/16 ~02:30: a grasp check that needs no lift and
does not rely only on gripper readings, with vision allowed.

Why: the 5080 bridge passes raw gripper metres (0-0.1) while the prompt calibrated holding at 0.32 / 0.23 (0-1 units).
In the 02:07 run (a34f7758) and in replays the LLM kept the robot in grasp: "readings 0.03, not the firm grasp
readings" / "tools still on the table, not lifted".

1. GripperTrack, per arm, from the reading history only (units-free: raw metres or 0-1 units, detected from the
   session peak). States: not opened yet / opening / open / closing / closed and steady. A gripper found steady
   clearly above its start reading without a seen opening (it opened and closed between two samples) counts as opened. Closed-and-steady means: it
   opened, then closed (by CLOSE_DROP from its open peak, or to below half open) and moved less than STEADY_TOL
   over STEADY_S. It is also
   marked as stopped above fully closed (likely something between the fingers) or fully closed (empty, or a very thin
   handle).
2. The LLM gets these words instead of numbers (the log line still carries the raw numbers for the viewer). The
   0.32 / 0.23 calibration is removed from the prompt. A new item d) asks, per gripper, what is between its fingers,
   judged only from the images: left_holds / right_holds = dustpan | brush | nothing | unclear.
3. Rule (it only adds a way into sweep; the LLM's own sweep answer still switches at once): while the LLM still says
   grasp, switch to sweep when, on 2 replies in a row, each arm is seen holding its own tool (the dustpan arm "dustpan",
   the other "brush") AND that gripper is closed and steady at query time (any reading).
   The LLM's own grasp -> sweep answer waits while a gripper is open / opening / not opened yet (an open gripper holds
   nothing; closing or stopped halfway does not block). A gripper that stops above half open counts as "stopped
   halfway" (no firm grip), not closed; a closed gripper that moves again is re-judged.
   Back from sweep to grasp (the LLM says grasp) only with evidence the grasp is lost: a gripper is no longer closed
   and steady, or an arm is not seen holding its tool; otherwise stay in sweep (no grasp/sweep ping-pong).
   Overrides are recorded in the scheduler events (grasp_rule), so they appear in the session summary.
usage: python3 patch_grasp_fused.py <harness.py>"""
import sys
F = sys.argv[1]
s = open(F).read()
if 'GripperTrack' in s:
    print('already patched'); sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# --- prompt: built from the formatted FIXED_SYSTEM, right after it is defined -----------------------------------
rep('''# ---------------------------------------------------------------------------------------------------------- schedulers
def grippers(obs):''', '''# team 11/16 (user): grasp judged by sight + gripper motion, not by readings or lifting (see GripperTrack)
for _old, _new in (
        ('Gripper readings: 0 means closed with nothing in it (the start and the end state); opening to grasp or to let '
         'go reads 0.4-0.9; holding the dustpan handle reads about 0.32; holding the brush handle reads about 0.23.',
         'Each update also says what each gripper is doing: not opened yet, opening, open, closing, or closed and steady. '
         'A gripper that closed and stopped above fully closed usually has something between its fingers; a very thin '
         'handle can still close fully; a gripper that stopped about halfway has no firm grip; an open or opening gripper '
         'holds nothing. The gripper words cannot show which tool is held: use the images for that.'),
        (' (gripper readings: about 0.32 on the dustpan arm, about 0.23 on the brush arm)', ''),
        ('(they read 0 again)', '(they are fully closed again)'),
        ('the robot then stops.\\n\\nReply with one JSON object and nothing else:',
         'the robot then stops.\\n'
         'd) For each gripper, say what is between its fingers right now, judged only from the images (that arm\\'s wrist '
         'camera first, then the top camera): "dustpan" (the dustpan handle), "brush" (the brush handle), "nothing", or '
         '"unclear". Do not decide this from the gripper words, and a tool does not have to be lifted to be held.\\n\\n'
         'Reply with one JSON object and nothing else:'),
        ('"subtask": "<grasp|sweep|dump|home>", "task_complete": false}',
         '"subtask": "<grasp|sweep|dump|home>", "left_holds": "<dustpan|brush|nothing|unclear>", '
         '"right_holds": "<dustpan|brush|nothing|unclear>", "task_complete": false}')):
    assert FIXED_SYSTEM.count(_old) == 1, _old[:60]
    FIXED_SYSTEM = FIXED_SYSTEM.replace(_old, _new)


class GripperTrack:
    """What one gripper is doing, from its reading history only. Works in 0-1 opening units: raw metres (fully open
    ~0.1, as the 5080 bridge passes them) are scaled x10, detected from the session peak."""
    OPEN_RISE, CLOSE_DROP, STEADY_TOL, STEADY_S, ABOVE_ZERO, CLOSED_MAX = 0.3, 0.3, 0.03, 1.2, 0.12, 0.5

    def __init__(self):
        self.samples = []                       # (t, raw)
        self.max_raw = 0.0
        self.low = self.peak = None             # raw: closed reference and highest reading since it
        self.opened, self.closed_since, self.closed_raw = False, None, None

    def k(self):
        return 10.0 if self.max_raw <= 0.2 else 1.0

    def window(self, t):
        """Samples of the last STEADY_S, or the last two if sampling is sparser (<= 2.5 s apart)."""
        win = [(a, b) for a, b in self.samples if a >= t - self.STEADY_S]
        if len(win) < 2 and len(self.samples) >= 2 and self.samples[-1][0] - self.samples[-2][0] <= 2.5:
            win = self.samples[-2:]
        return win

    def steady(self, t):
        win = self.window(t)
        return len(win) >= 2 and (max(b for a, b in win) - min(b for a, b in win)) * self.k() <= self.STEADY_TOL

    def update(self, t, raw):
        raw = float(raw)
        if self.samples and t <= self.samples[-1][0]:
            return
        self.samples.append((t, raw))
        self.samples = [(a, b) for a, b in self.samples if a >= t - 10.0]
        self.max_raw = max(self.max_raw, raw)
        k = self.k()
        if self.low is None:
            self.low = self.peak = raw
        steady = self.steady(t)
        if self.closed_since is not None:
            if (raw - self.closed_raw) * k >= self.OPEN_RISE:          # opens again (regrasp or release)
                self.closed_since, self.low, self.peak, self.opened = None, self.closed_raw, raw, True
                return
            if abs(raw - self.closed_raw) * k <= 2 * self.STEADY_TOL:
                return
            self.closed_since = None                                     # moves again: re-judged once steady
        if not self.opened:
            self.low = min(self.low, raw)
            self.peak = max(self.peak, raw)
            self.opened = (self.peak - self.low) * k >= self.OPEN_RISE
            if not self.opened and steady and (raw - self.low) * k >= self.ABOVE_ZERO:
                self.opened = True   # opened and closed between two samples: now resting on something
            if not self.opened:
                return
        self.peak = max(self.peak, raw)
        if steady and ((self.peak - raw) * k >= self.CLOSE_DROP or raw * k <= self.CLOSED_MAX):
            self.closed_since, self.closed_raw = self.window(t)[0][0], raw

    def snapshot(self, t):
        k = self.k()
        last = self.samples[-1][1] if self.samples else 0.0
        if self.closed_since is not None:
            return {'state': 'closed' if last * k <= self.CLOSED_MAX else 'partly', 'steady_s': round(t - self.closed_since, 1),
                    'above_zero': last * k >= self.ABOVE_ZERO}
        if not self.opened:
            return {'state': 'not_opened'}
        prev = self.samples[-2][1] if len(self.samples) >= 2 else last
        if self.steady(t):
            return {'state': 'open'}
        return {'state': 'closing' if last < prev else 'opening'}

    @staticmethod
    def words(snap):
        st = snap['state']
        if st == 'partly':
            return (f"stopped about halfway and steady for {snap['steady_s']:.1f} s: not a firm grip (the fingertips may "
                    "be on the table or on the edge of the tool)")
        if st == 'closed':
            where = ('stopped above fully closed (something is likely between the fingers)' if snap['above_zero']
                     else 'fully closed (empty, or a very thin handle)')
            return f"closed and steady for {snap['steady_s']:.1f} s, {where}"
        return {'not_opened': 'not opened yet', 'open': 'open', 'opening': 'opening', 'closing': 'closing'}[st]


# ---------------------------------------------------------------------------------------------------------- schedulers
def grippers(obs):''')

# --- Fixed: tracks, words in the user turn, the rule -------------------------------------------------------------
rep('''    def __init__(self, img_size=(320, 240), **kw):
        super().__init__(FIXED_SYSTEM, img_size=img_size, **kw)
        self.subtask, self.pan_side, self.n_side_mismatch = None, None, 0
        self.llm_complete_t, self.arm_hist = None, []
''', '''    GRASP_SEEN_N = 2   # team 11/16: replies in a row with both arms seen holding + grippers closed and steady

    def __init__(self, img_size=(320, 240), **kw):
        super().__init__(FIXED_SYSTEM, img_size=img_size, **kw)
        self.subtask, self.pan_side, self.n_side_mismatch = None, None, 0
        self.llm_complete_t, self.arm_hist = None, []
        self.tracks = {'left': GripperTrack(), 'right': GripperTrack()}
        self.grip_q, self.n_grasp_seen, self.n_grasp_override = None, 0, 0

    def track(self, obs, now):
        lg, rg = view(obs)[1]
        self.tracks['left'].update(now, lg); self.tracks['right'].update(now, rg)

    def user_turn(self, obs, clock, first):
        cams, (lg, rg) = view(obs)
        self.track(obs, clock.now)
        self.grip_q = {a: self.tracks[a].snapshot(clock.now) for a in ('left', 'right')}
        if first:
            head = 'Start of the task. Time 0.0 s of 120 s.'
        else:
            head = f'Time {clock.now:.1f} s of 120 s. Running now: {self.running_text()} (for {clock.now - self.since:.1f} s).'
        text = (f"{head} Grippers: left {GripperTrack.words(self.grip_q['left'])}; "
                f"right {GripperTrack.words(self.grip_q['right'])}.")
        log = f'{text} [raw left {lg:.3f}, right {rg:.3f}]'   # the viewer and log readers parse "left <number>"
        return [{'type': 'text', 'text': text}] + [jpeg_part(c, size=self.img_size) for c in cams], log

    def grasp_rule(self, d):
        """True when each arm is seen holding its own tool and its gripper is closed and steady (any reading)."""
        if self.pan_side not in ('left', 'right') or not self.grip_q:
            return False
        want = {self.pan_side: 'dustpan', RS.other(self.pan_side): 'brush'}
        return all(str(d.get(f'{a}_holds', '')).strip().lower() == want[a] and self.grip_q[a]['state'] == 'closed'
                   for a in ('left', 'right'))
''')

rep('''        self.subtask = d['subtask']
        self.set(RS.instruction(self.subtask, self.pan_side), now, subtask=self.subtask, side=self.pan_side,
                 observation=str(d.get('observation', ''))[:300])
''', '''        sub = d['subtask']
        if self.subtask == 'grasp' and sub == 'sweep' and self.grip_q:
            loose = [a for a in ('left', 'right') if self.grip_q[a]['state'] in ('open', 'opening', 'not_opened')]
            if loose:   # an open gripper holds nothing: wait (closing / stopped halfway do not block)
                sub = 'grasp'
                self.events.append({'t': round(now, 2), 'grasp_rule': f'stay in grasp: {"/".join(loose)} gripper '
                                    f'{self.grip_q[loose[0]]["state"]}', 'llm_subtask': 'sweep'})
        if self.subtask == 'grasp' and sub == 'grasp' and d['subtask'] == 'grasp':
            if self.grasp_rule(d):
                self.n_grasp_seen += 1
                if self.n_grasp_seen >= self.GRASP_SEEN_N:
                    sub = 'sweep'
                    self.n_grasp_override += 1
                    self.events.append({'t': round(now, 2), 'grasp_rule': 'both arms seen holding their tools, grippers '
                                        f'closed and steady on {self.n_grasp_seen} replies', 'llm_subtask': 'grasp'})
            else:
                self.n_grasp_seen = 0
        elif self.subtask == 'sweep' and sub == 'grasp' and self.grasp_rule(d):
            sub = 'sweep'   # back to grasp only with evidence the grasp is lost (a gripper reopened / a tool not seen held)
            self.n_grasp_seen = 0
            self.events.append({'t': round(now, 2), 'grasp_rule': 'stay in sweep: both arms still seen holding their '
                                'tools, grippers closed and steady', 'llm_subtask': 'grasp'})
        else:
            self.n_grasp_seen = 0
        self.subtask = sub
        self.set(RS.instruction(self.subtask, self.pan_side), now, subtask=self.subtask, side=self.pan_side,
                 observation=str(d.get('observation', ''))[:300])
''')

rep('''    def on_chunk(self, obs, clock, env=None):
        # Gripper evidence must repeat on two consecutive chunks and is ignored in 'home': on the 216 demos the''',
    '''    def on_chunk(self, obs, clock, env=None):
        self.track(obs, clock.now)   # team 11/16: gripper motion history for GripperTrack
        # Gripper evidence must repeat on two consecutive chunks and is ignored in 'home': on the 216 demos the''')

rep('''        s.update(pan_side=self.pan_side, n_side_mismatch=self.n_side_mismatch, llm_complete_t=self.llm_complete_t)
''', '''        s.update(pan_side=self.pan_side, n_side_mismatch=self.n_side_mismatch, llm_complete_t=self.llm_complete_t,
                 n_grasp_override=self.n_grasp_override)
''')

open(F, 'w').write(s)
print('patched')
