"""Team patch 11/16 for ~/robotharness/code/harness.py (the 11/14 20:06 Fixed harness on the 5080). Idempotent
(marker ONE_BALL_FILE).

User request: when the scene starts with one paper ball, go from sweep to dump when the three cameras show that one
ball inside the dustpan, and only then.
- Switch file ~/robotharness/one_ball, read when each session's scheduler is created (no restart needed to flip it):
  'on' = rule active; missing / anything else = the 11/14 harness byte for byte (same system prompt, same decisions).
- With the rule on, the prompt also gets the user's ball facts (11/16 00:50; the tool description stays as it was):
  the paper balls are hard, one red and one yellow at most (or only one of them); at the start every ball is fully
  visible in the top view. The LLM also returns balls_in_dustpan and balls_elsewhere as colour lists of the balls it actually SEES (a ball
  in several images is one ball); a hidden ball is in neither list. (21:04 run on 11/15: the ball stuck on the brush bristles
  went out of view and the LLM called dump five times: "no longer visible ... presumably inside the dustpan".)
- Ball count at the start = the most frequent non-zero number of distinct colours over the grasp-phase replies (balls are
  not moved before sweeping). The rule only acts when that count is 1; with two balls the LLM decides as before.
- One ball:
    sweep (or the reply that ends grasp): the ball seen in the dustpan and none elsewhere ->
           dump at once if the LLM also says dump; if the LLM still says sweep, dump on the second such reply in a
           row (20:56 run waited ~9 s; replay of real ep 178: one reply listed the ball while it was rolling on the
           lip and the next saw it on the table again).
           The LLM says dump without seeing the ball in the pan: allowed if the ball was last seen inside the
           dustpan (the pan is being lifted / tipped, or the ball is already in the basket); refused if it was last
           seen outside (21:04: the ball sat on the brush bristles, out of view) or is seen outside now.
    dump:  the LLM says sweep but sees no ball outside the dustpan (elsewhere == 0) -> stay in dump (the ball is
           just out of view while the pan is lifted or tipped); with a ball seen outside -> back to sweep.
    grasp -> sweep and dump -> home: the LLM decides as before.
  Every override is recorded in the scheduler events (ball_rule) and so in the session summary.
usage: python3 patch_one_ball.py <harness.py>"""
import sys
F = sys.argv[1]
s = open(F).read()
if 'ONE_BALL_FILE' in s:
    print('already patched'); sys.exit(0)


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, n, old[:90])
    s = s.replace(old, new)


# 1. the extra prompt section and JSON fields, built from the formatted FIXED_SYSTEM so the 'off' prompt is untouched
rep('''# ---------------------------------------------------------------------------------------------------------- schedulers
def grippers(obs):''', '''# team 11/16 (user): one-ball rule for sweep -> dump, switched by ~/robotharness/one_ball ('on' = active)
ONE_BALL_FILE = Path.home() / 'robotharness' / 'one_ball'


def one_ball_on():
    try:
        return ONE_BALL_FILE.read_text().strip() == 'on'
    except OSError:
        return False


_SCENE_OLD = ("A cream-coloured dustpan and a brush with a wooden handle and yellow bristles lie on the table, one on the "
              "left and one on the right (either way round). One or two crumpled paper balls (yellow and/or red) lie "
              "between them.")
_SCENE_NEW = (_SCENE_OLD.split(" One or two crumpled")[0] + " The paper balls are hard crumpled balls: one red and one "
              "yellow at most, never more than these two, and a scene may have only one of them. At the start they lie "
              "between the tools, and every paper ball of the scene is fully visible in the top image.")
_BALLS_TEXT = """d) Say which paper balls you can actually see, by colour, using all three images together (a ball that shows \\
in more than one image is still one ball):
- balls_in_dustpan: the balls you see lying inside the dustpan's bowl.
- balls_elsewhere: the balls you see anywhere else except the basket: on the table, on or stuck to the brush, on the \\
dustpan's rim or lip, in a gripper.
A ball you cannot see right now goes in neither list. Never put a ball in balls_in_dustpan only because it is no longer \\
visible on the table or on the brush.

"""
assert FIXED_SYSTEM.count(_SCENE_OLD) == 1
assert FIXED_SYSTEM.count('Reply with one JSON object and nothing else:') == 1
assert FIXED_SYSTEM.count('"subtask": "<grasp|sweep|dump|home>", "task_complete": false}') == 1
FIXED_SYSTEM_BALLS = FIXED_SYSTEM.replace(_SCENE_OLD, _SCENE_NEW).replace(
    'Reply with one JSON object and nothing else:', _BALLS_TEXT + 'Reply with one JSON object and nothing else:').replace(
    '"subtask": "<grasp|sweep|dump|home>", "task_complete": false}',
    '"subtask": "<grasp|sweep|dump|home>", "balls_in_dustpan": [<zero or more of "red", "yellow">], '
    '"balls_elsewhere": [<zero or more of "red", "yellow">], '
    '"task_complete": false}')


# ---------------------------------------------------------------------------------------------------------- schedulers
def grippers(obs):''')

# 2. Fixed picks the prompt at construction and keeps the ball state
rep('''    def __init__(self, img_size=(320, 240), **kw):
        super().__init__(FIXED_SYSTEM, img_size=img_size, **kw)
        self.subtask, self.pan_side, self.n_side_mismatch = None, None, 0
        self.llm_complete_t, self.arm_hist = None, []
''', '''    def __init__(self, img_size=(320, 240), **kw):
        self.one_ball = one_ball_on()   # team 11/16: read once per session
        super().__init__(FIXED_SYSTEM_BALLS if self.one_ball else FIXED_SYSTEM, img_size=img_size, **kw)
        self.subtask, self.pan_side, self.n_side_mismatch = None, None, 0
        self.llm_complete_t, self.arm_hist = None, []
        self.start_counts, self.n_ball_override, self.in_streak, self.last_seen = [], 0, 0, None
''')

# 3. the rule, applied to the LLM's subtask before it is used
rep('''        self.subtask = d['subtask']
        self.set(RS.instruction(self.subtask, self.pan_side), now, subtask=self.subtask, side=self.pan_side,
                 observation=str(d.get('observation', ''))[:300])
''', '''        sub, why = self.ball_rule(d)
        if why:
            self.n_ball_override += 1
            self.events.append({'t': round(now, 2), 'ball_rule': why, 'llm_subtask': d['subtask'], 'subtask': sub})
        self.subtask = sub
        self.set(RS.instruction(self.subtask, self.pan_side), now, subtask=self.subtask, side=self.pan_side,
                 observation=str(d.get('observation', ''))[:300])
''')

rep('''    def arms_still(self, now):''', '''    def balls_start(self):
        """Ball count at the start: most frequent non-zero total over the grasp-phase replies (ties -> larger)."""
        c = [n for n in self.start_counts if n > 0]
        return max(set(c), key=lambda n: (c.count(n), n)) if c else None

    def ball_rule(self, d):
        """(subtask to use, reason or None). See ONE_BALL_FILE: only with one ball at the start, only sweep <-> dump."""
        sub = d['subtask']
        if not self.one_ball:
            return sub, None

        def balls(k):   # colour list (numbers accepted too) -> (count, set of colours or None)
            v = d.get(k)
            if isinstance(v, list):
                c = {str(x).strip().lower() for x in v} & {'red', 'yellow'}
                return len(c), c
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 2:
                return int(v), None
            return None, None
        n_in, c_in = balls('balls_in_dustpan')
        n_out, c_out = balls('balls_elsewhere')
        if n_in is None or n_out is None:
            return sub, None
        if c_in is not None and c_out is not None:
            c_in = c_in - c_out          # a colour in both lists: not counted as inside
            n_in, total = len(c_in), len(c_in | c_out)
        else:
            total = min(n_in + n_out, 2)
        if n_out >= 1:
            self.last_seen = 'out'
        elif n_in >= 1:
            self.last_seen = 'in'
        cur = self.subtask
        if cur in (None, 'grasp') and sub == 'grasp':   # nothing has been swept yet
            self.start_counts.append(total)
        if self.balls_start() != 1:
            return sub, None
        if cur == 'sweep' or (cur == 'grasp' and sub in ('sweep', 'dump')):   # in sweep, or the grasp is done
            if n_in == 1 and n_out == 0:
                self.in_streak += 1
                if sub == 'dump':
                    self.in_streak = 0
                    return 'dump', None
                if self.in_streak >= 2:
                    self.in_streak = 0
                    return 'dump', 'dump: the one ball seen in the dustpan, none elsewhere, on 2 replies in a row'
                return sub, None
            self.in_streak = 0
            if sub == 'dump':
                if n_out == 0 and self.last_seen == 'in':
                    return 'dump', None   # last seen in the pan: now lifted / tipped / poured, out of view
                return 'sweep', f'stay in sweep: in dustpan {n_in}, elsewhere {n_out}, last seen {self.last_seen}'
        elif cur == 'dump' and sub == 'sweep' and n_out == 0:
            return 'dump', 'stay in dump: no ball seen outside the dustpan'
        return sub, None

    def arms_still(self, now):''')

rep('''        s.update(pan_side=self.pan_side, n_side_mismatch=self.n_side_mismatch, llm_complete_t=self.llm_complete_t)
''', '''        s.update(pan_side=self.pan_side, n_side_mismatch=self.n_side_mismatch, llm_complete_t=self.llm_complete_t,
                 one_ball=self.one_ball, balls_start=self.balls_start(), n_ball_override=self.n_ball_override)
''')

open(F, 'w').write(s)
print('patched')
