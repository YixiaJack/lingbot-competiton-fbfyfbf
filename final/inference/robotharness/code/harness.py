"""Robot harness for the sweeping task: decides which instruction LingBot-VLA gets at each action chunk. Every motion
comes from the VLA; the harness only chooses text (and stops sending chunks once the task is judged complete).

Schedulers (rollout.py --harness):
  none    the full task instruction for the whole episode (pure VLA baseline)
  timed   the canonical skill sequence on a fixed timetable, no perception (open-loop baseline)
  oracle  canonical skill picked from simulator state (upper bound on scheduling; never possible on the real robot)
  v1      LLM plans the decomposition from the skill library, then picks the active skill from the cameras
  v2      decomposition fixed in the agent prompt (5 subtasks); the LLM says which subtask the robot is in and writes
          the VLA instruction for it
  fixed   THE DEFAULT FROM 11/14 ON (real robot): the four subtasks of real_split.py, derived from the official real
          demos (grasp / sweep / dump / home). The LLM only says which subtask the robot is in (and, once, on which
          side the dustpan lies); the harness sends that subtask's fixed instruction, the same string the VLA was
          fine-tuned on. Takes real-robot observations (see view()).

LLM: DeepSeek (deepseek-flash = DeepSeek-V4.1-Flash, the account's model that accepts images; deepseek-v4-pro drops
them). Key: $DEEPSEEK_API_KEY, else the macOS keychain item robotharness.v4.1-flash, else ~/.deepseek_env. Each episode is one
append-only conversation: fixed system prompt first, then one user turn per query (time, grippers, three camera images)
and the model's JSON reply. Every request is the previous request plus one turn, so the provider's prefix cache covers
everything but the newest turn. One query is in flight at a time; it runs while the VLA keeps executing, and its answer
takes effect at the first chunk boundary after (issue time + measured latency) on the episode clock.

Episode clock: executed actions x 0.06 s (RoboTwin records one frame every 15 physics steps of 4 ms; the VLA was trained
at that rate) plus every blocking wait (VLA inference, the first LLM call). Limit 120 s, as in the final.
"""
import base64, io, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from skills import FULL_INSTRUCTION, SKILLS, SKILL_IDS, PROMPT
from prompts import SUBTASKS, SUBTASK_IDS
import real_split as RS

DT = 15 / 250
LIMIT_S = 120.0
DEPS = Path(__file__).resolve().parent.parent / 'deps'


class Clock:
    def __init__(self, limit=LIMIT_S):
        self.actions, self.blocked, self.limit = 0, 0.0, limit

    @property
    def now(self):
        return self.actions * DT + self.blocked

    def tick(self):
        self.actions += 1

    def block(self, seconds):
        self.blocked += float(seconds)

    @property
    def expired(self):
        return self.now >= self.limit


class WallClock:
    """Real-robot episode clock: wall time since start (waits are already inside it). Same interface as Clock."""

    def __init__(self, limit=LIMIT_S):
        self.t0, self.limit = time.monotonic(), limit

    @property
    def now(self):
        return time.monotonic() - self.t0

    def tick(self):
        pass

    def block(self, seconds):
        pass

    @property
    def expired(self):
        return self.now >= self.limit


# ---------------------------------------------------------------------------------------------------------- LLM client
KEYCHAIN_SERVICE = 'robotharness.v4.1-flash'


def _load_key():
    if os.environ.get('DEEPSEEK_API_KEY'):
        return os.environ['DEEPSEEK_API_KEY']
    if shutil.which('security'):   # macOS keychain (saved 11/14: service robotharness.v4.1-flash, account api-key)
        r = subprocess.run(['security', 'find-generic-password', '-s', KEYCHAIN_SERVICE, '-a', 'api-key', '-w'],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    p = Path.home() / '.deepseek_env'   # DEEPSEEK_API_KEY=..., mode 600 (copied from the repo's .env, never committed)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith('DEEPSEEK_API_KEY='):
                return line.split('=', 1)[1].strip()
    raise RuntimeError(f'no key: set DEEPSEEK_API_KEY, add keychain item {KEYCHAIN_SERVICE}, or write ~/.deepseek_env')


def jpeg_part(rgb, quality=85, size=None):
    im = Image.fromarray(rgb)
    if size and im.size != tuple(size):
        im = im.resize(tuple(size))
    buf = io.BytesIO(); im.save(buf, format='JPEG', quality=quality)
    return {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()}}


class Chat:
    """Append-only DeepSeek conversation with JSON replies. ask() is called from one worker thread at a time."""

    def __init__(self, system, model='deepseek-flash', thinking=False):
        if str(DEPS) not in sys.path:
            sys.path.append(str(DEPS))   # appended, so RoboTwin's own packages keep precedence
        from openai import OpenAI
        self.client = OpenAI(api_key=_load_key(), base_url='https://api.deepseek.com', timeout=30.0, max_retries=1)
        self.model, self.thinking = model, thinking
        self.messages = [{'role': 'system', 'content': system}]
        self.calls = []

    def ask(self, content, text_for_log):
        self.messages.append({'role': 'user', 'content': content})
        t0, err, text, usage = time.time(), None, '', None
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=self.messages, response_format={'type': 'json_object'},
                max_tokens=3000 if self.thinking else 500,
                extra_body=None if self.thinking else {'thinking': {'type': 'disabled'}})
            text, usage = r.choices[0].message.content or '', r.usage
        except Exception as e:   # network / API error: the decision is skipped, the conversation stays consistent
            err = f'{type(e).__name__}: {str(e)[:300]}'
        latency = time.time() - t0
        self.messages.append({'role': 'assistant', 'content': text or '{}'})
        parsed = None
        try:
            parsed = json.loads(text.strip().removeprefix('```json').removesuffix('```')) if text else None
        except json.JSONDecodeError as e:
            err = f'bad JSON: {e}'
        rec = {'latency_s': round(latency, 3), 'user': text_for_log, 'reply': text, 'error': err,
               'prompt_tokens': getattr(usage, 'prompt_tokens', None),
               'cache_hit_tokens': getattr(usage, 'prompt_cache_hit_tokens', None),
               'completion_tokens': getattr(usage, 'completion_tokens', None)}
        self.calls.append(rec)
        return parsed, rec


# ------------------------------------------------------------------------------------------------------- agent prompts
COMMON = """You are the task manager of a dual-arm robot at a table. You never move the robot yourself: a \
vision-language-action model (the VLA) makes every motion, following one short English instruction at a time. Your job \
is to watch the cameras and decide what the VLA should be doing now.

Scene: a blue dustpan with its handle on the left side and its open mouth facing the robot, a wooden T-shaped brush on \
the right, a red and a white crumpled paper ball on the table in front of the dustpan, and a round brown basket at the \
front left. The left arm uses the dustpan, the right arm uses the brush.
Each update shows three images: 1) head camera, an overview (the robot's left is the image's left); 2) left wrist \
camera; 3) right wrist camera. Gripper values go from 0 (closed) to 1 (open); a gripper holding a handle reads near 0.

Whole task: "{instruction}"
The task must finish within 120 seconds from the start. The VLA needs several seconds for each step; do not switch away \
from a step that is visibly making progress, and do not stay on a step whose goal is already visible.
"""

V1_SYSTEM = COMMON.format(instruction=FULL_INSTRUCTION) + """
The VLA only understands the instructions in this skill library (id: instruction, then when it is done):
""" + '\n'.join(f'- {k}: "{v["prompt"]}" Done when: {v["done_when"]}' for k, v in SKILLS.items()) + """

How you work:
1. At the start you get the first images. Decide how to split the task in this scene: the ordered list of skill ids \
that completes it (for example one sweep per ball, or one sweep for both balls when they lie side by side).
2. After that, every few seconds you get new images and the skill that is running. Say which skill the VLA should run \
now: keep the running one until its done-condition is visible, then move to the next one in your plan, or go back to \
an earlier skill if something went wrong (a ball rolled out of the dustpan, a tool fell). You may revise the plan.
3. Set task_complete to true only once put_down_and_home is visibly done; the robot then stops.

Reply with one JSON object and nothing else:
{"observation": "<one sentence: what you see that matters>", "active_skill": "<skill id to run now>", \
"plan": ["<skill ids still to do after the active one, in order>"], "task_complete": false}
"""

V2_SYSTEM = COMMON.format(instruction=FULL_INSTRUCTION) + """
The task is always done as these five subtasks, in this order:
""" + '\n'.join(f'{i + 1}. {s["id"]}: {s["name"]}. Goal: {s["goal"]} {s["steps"]} Example instructions: '
                + ' / '.join(f'"{e}"' for e in s['examples']) for i, s in enumerate(SUBTASKS)) + """

Each time you are asked, do two things:
a) Decide which subtask the robot is in now, from the images: the first subtask whose goal is not yet reached. Go back \
to an earlier subtask if its goal is lost (a ball rolled out of the dustpan, a tool fell).
b) Write the one instruction the VLA should follow now, inside that subtask. One short English imperative sentence in \
the style of the examples: name objects by colour and type (dustpan, brush, red / white paper ball, basket), say which \
gripper when grasping (left for the dustpan, right for the brush), add left/right position when it tells two balls \
apart. No numbers, no coordinates, no lists of steps. Keep the same wording while that step is still in progress.
Set task_complete to true only once the last subtask's goal is visibly reached; the robot then stops.

Reply with one JSON object and nothing else:
{"observation": "<one sentence: what you see that matters>", "subtask": "<subtask id>", \
"vla_instruction": "<instruction for the VLA>", "task_complete": false}
"""

# the default harness from 11/14: fixed four-subtask split of the real demos (real_split.py), real-robot cameras/grippers
FIXED_SYSTEM = """You are the task manager of a dual-arm robot at a table. You never move the robot yourself: a \
vision-language-action model (the VLA) makes every motion. The task is always done as the same four subtasks in the \
same order, and for each subtask the VLA gets one fixed instruction. Your only job is to watch the robot and say which \
subtask it is in now.

Scene: a teal table. A cream-coloured dustpan and a brush with a wooden handle and yellow bristles lie on the table, \
one on the left and one on the right (either way round). The paper balls are hard crumpled balls: one red and one \
yellow at most, never more than these two, and a scene may have only one of them. At the start they lie between the \
tools, and every paper ball of the scene is fully visible in the top image. A round white basket stands at the back. The arm on the dustpan's side always uses the dustpan, the \
other arm the brush.
Each update shows three images: 1) top camera, an overview from behind the robot (the robot's left is the image's \
left; the basket is at the far side); 2) left wrist camera; 3) right wrist camera. Gripper readings: 0 means closed \
with nothing in it (the start and the end state); opening to grasp or to let go reads 0.4-0.9; holding the dustpan \
handle reads about 0.32; holding the brush handle reads about 0.23.

Whole task: "{instruction}"
The task must finish within 120 seconds from the start. The VLA needs several seconds for each subtask; do not switch \
away from a subtask that is visibly making progress, and do not stay on one whose goal is already visible.

The four subtasks, in order:
{subtasks}
The grasp goal is reached as soon as both tools are held firmly by their handles. The tools do not have to be lifted \
off the table first: lifting them is part of sweeping.

Each time you are asked:
a) Say which subtask the robot is in now: the first one whose goal is not yet reached. Go back to an earlier subtask \
if its goal is lost (a paper ball is back on the table, a tool fell).
b) Say on which side of the top image the dustpan is (left or right). Decide it from the first images and keep the \
same answer afterwards.
c) Set task_complete to true only once the home goal is visibly reached; the robot then stops.

Reply with one JSON object and nothing else:
{{"observation": "<one sentence: what you see that matters>", "dustpan_side": "<left|right>", \
"subtask": "<grasp|sweep|dump|home>", "task_complete": false}}
""".format(instruction=RS.FULL_INSTRUCTION, subtasks='\n'.join(
    f'{i + 1}. {s["id"]}: {s["name"]}. Goal: {s["goal"]}' for i, s in enumerate(RS.SUBTASKS)))


# team 11/16 (user): grasp judged by sight + gripper motion, not by readings or lifting (see GripperTrack)
for _old, _new in (
        ('Gripper readings: 0 means closed with nothing in it (the start and the end state); opening to grasp or to let '
         'go reads 0.4-0.9; holding the dustpan handle reads about 0.32; holding the brush handle reads about 0.23.',
         'Each update also says what each gripper is doing: not opened yet, opening, open, closing, or closed and steady. '
         'A gripper that closed and stopped above fully closed usually has something between its fingers; a very thin '
         'handle can still close fully; a gripper that stopped about halfway has no firm grip; an open or opening gripper '
         'holds nothing. The gripper words cannot show which tool is held: use the images for that.'),
        (' (gripper readings: about 0.32 on the dustpan arm, about 0.23 on the brush arm)', ''),
        ('(they read 0 again)', '(they are fully closed again)'),
        ('the robot then stops.\n\nReply with one JSON object and nothing else:',
         'the robot then stops.\n'
         'd) For each gripper, say what is between its fingers right now, judged only from the images (that arm\'s wrist '
         'camera first, then the top camera): "dustpan" (the dustpan handle), "brush" (the brush handle), "nothing", or '
         '"unclear". Do not decide this from the gripper words, and a tool does not have to be lifted to be held.\n\n'
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


GRASPNET_FILE = Path(os.environ.get('ROBOTHARNESS_GRASPNET',
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
def grippers(obs):
    return view(obs)[1]


def view(obs):
    """(three RGB images [overview, left wrist, right wrist], (left gripper, right gripper)) from either
    a RoboTwin obs dict, or a real-robot obs {'images': {'top', 'left', 'right'}: HxWx3 uint8 RGB,
    'grippers': (left, right), 'arm': 12 joint positions (optional; Fixed uses it to decide when to stop)} with
    grippers = observation.state.effector.position and arm = observation.state.arm.position."""
    if 'images' in obs:
        im = obs['images']
        return [im['top'], im['left'], im['right']], (float(obs['grippers'][0]), float(obs['grippers'][1]))
    q = np.asarray(obs['joint_action']['vector'])
    return [obs['observation'][c]['rgb'] for c in ('head_camera', 'left_camera', 'right_camera')], (float(q[6]), float(q[13]))


class Scheduler:
    name = 'base'

    def __init__(self):
        self.done, self.events, self.current, self.since = False, [], None, 0.0

    def set(self, value, now, **info):
        if value != self.current:
            self.current, self.since = value, now
            self.events.append({'t': round(now, 2), 'set': value, **info})

    def start(self, obs, clock, env=None):
        pass

    def on_chunk(self, obs, clock, env=None):
        pass

    def prompt(self):
        return self.current

    def close(self):
        pass

    def summary(self):
        return {'events': self.events}


class NoHarness(Scheduler):
    name = 'none'

    def start(self, obs, clock, env=None):
        self.set(FULL_INSTRUCTION, 0.0)


class Timed(Scheduler):
    """Canonical skills on a fixed timetable (durations: expert medians x 1.3, from timing.json when present)."""
    name = 'timed'
    ORDER = ['grasp_dustpan', 'grasp_brush', 'sweep_red', 'sweep_white', 'carry_to_basket', 'pour', 'put_down_and_home']

    def __init__(self, durations):
        super().__init__(); self.d = durations

    def start(self, obs, clock, env=None):
        self.on_chunk(obs, clock, env)

    def on_chunk(self, obs, clock, env=None):
        t, acc = clock.now, 0.0
        for k in self.ORDER:
            acc += self.d[k]
            if t < acc:
                self.set(k, t); return
        self.done = True

    def prompt(self):
        return PROMPT[self.current]


class Oracle(Scheduler):
    """Canonical skill from simulator state (privileged; an upper bound for the scheduling part)."""
    name = 'oracle'

    def start(self, obs, clock, env=None):
        self.poured = False
        self.on_chunk(obs, clock, env)

    def on_chunk(self, obs, clock, env=None):
        import sweep_scene as S
        f = env.progress()
        if any(f['in_basket'].values()):
            self.poured = True
        if self.poured or (f['pan_lifted'] and not f['all_in_pan'] and not f['pan_over_basket']):
            if f['tools_down'] and f['grippers_open'] and f['arms_home']:
                self.done = True; return
            k = 'pour' if (f['pan_lifted'] and any(f['in_pan'].values()) and f['pan_over_basket']) else 'put_down_and_home'
        elif f['pan_lifted'] and f['all_in_pan']:
            k = 'pour' if f['pan_over_basket'] else 'carry_to_basket'
        elif not f['pan_held']['left']:
            k = 'grasp_dustpan'
        elif not f['brush_held']['right']:
            k = 'grasp_brush'
        elif not f['all_in_pan']:
            out = [c for c in S.BALLS if not f['in_pan'][c]]
            if len(out) == 2:
                xs = [env.ball_local(c)[0] for c in out]
                k = 'sweep_both' if abs(xs[0] - xs[1]) < S.HEAD_HW + S.BALL_R else 'sweep_red'
            else:
                k = f'sweep_{out[0]}'
        else:
            k = 'carry_to_basket'
        self.set(k, clock.now)

    def prompt(self):
        return PROMPT[self.current]


class LLMScheduler(Scheduler):
    """Shared async loop of v1 and v2: blocking first call, then one background query at a time."""

    def __init__(self, system, model='deepseek-flash', thinking=False, img_size=None):
        super().__init__()
        self.chat = Chat(system, model=model, thinking=thinking)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future, self.issue_t, self.n_invalid, self.img_size = None, 0.0, 0, img_size

    def user_turn(self, obs, clock, first):
        cams, (lg, rg) = view(obs)
        if first:
            head = f'Start of the task. Time 0.0 s of 120 s.'
        else:
            head = f'Time {clock.now:.1f} s of 120 s. Running now: {self.running_text()} (for {clock.now - self.since:.1f} s).'
        text = f'{head} Grippers: left {lg:.2f}, right {rg:.2f}.'
        return [{'type': 'text', 'text': text}] + [jpeg_part(c, size=self.img_size) for c in cams], text

    def start(self, obs, clock, env=None):
        content, text = self.user_turn(obs, clock, first=True)
        parsed, rec = self.chat.ask(content, text)
        clock.block(rec['latency_s'])   # the robot waits for the first decision
        if not self.apply(parsed, clock.now, rec):
            self.fallback_start(clock.now)

    def on_chunk(self, obs, clock, env=None):
        if self.future is not None and self.future.done():
            parsed, rec = self.future.result()
            if clock.now >= self.issue_t + rec['latency_s']:
                self.apply(parsed, clock.now, rec); self.future = None
        if self.future is None and not self.done:
            content, text = self.user_turn(obs, clock, first=False)
            self.issue_t = clock.now
            self.future = self.pool.submit(self.chat.ask, content, text)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def summary(self):
        lat = [c['latency_s'] for c in self.chat.calls]
        pt = [c['prompt_tokens'] or 0 for c in self.chat.calls]; ch = [c['cache_hit_tokens'] or 0 for c in self.chat.calls]
        return {'events': self.events, 'calls': self.chat.calls, 'n_calls': len(lat), 'n_invalid': self.n_invalid,
                'latency_mean_s': float(np.mean(lat)) if lat else None, 'latency_max_s': float(np.max(lat)) if lat else None,
                'prompt_tokens': int(sum(pt)), 'cache_hit_tokens': int(sum(ch)),
                'cache_hit_rate': (sum(ch) / sum(pt)) if sum(pt) else None}


class V1(LLMScheduler):
    name = 'v1'

    def __init__(self, **kw):
        super().__init__(V1_SYSTEM, **kw)
        self.plan = []

    def running_text(self):
        return self.current

    def apply(self, d, now, rec):
        if not isinstance(d, dict) or d.get('active_skill') not in SKILL_IDS:
            self.n_invalid += 1; return False
        self.plan = [k for k in d.get('plan', []) if k in SKILL_IDS]
        self.set(d['active_skill'], now, plan=self.plan, observation=str(d.get('observation', ''))[:300])
        if d.get('task_complete') is True:
            self.done = True; self.events.append({'t': round(now, 2), 'task_complete': True})
        return True

    def fallback_start(self, now):
        self.set('grasp_dustpan', now, fallback=True)

    def prompt(self):
        return PROMPT[self.current]


class V2(LLMScheduler):
    name = 'v2'

    def __init__(self, **kw):
        super().__init__(V2_SYSTEM, **kw)
        self.subtask = None

    def running_text(self):
        return f'subtask {self.subtask}, instruction "{self.current}"'

    def apply(self, d, now, rec):
        instr = str(d.get('vla_instruction', '')).strip() if isinstance(d, dict) else ''
        if not isinstance(d, dict) or d.get('subtask') not in SUBTASK_IDS or not instr or len(instr) > 200:
            self.n_invalid += 1; return False
        self.subtask = d['subtask']
        self.set(instr, now, subtask=self.subtask, observation=str(d.get('observation', ''))[:300])
        if d.get('task_complete') is True:
            self.done = True; self.events.append({'t': round(now, 2), 'task_complete': True})
        return True

    def fallback_start(self, now):
        self.subtask = 'grasp'; self.set(PROMPT['grasp_dustpan'], now, subtask='grasp', fallback=True)


class Fixed(LLMScheduler):
    """The default harness: the four fixed subtasks of real_split.py. The LLM answers which subtask (and, at the start,
    the dustpan's side); the harness sends real_split.instruction(subtask, side), the exact training string.
    The side is locked after the first valid answer. Once both tools are held, the gripper readings tell the roles apart
    without error on the 216 demos (dustpan 0.318-0.338, brush 0.218-0.249), so they override a wrong first answer.

    Stopping: "task_complete" from the LLM does not stop the robot at once. In the offline test (judge_real.py, 24 demos)
    the LLM said complete 0.5-4.8 s (median 2.5 s) before the arms were actually home in every demo, so stopping then
    would freeze the arms mid-way and lose the homing points. After it, the harness keeps sending the home instruction
    and stops once both arms have been still (every joint within FINISH_TOL rad) for FINISH_STILL_S and both grippers
    read ~0. On the 216 demos a 2.0 s window never fires before the arms are home (1.5 s fires early once); 2.5 s keeps
    a margin. Without joint readings in obs, it stops FINISH_GRACE_S after the LLM's call."""
    name = 'fixed'
    FINISH_STILL_S, FINISH_TOL, FINISH_GRACE_S = 2.5, 0.02, 6.0

    GRASP_SEEN_N = 2   # team 11/16: replies in a row with both arms seen holding + grippers closed and steady

    def __init__(self, img_size=(320, 240), **kw):
        super().__init__(FIXED_SYSTEM, img_size=img_size, **kw)
        self.subtask, self.pan_side, self.n_side_mismatch = None, None, 0
        self.llm_complete_t, self.arm_hist = None, []
        self.tracks = {'left': GripperTrack(), 'right': GripperTrack()}
        self.grip_q, self.n_grasp_seen, self.n_grasp_override = None, 0, 0
        self.net_q, self.n_sweep_wait = None, 0

    def track(self, obs, now):
        lg, rg = view(obs)[1]
        self.tracks['left'].update(now, lg); self.tracks['right'].update(now, rg)

    def user_turn(self, obs, clock, first):
        cams, (lg, rg) = view(obs)
        self.track(obs, clock.now)
        self.grip_q = {a: self.tracks[a].snapshot(clock.now) for a in ('left', 'right')}
        p = GraspNet.probs([cams[1], cams[2]])
        self.net_q = None if p is None else {'left': p[0], 'right': p[1]}
        if first:
            head = 'Start of the task. Time 0.0 s of 120 s.'
        else:
            head = f'Time {clock.now:.1f} s of 120 s. Running now: {self.running_text()} (for {clock.now - self.since:.1f} s).'
        text = (f"{head} Grippers: left {GripperTrack.words(self.grip_q['left'])}; "
                f"right {GripperTrack.words(self.grip_q['right'])}.")
        if self.net_q:
            hw = lambda a: 'holds a handle' if self.net_q[a] >= 0.5 else 'holds nothing'
            text += f" Grasp check from the wrist cameras: left {hw('left')}, right {hw('right')}."
        log = f'{text} [raw left {lg:.3f}, right {rg:.3f}]'   # the viewer and log readers parse "left <number>"
        if self.net_q:
            log += f" [net left {self.net_q['left']:.2f}, right {self.net_q['right']:.2f}]"
        return [{'type': 'text', 'text': text}] + [jpeg_part(c, size=self.img_size) for c in cams], log

    WAIT_MAX = 4   # LLM sweep answers held back in a row while the network says an arm does not hold

    def arm_held(self, a):
        return bool(self.net_q) and self.net_q[a] >= 0.5 and self.grip_q[a]['state'] == 'closed'

    def grasp_rule(self, d):
        """True when each arm is seen holding its own tool and its gripper is closed and steady (any reading).
        With the grasp network available, 'seen holding' comes from the network instead of the LLM's fields."""
        if self.net_q and self.grip_q:
            return all(self.arm_held(a) for a in ('left', 'right'))
        if self.pan_side not in ('left', 'right') or not self.grip_q:
            return False
        want = {self.pan_side: 'dustpan', RS.other(self.pan_side): 'brush'}
        return all(str(d.get(f'{a}_holds', '')).strip().lower() == want[a] and self.grip_q[a]['state'] == 'closed'
                   for a in ('left', 'right'))

    def running_text(self):
        return f'subtask {self.subtask}'

    def apply(self, d, now, rec):
        if not isinstance(d, dict) or d.get('subtask') not in RS.SUBTASK_IDS:
            self.n_invalid += 1; return False
        side = d.get('dustpan_side')
        if self.pan_side is None:
            if side not in ('left', 'right'):
                self.n_invalid += 1; return False
            self.pan_side = side
        elif side != self.pan_side:
            self.n_side_mismatch += 1
        sub = d['subtask']
        if self.subtask == 'grasp' and sub == 'sweep' and self.grip_q:
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
            self.n_sweep_wait = 0
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
        if d.get('task_complete') is True and self.subtask == 'home':
            if self.llm_complete_t is None:
                self.llm_complete_t = now; self.events.append({'t': round(now, 2), 'llm_task_complete': True})
        elif self.llm_complete_t is not None:   # the LLM takes it back (something went wrong): keep working
            self.llm_complete_t = None; self.events.append({'t': round(now, 2), 'llm_task_complete': False})
        return True

    def arms_still(self, now):
        h = [(t, q) for t, q in self.arm_hist if t >= now - self.FINISH_STILL_S - 1.0]
        if not h or h[0][0] > now - self.FINISH_STILL_S:
            return False   # not enough history yet
        last = h[-1][1]
        return all(np.abs(q - last).max() < self.FINISH_TOL for t, q in h if t >= now - self.FINISH_STILL_S)

    def start(self, obs, clock, env=None):
        for attempt in range(2):   # the side is needed from the first chunk on: ask once more before falling back
            content, text = self.user_turn(obs, clock, first=True)
            parsed, rec = self.chat.ask(content, text)
            clock.block(rec['latency_s'])
            if self.apply(parsed, clock.now, rec):
                return
        self.fallback_start(clock.now)

    def on_chunk(self, obs, clock, env=None):
        self.track(obs, clock.now)   # team 11/16: gripper motion history for GripperTrack
        # Gripper evidence must repeat on two consecutive chunks and is ignored in 'home': on the 216 demos the
        # readings point to the wrong side in 13 frames, all while both grippers close together after the release.
        ev = RS.roles_from_grippers(*view(obs)[1])
        prev, self._ev_prev = getattr(self, '_ev_prev', None), ev
        if ev and ev == prev and self.subtask != 'home' and self.pan_side and ev != self.pan_side:
            self.events.append({'t': round(clock.now, 2), 'side_from_grippers': ev, 'was': self.pan_side})
            self.pan_side = ev
            self.set(RS.instruction(self.subtask or 'grasp', ev), clock.now, subtask=self.subtask, side=ev)
        now = clock.now
        if obs.get('arm') is not None:
            self.arm_hist.append((now, np.asarray(obs['arm'], float)))
            self.arm_hist = [(t, q) for t, q in self.arm_hist if t >= now - self.FINISH_STILL_S - 1.0]
        if self.llm_complete_t is not None:
            grippers_shut = max(view(obs)[1]) < 0.05
            if obs.get('arm') is not None:
                stop = grippers_shut and self.arms_still(now)
            else:
                stop = now - self.llm_complete_t >= self.FINISH_GRACE_S
            if stop:
                self.done = True; self.events.append({'t': round(now, 2), 'stop': True}); return
        super().on_chunk(obs, clock, env)

    def fallback_start(self, now):
        self.subtask, self.pan_side = 'grasp', self.pan_side or 'right'
        self.set(RS.instruction('grasp', self.pan_side), now, subtask='grasp', side=self.pan_side, fallback=True)

    def summary(self):
        s = super().summary()
        s.update(pan_side=self.pan_side, n_side_mismatch=self.n_side_mismatch, llm_complete_t=self.llm_complete_t,
                 n_grasp_override=self.n_grasp_override)
        return s


def make_scheduler(kind='fixed', timing=None, model='deepseek-flash', thinking=False):
    if kind == 'fixed': return Fixed(model=model, thinking=thinking)
    if kind == 'none': return NoHarness()
    if kind == 'timed': return Timed(timing)
    if kind == 'oracle': return Oracle()
    if kind == 'v1': return V1(model=model, thinking=thinking)
    if kind == 'v2': return V2(model=model, thinking=thinking)
    raise ValueError(kind)


def real_robot_loop(get_obs, run_chunk, model='deepseek-flash', limit=LIMIT_S):
    """How the real robot uses the harness. get_obs() -> {'images': {'top','left','right'}, 'grippers': (l, r),
    'arm': 12 joints}; run_chunk(instruction) runs one VLA action chunk with that instruction. Returns the summary."""
    sched, clock = make_scheduler('fixed', model=model), WallClock(limit)
    obs = get_obs(); sched.start(obs, clock)
    try:
        while not sched.done and not clock.expired:
            run_chunk(sched.prompt())
            obs = get_obs(); sched.on_chunk(obs, clock)
    finally:
        sched.close()
    return sched.summary()


# ------------------------------------------------------------------------------------------------------------- rubric
class Rubric:
    """The official task-3 rubric, judged from simulator state sampled every few actions.
    15 one gripper holds the dustpan stably (>= 1 s)      15 the other gripper holds the brush stably (>= 1 s)
    30 both balls inside the dustpan                         10 dustpan with both balls held over the basket
    20 at the end both balls in the basket (none outside)    10 at the end tools down, grippers open, arms home
    Success (SR) = the first five within the time limit (homing excluded, as in the rules).
    Episode score = 0.7 x process points + 0.3 x 100 x success."""
    POINTS = {'grasp_dustpan': 15, 'grasp_brush': 15, 'sweep': 30, 'carry': 10, 'pour': 20, 'home': 10}

    def __init__(self, hold_s=1.0):
        self.t, self.run, self.pan_arm, self.hold_s, self.last_t = {}, {'pan': 0.0, 'brush': 0.0}, None, hold_s, None

    def update(self, f, now):
        dt = 0.0 if self.last_t is None else now - self.last_t
        self.last_t = now
        pan_arms = [a for a in ('left', 'right') if f['pan_held'][a]]
        self.run['pan'] = self.run['pan'] + dt if pan_arms else 0.0
        if 'grasp_dustpan' not in self.t and pan_arms and self.run['pan'] >= self.hold_s:
            self.t['grasp_dustpan'], self.pan_arm = now, pan_arms[0]
        other = [a for a in ('left', 'right') if f['brush_held'][a] and a != self.pan_arm]
        self.run['brush'] = self.run['brush'] + dt if other else 0.0
        if 'grasp_brush' not in self.t and other and self.run['brush'] >= self.hold_s:
            self.t['grasp_brush'] = now
        if 'sweep' not in self.t and f['all_in_pan']:
            self.t['sweep'] = now
        if 'carry' not in self.t and f['all_in_pan'] and f['pan_over_basket']:
            self.t['carry'] = now
        if 'pour' not in self.t and f['all_in_basket']:
            self.t['pour_first_seen'] = now

    def final(self, f, now, limit=LIMIT_S):
        got = {k: k in self.t and self.t[k] <= limit for k in ('grasp_dustpan', 'grasp_brush', 'sweep', 'carry')}
        got['pour'] = bool(f['all_in_basket']) and self.t.get('pour_first_seen', 1e9) <= limit
        # 'put the tools down and return home': only counts after both tools were actually picked up
        got['home'] = bool(got['grasp_dustpan'] and got['grasp_brush'] and f['tools_down'] and f['grippers_open'] and f['arms_home'])
        process = sum(self.POINTS[k] for k, v in got.items() if v)
        success = all(got[k] for k in ('grasp_dustpan', 'grasp_brush', 'sweep', 'carry', 'pour'))
        return {'stages': got, 'stage_times': {k: round(v, 2) for k, v in self.t.items()}, 'process_points': process,
                'success': success, 'score': 0.7 * process + 0.3 * 100 * success}
