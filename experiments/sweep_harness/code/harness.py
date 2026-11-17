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
one on the left and one on the right (either way round). One or two crumpled paper balls (yellow and/or red) lie \
between them. A round white basket stands at the back. The arm on the dustpan's side always uses the dustpan, the \
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


PROPRIO_SYSTEM = """You watch a dual-arm robot sweeping paper balls at a table. You never move the robot and you do \
not decide what it does next; you only report what you can see, as counts.

Scene: a teal table. A cream-coloured dustpan and a brush with a wooden handle and yellow bristles lie on the table, \
one on the left and one on the right (either way round). One or two crumpled paper balls (yellow and/or red) lie \
between them. A round white basket stands at the back.
Each update shows three images: 1) top camera, an overview from behind the robot (the robot's left is the image's \
left); 2) left wrist camera; 3) right wrist camera. The wrist camera on the dustpan's side usually looks into the \
dustpan's bowl. Gripper readings: 0 = closed and empty, 0.4-0.9 = open, about 0.32 = holding the dustpan handle, \
about 0.23 = holding the brush handle.

Each time, count the paper balls:
- balls_in_dustpan: balls you can SEE lying inside the dustpan's bowl, in any of the three images.
- balls_on_table: balls you can SEE on the table outside the dustpan (also a ball on the brush bristles or at the \
dustpan's lip, not yet in the bowl).
- balls_hidden: balls you cannot see right now (covered by the brush, an arm, the dustpan rim). A ball you cannot see \
is hidden, never "in the dustpan" and never "gone".
- balls_total: how many paper balls there are in this run. Count them in the first images, when nothing covers them, \
and keep the same number afterwards.
Also say on which side of the top image the dustpan is, from the first images, and keep it.

Reply with one JSON object and nothing else:
{"observation": "<one sentence: where each ball is>", "dustpan_side": "<left|right>", "balls_total": <int>, \
"balls_in_dustpan": <int>, "balls_on_table": <int>, "balls_hidden": <int>}
"""


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

    def __init__(self, img_size=(320, 240), **kw):
        super().__init__(FIXED_SYSTEM, img_size=img_size, **kw)
        self.subtask, self.pan_side, self.n_side_mismatch = None, None, 0
        self.llm_complete_t, self.arm_hist = None, []

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
        self.subtask = d['subtask']
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
        s.update(pan_side=self.pan_side, n_side_mismatch=self.n_side_mismatch, llm_complete_t=self.llm_complete_t)
        return s


class Proprio(Fixed):
    """Same four subtasks and instructions as Fixed, but the phase comes from the robot's own readings (gripper
    openings and arm joints) plus time limits, never from whether paper balls are visible.

    Why (2026-11-14 evening real-robot eval, 10 runs of task 3 with Fixed): the brush and the dustpan hide the balls,
    so the LLM's "no ball on the table" / "a ball is back on the table" calls flipped. 13 dumps were started and 11 were
    called off again, 8 of them within 3.3 s; no run ever reached home. The demos never sweep again once the dustpan
    is lifted (216/216), so going back to sweep with the dustpan in the air is outside the training data as well.

    The LLM is asked once at the start for the dustpan side (the scene is still uncluttered then; 24/24 offline,
    10/10 live). Later LLM replies are logged as observations only and never change the subtask.

    Rules (thresholds from calib_proprio.py on the 216 official demos; grippers in dataset units = live metres x 10):
      grasp -> sweep  both grippers in the holding band (HOLD_LO..HOLD_HI) on HOLD_N consecutive observations. The band
                      is apart from closed-empty (0, the start) and open (> 0.38); two observations 0.8 s apart skip
                      the moment a closing gripper passes through it. Not "seen open first": at 0.8 s per observation
                      a short opening is missed (15 of 216 demos stuck in grasp that way). Demos: 0.7-1.5 s after the
                      label, 0 missed.
      sweep -> dump   no reliable signal for "all balls in the pan" without seeing them. Leaves sweep once the brush
                      arm has been still for BRUSH_IDLE_S, after at least SWEEP_MIN_S. No time limit on sweeping
                      (11/15: in the first Proprio run the 14 s limit sent the robot to dump with both balls
                      still at the dustpan's edge); only the whole-task HOME_BY_S below still applies.
      sweep -> grasp  the only way back: the brush gripper reads empty (< RELEASED) or open (> OPEN) on DROP_N
                      consecutive observations (the brush fell, as in 2 of the 10 eval runs).
      dump -> home    the dustpan arm's joint displacement from its pose at the dump start has risen above DUMP_D_MIN
                      rad and fallen back by DUMP_RETURN of its peak (it is on its way back from the pour). At chunk
                      rate on the demos (864 trials): median 0.9 s after the label (the label is ~1 s after the balls
                      drop), 1.5% more than 0.5 s before it, 1.6% never fire -> DUMP_MAX_S.
      any -> home     at HOME_BY_S of the 120 s, so the tools are put down and the homing points are not lost.
      home -> stop    both arms still for FINISH_STILL_S and both grippers released, at least HOME_MIN_S into home.
    """
    name = 'proprio'
    HOLD_LO, HOLD_HI, OPEN, RELEASED = 0.15, 0.38, 0.38, 0.05
    HOLD_N, DROP_N = 2, 2
    SWEEP_MIN_S, SWEEP_MAX_S, BRUSH_IDLE_S, IDLE_TOL = 3.0, None, 2.5, 0.02   # SWEEP_MAX_S None = no limit
    ALL_IN_N = 2   # consecutive LLM replies with every ball seen in the dustpan and none outside
    DUMP_D_MIN, DUMP_RETURN, DUMP_MAX_S = 1.0, 0.2, 14.0
    HOME_BY_S, HOME_MIN_S = 100.0, 4.0

    def __init__(self, observe=True, img_size=(320, 240), **kw):
        LLMScheduler.__init__(self, PROPRIO_SYSTEM, img_size=img_size, **kw)
        self.subtask, self.pan_side, self.n_side_mismatch = None, None, 0
        self.llm_complete_t, self.arm_hist = None, []
        self.observe, self.observations = observe, []
        self.n_total, self.n_reply, self.n_all_in, self.last_counts = None, 0, 0, None
        self.opened = [False, False]
        self.n_hold = self.n_drop = 0
        self.phase_t = 0.0
        self.q_dump0, self.d_peak = None, 0.0

    @staticmethod
    def _count(d, k):
        try:
            v = int(d.get(k))
            return v if 0 <= v <= 4 else None
        except (TypeError, ValueError):
            return None

    def apply(self, d, now, rec):
        if not isinstance(d, dict):
            self.n_invalid += 1; return False
        c = {k: self._count(d, k) for k in ('balls_total', 'balls_in_dustpan', 'balls_on_table', 'balls_hidden')}
        if self.subtask is None:   # the first answer: the dustpan side and the number of balls; the robot starts with grasp
            side = d.get('dustpan_side')
            if side not in ('left', 'right') or not c['balls_total']:
                self.n_invalid += 1; return False
            self.pan_side, self.n_total = side, c['balls_total']
            self.subtask = 'grasp'
            self.set(RS.instruction('grasp', side), now, subtask='grasp', side=side, balls_total=self.n_total,
                     observation=str(d.get('observation', ''))[:300])
            return True
        if self.subtask == 'grasp' and c['balls_total'] and c['balls_total'] > (self.n_total or 0):
            self.n_total = c['balls_total']   # a ball first seen once the arms move away: the count only goes up
        # sweep -> dump needs positive evidence: every ball SEEN in the bowl and none seen outside, on two replies
        # in a row. A hidden ball keeps the robot sweeping (11/14 eval: "no ball visible" was read as "swept").
        all_in = (c['balls_in_dustpan'] is not None and self.n_total is not None and c['balls_in_dustpan'] >= self.n_total
                  and c['balls_on_table'] == 0)
        if self.subtask == 'sweep':
            self.n_all_in = self.n_all_in + 1 if all_in else 0
        self.n_reply += 1; self.last_counts = c
        self.observations.append({'t': round(now, 2), **c, 'all_in': all_in, 'observation': str(d.get('observation', ''))[:200]})
        return True

    def fallback_start(self, now):
        super().fallback_start(now)
        self.n_total = self.n_total or 2   # the official layout has one red and one white/yellow ball

    def start(self, obs, clock, env=None):
        super().start(obs, clock, env)
        self.phase_t = clock.now

    def go(self, subtask, now, why):
        self.subtask, self.phase_t = subtask, now
        self.n_hold = self.n_drop = self.n_all_in = 0
        self.set(RS.instruction(subtask, self.pan_side), now, subtask=subtask, side=self.pan_side, why=why)

    def still_for(self, idx, window, now):
        """True when the joints idx have moved less than IDLE_TOL over the last `window` seconds."""
        h = [(t, q[idx]) for t, q in self.arm_hist if t >= now - window - 1.0]
        if not h or h[0][0] > now - window:
            return False
        last = h[-1][1]
        return all(np.abs(q - last).max() < self.IDLE_TOL for t, q in h if t >= now - window)

    def on_chunk(self, obs, clock, env=None):
        now = clock.now
        lg, rg = view(obs)[1]
        g = (lg, rg)
        for a in (0, 1):
            self.opened[a] = self.opened[a] or g[a] > self.OPEN
        # side from the grippers (Fixed's rule, now that the readings are in dataset units)
        ev = RS.roles_from_grippers(lg, rg)
        prev, self._ev_prev = getattr(self, '_ev_prev', None), ev
        if ev and ev == prev and self.subtask != 'home' and ev != self.pan_side:
            self.events.append({'t': round(now, 2), 'side_from_grippers': ev, 'was': self.pan_side})
            self.pan_side = ev
            self.set(RS.instruction(self.subtask or 'grasp', ev), now, subtask=self.subtask, side=ev)
        arm = obs.get('arm')
        if arm is not None:
            self.arm_hist.append((now, np.asarray(arm, float)))
            self.arm_hist = [(t, q) for t, q in self.arm_hist if t >= now - max(self.FINISH_STILL_S, self.BRUSH_IDLE_S) - 1.0]
        pan = 0 if self.pan_side == 'left' else 1
        in_phase = now - self.phase_t
        s = self.subtask
        if s != 'home' and now >= self.HOME_BY_S:
            self.go('home', now, f'time {now:.0f} s >= {self.HOME_BY_S:.0f} s')
        elif s == 'grasp':
            held = all(self.HOLD_LO < g[a] < self.HOLD_HI for a in (0, 1))
            self.n_hold = self.n_hold + 1 if held else 0
            if self.n_hold >= self.HOLD_N:
                self.go('sweep', now, f'both tools held ({lg:.2f}, {rg:.2f})')
        elif s == 'sweep':
            b = g[1 - pan]
            dropped = b < self.RELEASED or b > self.OPEN
            self.n_drop = self.n_drop + 1 if dropped else 0
            if self.n_drop >= self.DROP_N:
                self.go('grasp', now, f'brush gripper reads {b:.2f}')
            elif self.SWEEP_MAX_S is not None and in_phase >= self.SWEEP_MAX_S:
                self.go('dump', now, f'sweep for {in_phase:.1f} s')
            elif in_phase >= self.SWEEP_MIN_S and self.n_all_in >= self.ALL_IN_N:
                self.go('dump', now, f'all {self.n_total} balls seen in the dustpan on {self.n_all_in} replies in a row '
                                     f'after {in_phase:.1f} s of sweeping')
            if self.subtask == 'dump':
                self.q_dump0 = None if arm is None else np.asarray(arm, float)[6 * pan: 6 * pan + 6]
                self.d_peak = 0.0
        elif s == 'dump':
            fire = False
            if arm is not None and self.q_dump0 is not None:
                dq = float(np.linalg.norm(np.asarray(arm, float)[6 * pan: 6 * pan + 6] - self.q_dump0))
                self.d_peak = max(self.d_peak, dq)
                fire = self.d_peak >= self.DUMP_D_MIN and dq <= self.d_peak * (1 - self.DUMP_RETURN)
            if fire:
                self.go('home', now, f'dustpan arm back from the pour (peak {self.d_peak:.2f} rad)')
            elif in_phase >= self.DUMP_MAX_S:
                self.go('home', now, f'dump for {in_phase:.1f} s')
        elif s == 'home':
            released = max(g) < self.RELEASED
            if arm is not None:
                stop = released and in_phase >= self.HOME_MIN_S and self.arms_still(now)
            else:
                stop = in_phase >= self.HOME_MIN_S + self.FINISH_GRACE_S
            if stop:
                self.done = True; self.events.append({'t': round(now, 2), 'stop': True}); return
        LLMScheduler.on_chunk(self, obs, clock, env)   # the ball counts come from these background replies

    def summary(self):
        s = super().summary()
        s.update(observations=self.observations[-80:], balls_total=self.n_total)
        return s


PROPRIO2_SYSTEM = """You watch a dual-arm robot sweeping paper balls at a table. You never move the robot and you do \
not decide what it does next; you only report what you can see.

Scene: a teal table. A cream-coloured dustpan and a brush with a wooden handle and yellow bristles lie on the table, \
one on the left and one on the right (either way round). One or two crumpled paper balls (yellow and/or red) lie \
between them. A round white mesh basket stands at the back.
Each update shows three images: 1) top camera, an overview from behind the robot (the robot's left is the image's \
left); 2) left wrist camera; 3) right wrist camera. The wrist camera on the dustpan's side usually looks into the \
dustpan's bowl; use it to count the balls in the bowl. Gripper readings: 0 = closed, 0.4-0.9 = open. A gripper \
holding the dustpan handle reads about 0.3; a gripper pinching the thin rim of the dustpan, or the thin brush handle, \
can read close to 0 and still hold it, so judge holding from the images, not from the number.

Report, every time:
- balls_total: how many paper balls there are in this run. Count them in the first images, when nothing covers them, \
and keep the same number afterwards.
- balls_in_dustpan: balls you can SEE lying inside the dustpan's bowl, past its lip, in any of the three images. A \
ball at or on the lip, or partly hidden behind the brush or a gripper at the lip, is NOT in the dustpan yet: count \
it on the table. If you are unsure whether a ball is in the bowl, it is not. While the dustpan is tipped over the \
basket, a ball still lying on the dustpan (even at its lip) is in the dustpan until it has fallen off.
- balls_on_table: balls you can SEE on the table outside the dustpan (also a ball on the brush bristles, at the \
dustpan's lip but not yet in the bowl, or one that fell next to the basket).
- balls_in_basket: balls you can SEE inside the basket (through its mesh or from above).
- balls_hidden: balls you cannot see right now. A ball you cannot see is hidden, never "in the dustpan" and never \
"in the basket" (a wrist camera blocked by the dustpan shows nothing about its contents).
- The four counts (in the dustpan, on the table, in the basket, hidden) must add up to balls_total.
- dustpan_held: true only if a gripper is clearly closed on the dustpan (handle or rim), so the dustpan moves with \
that arm; false if no gripper is on it or a gripper is only touching or hovering next to it.
- brush_held: the same for the brush.
- dustpan_state: exactly one of
  "on_table"   the dustpan rests on the table, bowl level (also while being pushed along the table);
  "lifting"    the dustpan is off the table and level or nearly level, not yet above the basket;
  "over_basket" the dustpan is above or at the basket's rim, not yet tipped;
  "tipped"     the dustpan is tilted so that its contents slide or fall out (pouring);
  "set_down"   after a pour, the dustpan has been put back on the table.
- dustpan_side: on which side of the top image the dustpan started, from the first images; keep it.

Reply with one JSON object and nothing else:
{"observation": "<one sentence: where the dustpan, the brush and each ball are>", "dustpan_side": "<left|right>", \
"balls_total": <int>, "balls_in_dustpan": <int>, "balls_on_table": <int>, "balls_in_basket": <int>, "balls_hidden": <int>, \
"dustpan_held": <true|false>, "brush_held": <true|false>, "dustpan_state": "<on_table|lifting|over_basket|tipped|set_down>"}
"""


class Proprio2(Proprio):
    """Proprio plus visual evidence for the decisions the robot's readings cannot make (11/15, from the 14 Proprio runs
    of 03:05-03:58 on the 5080; 8 of them matched to our recorded sessions eps 172-179 by gripper traces):

      grasp stuck   4/14 runs never left grasp: the dustpan gripper read 0.01-0.12 (rim or edge grasp) while the robot
                    went on to sweep and pour. Added: both tools seen held on GRASP_SEEN_N replies with neither gripper
                    open, or a ball seen in the dustpan on GRASP_SEEN_N replies (the robot is already sweeping).
      sweep lag     5/8 recorded runs: the VLA lifted the dustpan 3-4 s before the harness said dump, so "sweep" was
                    sent with the dustpan in the air. Added: all balls in on the latest reply and the dustpan seen
                    lifting / over the basket / tipped -> dump at once; the dustpan seen over the basket or tipped on
                    LIFT_SEEN_N replies -> dump regardless of the counts (the VLA is pouring).
                    The dustpan arm's joints move as much during sweeping (0.44-1.34 rad) as at the lift, so the lift
                    comes from the images, not the joints.
      early drop    ep175: a ball fell back onto the table while the dustpan was still on the table at the start of
                    dump. Added: within DUMP_BACK_S of the dump start, a ball seen on the table with the dustpan seen
                    on the table, on BACK_N replies -> back to sweep (at most MAX_BACK times). Never once the dustpan is
                    off the table (the demos never sweep with it lifted).
      home too soon ep176 (+2 unrecorded runs): DUMP_MAX_S sent home at 14 s with the balls still in the lifted
                    dustpan. Now: home when the dustpan is seen empty on EMPTY_N replies after it was seen off the
                    table, or the arm-return rule fires while the last reply sees no ball in the dustpan; at DUMP_MAX_S
                    only if no ball is seen in the dustpan, otherwise wait up to DUMP_HARD_S. HOME_BY_S still wins.
      empty         "empty" needs counts that add up to balls_total, no ball in the dustpan, and either every ball
                    seen in the basket or on the table, or the dustpan seen tipped (the pour): in ep175 the dustpan wrist camera was blocked and the LLM reported both balls
                    hidden while they were still in the low dustpan. Same test keeps dump going past DUMP_MAX_S.
    Everything else (thresholds, grasp band, brush-drop regrasp, stop rule) is Proprio's."""
    name = 'proprio2'
    GRASP_SEEN_N, LIFT_SEEN_N, BACK_N, EMPTY_N = 2, 2, 2, 2
    DUMP_BACK_S, MAX_BACK, DUMP_HARD_S = 8.0, 2, 26.0
    OFF_TABLE = ('lifting', 'over_basket', 'tipped')

    def __init__(self, **kw):
        super().__init__(**kw)
        self.chat.messages[0]['content'] = PROPRIO2_SYSTEM
        self.n_seen_held = self.n_seen_ball = self.n_lift = self.n_back = self.n_empty = 0
        self.n_backs, self.pan_was_up, self.last_state, self.last_in = 0, False, None, None

    def apply(self, d, now, rec):
        ok = super().apply(d, now, rec)
        if not ok or self.subtask is None or not isinstance(d, dict) or self.n_reply == 0:
            return ok
        c = self.last_counts or {}
        st = d.get('dustpan_state') if d.get('dustpan_state') in ('on_table', 'lifting', 'over_basket', 'tipped', 'set_down') else None
        held = d.get('dustpan_held') is True and d.get('brush_held') is True
        n_in, n_tab = c.get('balls_in_dustpan'), c.get('balls_on_table')
        self.last_state, self.last_in = st, n_in
        self.n_seen_held = self.n_seen_held + 1 if held else 0
        self.n_seen_ball = self.n_seen_ball + 1 if (n_in or 0) >= 1 else 0
        self.n_lift = self.n_lift + 1 if st in ('over_basket', 'tipped') else 0
        self.n_back = self.n_back + 1 if (st == 'on_table' and (n_tab or 0) >= 1) else 0
        n_bas = self._count(d, 'balls_in_basket') or 0; n_hid = c.get('balls_hidden') or 0
        if st == 'tipped': self.pour_seen = True
        # out of the dustpan for sure: every ball seen in the basket or on the table, or the pour seen and none in or hidden
        accounted = self.n_total is not None and n_bas + (n_tab or 0) >= self.n_total
        consistent = self.n_total is not None and None not in (n_in, n_tab) and n_in + n_tab + n_bas + n_hid == self.n_total
        if self.subtask == 'sweep' and not consistent:
            self.n_all_in = 0   # counts that do not add up are not evidence for 'all in'
        empty = consistent and n_in == 0 and (accounted or getattr(self, 'pour_seen', False))
        self.n_empty = self.n_empty + 1 if (empty and st is not None and st != 'on_table') else 0
        self.maybe_in = not empty   # a ball may still be in the dustpan
        if st in self.OFF_TABLE:
            self.pan_was_up = True
        if self.observations:
            self.observations[-1].update(state=st, held=held, in_basket=self._count(d, 'balls_in_basket'))
        return ok

    def go(self, subtask, now, why):
        super().go(subtask, now, why)
        self.n_seen_held = self.n_seen_ball = self.n_lift = self.n_back = self.n_empty = 0
        if subtask == 'dump':
            self.pan_was_up = self.last_state in self.OFF_TABLE; self.pour_seen = self.last_state == 'tipped'

    def on_chunk(self, obs, clock, env=None):
        now = clock.now; s = self.subtask; in_phase = now - self.phase_t
        g = view(obs)[1]
        not_open = all(x < self.OPEN for x in g)
        if s == 'grasp' and now < self.HOME_BY_S:
            if self.n_seen_held >= self.GRASP_SEEN_N and not_open:
                return self._go_then(obs, clock, env, 'sweep', f'both tools seen held on {self.n_seen_held} replies (grippers {g[0]:.2f}, {g[1]:.2f})')
            if self.n_seen_ball >= self.GRASP_SEEN_N:
                return self._go_then(obs, clock, env, 'sweep', f'a ball seen in the dustpan on {self.n_seen_ball} replies: already sweeping')
        elif s == 'sweep' and now < self.HOME_BY_S:
            c = self.last_counts or {}
            all_in_last = (self.n_total is not None and (c.get('balls_in_dustpan') or 0) >= self.n_total and c.get('balls_on_table') == 0)
            if all_in_last and self.last_state in self.OFF_TABLE and self.n_all_in >= 1:
                return self._go_then(obs, clock, env, 'dump', f'all {self.n_total} balls in and the dustpan already {self.last_state}')
            if self.n_lift >= self.LIFT_SEEN_N:
                return self._go_then(obs, clock, env, 'dump', f'the dustpan seen {self.last_state} on {self.n_lift} replies: the robot is pouring')
        elif s == 'dump' and now < self.HOME_BY_S:
            if (in_phase <= self.DUMP_BACK_S and not self.pan_was_up and self.n_back >= self.BACK_N and self.n_backs < self.MAX_BACK):
                self.n_backs += 1
                return self._go_then(obs, clock, env, 'sweep', f'a ball back on the table, dustpan still on the table ({self.n_backs})')
            if self.pan_was_up and self.n_empty >= self.EMPTY_N:
                return self._go_then(obs, clock, env, 'home', f'the dustpan seen empty on {self.n_empty} replies after it was lifted')
            if in_phase >= self.DUMP_MAX_S and getattr(self, 'maybe_in', True) and in_phase < self.DUMP_HARD_S:
                # Proprio would go home here; balls are still seen in the dustpan, so keep pouring (arm rule is gated too)
                self._dump_wait = True
                return self._observe_only(obs, clock, env)
            if getattr(self, 'maybe_in', True) and in_phase < self.DUMP_HARD_S:
                self._dump_hold = True   # gates Proprio's arm-return rule below
        return self._proprio_chunk(obs, clock, env)

    def _go_then(self, obs, clock, env, subtask, why):
        self.go(subtask, clock.now, why)
        if subtask == 'dump':
            self.q_dump0 = None if obs.get('arm') is None else np.asarray(obs['arm'], float)[6 * (0 if self.pan_side == 'left' else 1):][:6]
            self.d_peak = 0.0
        return self._observe_only(obs, clock, env)

    def _observe_only(self, obs, clock, env):
        """Keep the joint history and the background LLM queries going without re-running Proprio's rules."""
        arm = obs.get('arm')
        if arm is not None:
            self.arm_hist.append((clock.now, np.asarray(arm, float)))
            self.arm_hist = [(t, q) for t, q in self.arm_hist if t >= clock.now - max(self.FINISH_STILL_S, self.BRUSH_IDLE_S) - 1.0]
        if self.subtask != 'home' and clock.now >= self.HOME_BY_S:
            self.go('home', clock.now, f'time {clock.now:.0f} s >= {self.HOME_BY_S:.0f} s')
        LLMScheduler.on_chunk(self, obs, clock, env)

    def _proprio_chunk(self, obs, clock, env):
        hold = getattr(self, '_dump_hold', False); self._dump_hold = False
        if hold and self.subtask == 'dump':
            # Proprio's dump branch would send home on the arm-return rule; balls are still seen in the dustpan
            saved, self.DUMP_D_MIN = self.DUMP_D_MIN, float('inf')
            try:
                return Proprio.on_chunk(self, obs, clock, env)
            finally:
                self.DUMP_D_MIN = saved
        return Proprio.on_chunk(self, obs, clock, env)

    def summary(self):
        s = super().summary(); s.update(n_backs=self.n_backs); return s


def make_scheduler(kind='proprio', timing=None, model='deepseek-flash', thinking=False):
    if kind == 'proprio': return Proprio(model=model, thinking=thinking)
    if kind == 'proprio2': return Proprio2(model=model, thinking=thinking)
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
