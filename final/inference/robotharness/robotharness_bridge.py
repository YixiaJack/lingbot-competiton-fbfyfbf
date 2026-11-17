"""robotharness inside lingbot_infer (5080): the task-3 subtask scheduler (harness.Fixed, DeepSeek-V4.1-Flash) drives
the instruction the VLA gets, without stopping or restarting the inference session.

The policy adapter attaches self._task to every observation it sends to the model, so switching the subtask is one
attribute write between two inferences. Layout on the 5080 (~/robotharness):
  mode                 off (default when missing) | shadow (decide and log, VLA keeps the original task) | on
  code/                harness.py, real_split.py, prompts.py, skills.py (copied unchanged from the notes repo)
  deps/openai/         standard-library stand-in for the openai client (the inference env has no openai)
  ~/.deepseek_env      DEEPSEEK_API_KEY=... (mode 600; harness._load_key reads it)
Only sessions whose task mentions the dustpan / paper balls / basket (task 3) are touched; tasks 1 and 2 never are.
Log: <runtime.log_dir>/robotharness_<time>_<session>.jsonl (start, every LLM call without images, every prompt change,
done, errors, summary). Any failure leaves the VLA on the instruction it already has.
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT / 'code') not in sys.path:
    sys.path.append(str(ROOT / 'code'))   # appended: the inference process keeps its own modules first
T3_WORDS = ('dustpan', 'paper ball', 'basket')
PERIOD_S = 0.8   # ~one action chunk (use_length 25 at 30 Hz); harness.Fixed keeps one LLM query in flight at a time


class HarnessBridge:
    def __init__(self, server, session, policy, mode, logger, log_dir):
        self.server, self.session, self.policy, self.mode, self.logger = server, session, policy, mode, logger
        self.task = getattr(session, 'task', None) or ''
        self.sid = str(getattr(session, 'session_id', ''))
        self.log_path = Path(log_dir) / f"robotharness_{time.strftime('%Y%m%d_%H%M%S')}_{self.sid[:8]}.jsonl"
        self.stop_evt, self.sched, self.clock, self.last, self.thread = threading.Event(), None, None, None, None

    def wanted(self):
        t = self.task.lower()
        return any(w in t for w in T3_WORDS)

    def _log(self, **rec):
        rec = {'wall_time': round(time.time(), 3), **rec}
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        except Exception:
            self.logger.exception('robotharness log write failed')

    def obs(self):
        rt = self.server._robot_runtime
        mi = rt.mailbox.get() if rt is not None else None
        if mi is None:
            raise RuntimeError('no observation in the mailbox')
        f = mi.features
        img = lambda k: np.asarray(f[f'observation.images.{k}'])
        st = lambda k: np.asarray(f[f'observation.state.{k}.position'], dtype=float).ravel()
        return {'images': {'top': img('camera_top'), 'left': img('camera_wrist_left'), 'right': img('camera_wrist_right')},
                'grippers': (float(st('left_gripper')[0]), float(st('right_gripper')[0])),
                'arm': np.concatenate([st('left_arm'), st('right_arm')])}

    def _apply(self, why):
        p = self.sched.prompt()
        if p and p != self.last:
            applied = self.mode == 'on' and hasattr(self.policy, '_task')
            if applied:
                self.policy._task = p
            self._log(event='prompt', why=why, prompt=p, applied=applied, t=round(self.clock.now, 2),
                      subtask=getattr(self.sched, 'subtask', None), side=getattr(self.sched, 'pan_side', None))
            self.last = p

    def _log_calls(self, start):
        for c in self.sched.chat.calls[start:]:
            self._log(event='llm', **{k: c.get(k) for k in ('latency_s', 'user', 'reply', 'error', 'prompt_tokens', 'cache_hit_tokens')})
        return len(self.sched.chat.calls)

    def first_decision(self):
        """Blocking (called before the control loop starts), so the VLA's first chunk already gets the subtask."""
        import harness as H
        self.sched, self.clock = H.make_scheduler('fixed'), H.WallClock(H.LIMIT_S)
        self._log(event='start', mode=self.mode, task=self.task, session_id=self.sid, harness_file=H.__file__)
        self.sched.start(self.obs(), self.clock)
        self.n_calls = self._log_calls(0)
        self._apply('first decision')

    def run_async(self):
        self.thread = threading.Thread(target=self._loop, name='robotharness', daemon=True)
        self.thread.start()

    def _alive(self):
        rt = self.server._session_runtime
        return rt is not None and rt.session is self.session and not self.stop_evt.is_set()

    def _loop(self):
        try:
            while self._alive() and not self.sched.done and not self.clock.expired:
                time.sleep(PERIOD_S)
                if not self._alive():
                    break
                self.sched.on_chunk(self.obs(), self.clock)
                self.n_calls = self._log_calls(self.n_calls)
                self._apply('llm')
            if self.sched.done:
                self._log(event='done', t=round(self.clock.now, 2), prompt=self.last)
        except Exception as e:
            self._log(event='error', error=repr(e))
            self.logger.exception('robotharness loop failed; the VLA keeps its current instruction')
        finally:
            try:
                s = self.sched.summary()
                s.pop('calls', None)
                self._log(event='summary', **s)
            except Exception:
                pass
            self.sched.close()

    def close(self):
        self.stop_evt.set()
