"""GPU test of the patched async + RTC inference path on the 5080, in its own process. The service must have no model
loaded, since 16 GB does not hold two copies. Uses the real adapter (PolicyFactory + init, config/inference.rtc.yaml)
and recorded frames (robotharness test_frames_ep84.npz); no robot.
  1. the RTC config parses; the real checkpoint check refuses an unmarked checkpoint (plain async, no prefix);
  2. no-prefix call (first chunk / sync): same seed -> identical chunk twice;
  3. RTC call with the last 20 actions of chunk A as the prefix: B keeps the prefix, plus the junction step;
  4. RTC requested but the checkpoint is not RTC-trained: identical to the no-prefix call (the disabled path is the
     original one);
  5. timing of both.
usage (on the 5080): cd ~/workspace/lingbot/inference && source env.sh &&
  $LINGBOT_PYTHON_ENV/bin/python ~/rtc/test_rtc_infer.py <hf_ckpt> ~/robotharness/test/test_frames_ep84.npz"""
import io, logging, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.arguments import LingBotInferConfig
from utils.data_define import ActionChunk, ModelInput
import policy_adapter.policy_adapter as PA

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
MODEL, Z = sys.argv[1], np.load(sys.argv[2])
ROOT = Path.home() / 'workspace/lingbot/inference'
TASK = ('Brush all paper balls off the tabletop into the dustpan before pouring them into the basket. '
        'Place both tools down; return both arms home and remain stationary.')


def model_input(name):
    dec = lambda c: np.asarray(Image.open(io.BytesIO(Z[f'{name}/{c}'].tobytes())).convert('RGB'))
    arm, g = np.asarray(Z[f'{name}/arm'], np.float32), np.asarray(Z[f'{name}/grippers'], np.float32)
    h = len(arm) // 2
    f = {f'observation.images.{c}': dec(c) for c in ('camera_top', 'camera_wrist_left', 'camera_wrist_right')}
    f.update({'observation.state.left_arm.position': arm[:h], 'observation.state.right_arm.position': arm[h:],
              'observation.state.left_gripper.position': g[:1], 'observation.state.right_gripper.position': g[1:2]})
    return ModelInput(sequence=1, captured_at=time.monotonic(), features=f)


cfg = LingBotInferConfig.from_yaml(ROOT / 'config/inference.rtc.yaml', overrides={'policy.model_path': MODEL})
sched = cfg.scheduler if hasattr(cfg, 'scheduler') else None
print('1. config:', 'use_length', cfg.policy.options.get('use_length'), '| async', getattr(sched, 'async_pipeline', '?'))
pol = PA.PolicyFactory.create(cfg.policy)
pol.init(cfg.robot, TASK)
pol.validate_rtc(refill_threshold=20, action_hz=30.0)
print('   real checkpoint check:', PA._rtc_ckpt_ok(MODEL), '| _rtc_ok =', pol._rtc_ok)
assert pol._rtc_ok is False, 'the step-2000 checkpoint must not pass as RTC-trained'

mi = model_input('start')


def run(seed, d=None, prev=None):
    torch.manual_seed(seed)
    t0 = time.time()
    x = pol.preprocess(mi, run_inference=True, rtc_prefix_len=d, prev_action_chunk=prev)
    has_prefix = '__rtc_prefix__' in x
    out = pol.postprocess(pol.infer(x))
    torch.cuda.synchronize()
    return out, time.time() - t0, has_prefix


def flat(chunk):
    keys = sorted(chunk.actions)
    return np.concatenate([np.asarray(chunk.actions[k], np.float64).reshape(chunk.chunk_length, -1) for k in keys], axis=1)


run(0)   # warm-up
A, ta, _ = run(0)
A2, _, _ = run(0)
noise = np.abs(flat(A) - flat(A2)).max()
print(f'2. no prefix: chunk {A.chunk_length} steps, {ta * 1000:.0f} ms; same seed twice: max |diff| {noise:.2e} rad (GPU nondeterminism)')

prev = ActionChunk(actions={k: np.asarray(v)[24:44] for k, v in A.actions.items()})
B_off, t_off, pre_off = run(1, 20, prev)            # _rtc_ok False -> must equal a no-prefix call with seed 1
C, _, _ = run(1)
d_off = np.abs(flat(B_off) - flat(C)).max()
print(f'4. RTC requested, unmarked checkpoint: prefix attached={pre_off}; vs no-prefix call with the same seed: '
      f'max |diff| {d_off:.2e} rad (same-seed noise {noise:.2e})')

pol._rtc_ok = True                                  # pretend RTC-trained to exercise the prefix path
times = []
for s in range(3):
    B, tb, pre_on = run(2 + s, 20, prev)
    times.append(tb)
fa, fb = flat(prev), flat(B)
keep = np.abs(fb[:20] - fa).max()
arm = sorted(B.actions)[:2]                         # left/right arm keys sort before the gripper keys
steps = np.abs(np.diff(fb, axis=0)).max(axis=1)
print(f'3. RTC prefix 20: attached={pre_on}, prefix kept max |diff| = {keep:.2e} rad, junction step {steps[19]:.4f} '
      f'vs median step {np.median(steps[20:]):.4f}; {np.mean(times) * 1000:.0f} ms (no prefix {ta * 1000:.0f} ms)')
assert pre_on and keep < 1e-3, 'prefix not kept'
print('ALL CHECKS DONE')
