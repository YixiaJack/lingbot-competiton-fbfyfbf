"""Offline test of the robotharness hook on the 5080, in its own process (no robot, no GPU, the service is untouched).
  1. bridge: task-3 detection and the observation conversion from a ModelInput built from recorded frames (own ep84)
  2. the patched server's _robotharness_begin with the real mode file: 'off' must leave everything alone
  3. --llm: shadow-mode decisions from DeepSeek on four moments of ep84 (start / sweeping / pouring / home), latency
usage (inside env.sh): CUDA_VISIBLE_DEVICES= python test_rh.py test_frames_ep84.npz [--llm]"""
import io, json, logging, sys, time, uuid
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from PIL import Image

sys.path.append(str(Path.home() / 'robotharness'))
import robotharness_bridge as RB
from utils.data_define import ModelInput
from utils.data_struct import LatestValueMailbox

Z = np.load(sys.argv[1]); LLM = '--llm' in sys.argv
logging.basicConfig(level=logging.INFO); log = logging.getLogger('test_rh')
T3 = 'Brush all paper balls off the tabletop into the dustpan before pouring them into the basket. Place both tools down; return both arms home and remain stationary.'


def model_input(name):
    dec = lambda c: np.asarray(Image.open(io.BytesIO(Z[f'{name}/{c}'].tobytes())).convert('RGB'))
    arm, g = np.asarray(Z[f'{name}/arm'], float), np.asarray(Z[f'{name}/grippers'], float)
    h = len(arm) // 2
    f = {f'observation.images.{c}': dec(c) for c in ('camera_top', 'camera_wrist_left', 'camera_wrist_right')}
    f.update({'observation.state.left_arm.position': arm[:h], 'observation.state.right_arm.position': arm[h:],
              'observation.state.left_gripper.position': g[:1], 'observation.state.right_gripper.position': g[1:2]})
    return ModelInput(sequence=1, captured_at=time.monotonic(), features=f)


mb = LatestValueMailbox(); mb.set(model_input('start'))
session = NS(task=T3, session_id=uuid.uuid4()); policy = NS(_task=T3)
server = NS(_robot_runtime=NS(mailbox=mb), _session_runtime=NS(session=session), _logger=log,
            _config=NS(runtime=NS(log_dir=Path('/tmp/rh_test'))))
Path('/tmp/rh_test').mkdir(exist_ok=True)

b = RB.HarnessBridge(server, session, policy, 'shadow', log, '/tmp/rh_test')
o = b.obs()
print('1. wanted(task 3) =', b.wanted(), '| wanted(task 1) =', RB.HarnessBridge(server, NS(task='Put the lemon on the white plate.', session_id='x'), policy, 'shadow', log, '/tmp').wanted(),
      '| images', {k: v.shape for k, v in o['images'].items()}, '| grippers', o['grippers'], '| arm', o['arm'].shape)

from lingbot_infer.lingbot_infer_server import LingBotInferenceServer as S
mode_file = Path.home() / 'robotharness/mode'
print('2. mode file:', mode_file.read_text().strip() if mode_file.exists() else '(missing = off)',
      '-> _robotharness_begin returned', S._robotharness_begin(server, session, policy), '| policy task unchanged:', policy._task == T3)

if LLM:
    t0 = time.time(); b.first_decision()
    print(f"3. start: subtask={b.sched.subtask} side={b.sched.pan_side} in {time.time() - t0:.1f}s -> {b.sched.prompt()!r}")
    def settle():   # wait for the query in flight and apply it, so the next query sees the next frame
        while b.sched.future is not None:
            while not b.sched.future.done():
                time.sleep(0.1)
            parsed, rec = b.sched.future.result(); b.sched.apply(parsed, b.clock.now, rec); b.sched.future = None
    for name in ('sweep', 'dump', 'home'):
        settle()
        mb.set(model_input(name))
        b.sched.on_chunk(b.obs(), b.clock)                     # submits the query for this frame
        settle()                                               # applies this frame's answer
        c = b.sched.chat.calls[-1]
        print(f"   {name:6s}: subtask={b.sched.subtask} side={b.sched.pan_side} latency={c['latency_s']}s error={c['error']} reply={str(c['reply'])[:160]}")
    b.sched.close()
    print('   policy._task untouched in shadow mode:', policy._task == T3, '| log:', b.log_path)
