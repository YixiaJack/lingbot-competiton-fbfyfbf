"""Offline check of the patched helper on fake runtime objects (no robot, no GPU). Usage: python dagger_trace_test.py <tmpdir>"""
import json, logging, sys, uuid
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from lingbot_infer.lingbot_infer_server import LingBotInferenceServer as S
from utils.data_define import ModelInput
from utils.data_struct import LatestValueMailbox

d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
mb = LatestValueMailbox(); mb.set(ModelInput(sequence=np.int64(7), captured_at=np.float64(123.5), features={
    'observation.state.left_arm.position': np.arange(6, dtype=np.float32) / 3, 'observation.state.left_gripper.position': np.array([0.4]),
    'observation.images.camera_top': np.zeros((4, 4, 3), np.uint8), 'task': 'pick'}))
sess = NS(session_id=uuid.uuid4(), task='Pick up the apple', task_type_id=1)
fake = NS(_session_runtime=NS(session=sess), _robot_runtime=NS(mailbox=mb), _clock=NS(now=lambda: 42.0),
          _config=NS(runtime=NS(log_dir=d)), _logger=logging.getLogger('t'))
for ev in ('start', 'pause', 'resume'):
    S._log_takeover_event(fake, ev)
fake._session_runtime = None
S._log_takeover_event(fake, 'stop', sess)
fake._robot_runtime = None; S._log_takeover_event(fake, 'stop')          # no robot, no session: still one line
fake._config = None; S._log_takeover_event(fake, 'x')                    # broken config: must not raise
rows = [json.loads(l) for l in open(d / 'takeover_events.jsonl')]
assert [r['event'] for r in rows] == ['start', 'pause', 'resume', 'stop', 'stop'], rows
assert rows[1]['state']['observation.state.left_arm.position'][1] == 0.33333 and 'observation.images.camera_top' not in rows[1]['state']
assert rows[3]['session_id'] == str(sess.session_id) and rows[4]['session_id'] is None and rows[4]['state'] == {}
print('ok', json.dumps(rows[1], ensure_ascii=False)[:300])
