"""Server-path test of the takeover_watch patch on a COPY of lingbot_infer (5080 service python, no ROS, no model).
usage: python test_takeover_server.py <deploy dir containing the patched lingbot_infer/>"""
import sys, threading, time
from types import SimpleNamespace
from unittest.mock import MagicMock
sys.path.insert(0, sys.argv[1])
import lingbot_infer.lingbot_infer_server as S

srv = object.__new__(S.LingBotInferenceServer)
srv._chunk_transition_lock = threading.Lock()
srv._logger = MagicMock(); srv._clock = SimpleNamespace(now=time.monotonic)
srv._is_shutdown = False; srv._policy_loading = False; srv._robot_runtime = MagicMock()
srv._log_takeover_event = lambda ev, session=None: events.append(ev)
srv._yolocue = SimpleNamespace(on_resume=lambda: events.append('yolocue_on_resume'))
events, sent = [], []
buf = MagicMock(); buf.sample_status.return_value = ('CTRL', 3, False); buf.last_sent = 'LAST_SENT'
robot = SimpleNamespace(communication=SimpleNamespace(send=lambda m: sent.append(m)),
                        robot_adapter=SimpleNamespace(_action_channel='robot_action'), scheduler_config=MagicMock())
sess = SimpleNamespace(session_id='sid')
rt = S._SessionRuntime(session=sess, robot=robot, action_buffer=buf, scheduler=MagicMock())
srv._session_runtime = rt
srv._send_control = lambda runtime, control: sent.append(('frozen', control))
srv._takeover_watch = SimpleNamespace(follow_message=lambda: 'TELEOP_MSG')
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print('OK ' if cond else 'BAD', name)

check('fields default', rt.follow is False and rt.takeover_auto is False)
srv._takeover_watch_start()
check('auto pause', rt.paused and rt.follow and rt.takeover_auto and rt.paused_control == 'LAST_SENT' and events[-1] == 'pause')
srv._on_control_tick('sid', time.monotonic())
check('paused tick republishes teleop', len(sent) == 1 and sent[0].channel == 'robot_action' and sent[0].payload == 'TELEOP_MSG')
srv._takeover_watch = SimpleNamespace(follow_message=lambda: None)
srv._on_control_tick('sid', time.monotonic())
check('no teleop msg -> frozen target', sent[-1] == ('frozen', 'LAST_SENT'))
srv._takeover_watch = SimpleNamespace(follow_message=lambda: 'TELEOP_MSG')
srv._takeover_watch_end(True)
check('auto resume via DAgger path', not rt.paused and not rt.follow and not rt.takeover_auto and buf.clear.called
      and rt.pending_request_id is None and events[-2:] == ['resume', 'yolocue_on_resume'])
# platform PAUSE during an auto takeover -> accepted, no auto resume
buf.clear.reset_mock(); srv._takeover_watch_start(); n = len(events)
srv.pause()
check('platform PAUSE during auto takeover accepted', rt.paused and rt.follow and not rt.takeover_auto and len(events) == n)
srv._takeover_watch_end(True)
check('...then no auto resume at handback', rt.paused and not buf.clear.called)
srv._on_control_tick('sid', time.monotonic())
check('...and it keeps holding the human pose', sent[-1].payload == 'TELEOP_MSG')
srv.start()
check('platform START resumes', not rt.paused and not rt.follow and buf.clear.called)
# takeover while the platform paused us: follow, no auto resume
srv.pause(); buf.clear.reset_mock(); srv._takeover_watch_start()
check('takeover during platform pause: follow, not auto', rt.paused and rt.follow and not rt.takeover_auto)
srv._takeover_watch_end(True)
check('...no auto resume', rt.paused and not buf.clear.called)
srv.start()
# auto_resume=False config
srv._takeover_watch_start(); srv._takeover_watch_end(False)
check('auto_resume false: stays paused + follow', rt.paused and rt.follow)
srv.start()
# no session
srv._session_runtime = None; srv._takeover_watch_start(); srv._takeover_watch_end(True)
check('no session: no-op', True)
print('SERVER TEST OK' if ok else 'SERVER TEST FAILED')
