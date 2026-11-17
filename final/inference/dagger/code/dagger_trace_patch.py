"""DAgger takeover trace for lingbot_infer (5080). Minimal: one helper + one call in start / resume / pause / stop.

The platform's 人工接管 pauses inference and hands the slave arms to the master arms; 退出接管 resumes it (official
training notes). Each of these transitions appends one line to <runtime.log_dir>/takeover_events.jsonl with wall time,
monotonic time, session id, task, the latest observation's sequence/captured_at and its observation.state.* values,
and writes "Takeover trace: ..." to the service log. Frames of a recorded rollout between a pause and the next
resume/stop are the human correction (DAgger data); they are found offline by wall time plus the state snapshot.
Never raises: a failed write only logs an exception. Usage: python dagger_trace_patch.py <lingbot_infer_server.py>
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
MARK = '_log_takeover_event'
if MARK in s:
    print('already patched'); sys.exit(0)

helper = '''
    def _log_takeover_event(self, event: str, session=None) -> None:
        # DAgger trace (team patch): pause = 人工接管 starts, resume = 退出接管. One JSON line per transition.
        try:
            import time as _time
            if session is None:
                runtime = self._session_runtime
                session = runtime.session if runtime is not None else None
            robot = self._robot_runtime
            model_input = robot.mailbox.get() if robot is not None else None
            state = {}
            if model_input is not None:
                for key, value in model_input.features.items():
                    if key.startswith("observation.state") and not isinstance(value, str):
                        state[key] = np.asarray(value, dtype=float).round(5).tolist()
            record = {
                "wall_time": _time.time(),
                "monotonic": float(self._clock.now()),
                "event": event,
                "session_id": str(session.session_id) if session is not None else None,
                "task": getattr(session, "task", None),
                "task_type_id": getattr(session, "task_type_id", None),
                "obs_sequence": int(model_input.sequence) if model_input is not None else None,
                "obs_captured_at": float(model_input.captured_at) if model_input is not None else None,
                "state": state,
            }
            path = Path(self._config.runtime.log_dir) / "takeover_events.jsonl"
            with path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\\n")
            self._logger.info("Takeover trace: event=%s session_id=%s", event, record["session_id"])
        except Exception:
            self._logger.exception("Takeover trace failed: event=%s", event)
'''

edits = [
    ('''        if resumed:
            self._logger.info("Inference resumed")
            return
''', '''        if resumed:
            self._logger.info("Inference resumed")
            self._log_takeover_event("resume")
            return
'''),
    ('''        self._logger.info("Inference started: session_id=%s", session.session_id)
''', '''        self._logger.info("Inference started: session_id=%s", session.session_id)
        self._log_takeover_event("start")
'''),
    ('''        runtime.close()
        self._wait_for_inference()
        self._logger.info("Inference server stopped")
''', '''        runtime.close()
        self._wait_for_inference()
        self._logger.info("Inference server stopped")
        self._log_takeover_event("stop", runtime.session)
'''),
    ('''            runtime.paused = True
        self._logger.info("Inference server paused")
''', '''            runtime.paused = True
        self._logger.info("Inference server paused")
        self._log_takeover_event("pause")
'''),
    ('''    def status(self) -> RunningState:
''', helper.lstrip('\n') + '''
    def status(self) -> RunningState:
'''),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, (n, old[:60])
    s = s.replace(old, new)
p.write_text(s)
print('patched', p)
