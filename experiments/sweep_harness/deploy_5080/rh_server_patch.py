"""Hook robotharness into lingbot_infer_server.py (5080). Three call sites plus two helpers; every hook is wrapped so a
harness failure only logs and the policy keeps the original task. Mode file ~/robotharness/mode (off when missing).
Applies on top of the takeover-trace patch. usage: python rh_server_patch.py <lingbot_infer_server.py>"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
if '_robotharness_begin' in s:
    print('already patched'); sys.exit(0)
assert '_log_takeover_event' in s, 'expects the takeover-trace patch first'

helpers = '''
    def _robotharness_begin(self, session, policy):
        # Team patch: robotharness (task-3 subtask scheduler, DeepSeek-V4.1-Flash). Mode file ~/robotharness/mode:
        # off (default) | shadow (decide and log, VLA keeps the original task) | on (VLA gets the subtask instruction).
        try:
            import sys as _sys
            root = Path.home() / "robotharness"
            mode_file = root / "mode"
            mode = mode_file.read_text().strip() if mode_file.exists() else "off"
            if mode not in ("shadow", "on"):
                return None
            if str(root) not in _sys.path:
                _sys.path.append(str(root))
            import robotharness_bridge as _rh
            bridge = _rh.HarnessBridge(server=self, session=session, policy=policy, mode=mode,
                                       logger=self._logger, log_dir=self._config.runtime.log_dir)
            if not bridge.wanted():
                return None
            bridge.first_decision()
            self._robotharness = bridge
            self._logger.info("robotharness: mode=%s session_id=%s first instruction=%r", mode, session.session_id, bridge.last)
            return bridge
        except Exception:
            self._logger.exception("robotharness start failed; the policy keeps the original task")
            return None

    def _robotharness_end(self) -> None:
        bridge = getattr(self, "_robotharness", None)
        if bridge is not None:
            bridge.close()
            self._robotharness = None
'''

edits = [
    ('''        try:
            with self._policy_lock:
                with self._chunk_transition_lock:
                    if (
                        self._is_shutdown
                        or self._session_runtime is not candidate
                        or self._robot_runtime is not robot_runtime
                    ):
                        raise RuntimeError("START was cancelled before Policy init")
''', '''        rh_bridge = None
        try:
            with self._policy_lock:
                with self._chunk_transition_lock:
                    if (
                        self._is_shutdown
                        or self._session_runtime is not candidate
                        or self._robot_runtime is not robot_runtime
                    ):
                        raise RuntimeError("START was cancelled before Policy init")
'''),
    ('''                if scheduler_config.async_pipeline.rtc_enabled:
                    policy.validate_rtc(
                        refill_threshold=scheduler_config.async_pipeline.refill_threshold,
                        action_hz=scheduler_config.action_hz,
                    )

            with self._chunk_transition_lock:
''', '''                if scheduler_config.async_pipeline.rtc_enabled:
                    policy.validate_rtc(
                        refill_threshold=scheduler_config.async_pipeline.refill_threshold,
                        action_hz=scheduler_config.action_hz,
                    )

            rh_bridge = self._robotharness_begin(session, policy)
            with self._chunk_transition_lock:
'''),
    ('''        self._logger.info("Inference started: session_id=%s", session.session_id)
        self._log_takeover_event("start")
''', '''        self._logger.info("Inference started: session_id=%s", session.session_id)
        self._log_takeover_event("start")
        if rh_bridge is not None:
            rh_bridge.run_async()
'''),
    ('''        self._logger.info("Inference server stopped")
        self._log_takeover_event("stop", runtime.session)
''', '''        self._logger.info("Inference server stopped")
        self._log_takeover_event("stop", runtime.session)
        self._robotharness_end()
'''),
    ('''    def status(self) -> RunningState:
''', helpers.lstrip('\n') + '''
    def status(self) -> RunningState:
'''),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, (n, old[:70])
    s = s.replace(old, new)
p.write_text(s)
print('patched robotharness hooks into', p)
