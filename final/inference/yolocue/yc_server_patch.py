"""Hook yolocue into lingbot_infer_server.py (5080), on top of the takeover-trace and robotharness patches.

Three call sites: start() right after robotharness (before the control loop), stop() (reset() goes through stop()),
serve() (background preload of the detector). Every hook catches its own errors: a yolocue failure only logs and the
VLA gets the raw image. Mode file ~/yolocue/mode (off when missing), read at every start.
usage: python yc_server_patch.py <lingbot_infer_server.py>
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
if '_yolocue_begin' in s:
    print('already patched')
    sys.exit(0)
assert '_robotharness_begin' in s, 'expects the robotharness patch first'

helpers = '''
    def _yolocue_import(self):
        import sys as _sys
        root = Path.home() / "yolocue"
        if str(root) not in _sys.path:
            _sys.path.append(str(root))
        import yolocue_bridge as _yc
        return _yc

    def _yolocue_begin(self, session, policy):
        # Team patch: yolocue (the YOLO box cue of the box/plain training mix, tasks 1/2). Mode file ~/yolocue/mode:
        # off (default) | shadow (detect and log, the VLA keeps the raw image) | on (the VLA gets the cue).
        self._yolocue_end()
        try:
            root = Path.home() / "yolocue"
            mode_file = root / "mode"
            mode = mode_file.read_text().strip() if mode_file.exists() else "off"
            if mode not in ("shadow", "on"):
                return None
            _yc = self._yolocue_import()
            cue = _yc.YoloCue(server=self, session=session, policy=policy, mode=mode,
                              logger=self._logger, log_dir=self._config.runtime.log_dir)
            if not cue.wanted():
                return None
            self._yolocue = cue
            cue.begin()
            self._logger.info("yolocue: mode=%s session_id=%s %s", mode, session.session_id, cue.state)
            return cue
        except Exception:
            self._logger.exception("yolocue start failed; the VLA gets the raw image")
            self._yolocue_end()
            try:
                self._yolocue_import().uninstall(policy)
            except Exception:
                pass
            return None

    def _yolocue_end(self) -> None:
        cue = getattr(self, "_yolocue", None)
        self._yolocue = None
        if cue is not None:
            try:
                cue.close()
            except Exception:
                self._logger.exception("yolocue close failed")

    def _yolocue_preload(self) -> None:
        try:
            self._yolocue_import().preload(self._logger)
        except Exception:
            self._logger.exception("yolocue preload failed")
'''

edits = [
    ('''            rh_bridge = self._robotharness_begin(session, policy)
''', '''            rh_bridge = self._robotharness_begin(session, policy)
            self._yolocue_begin(session, policy)
'''),
    ('''        self._log_takeover_event("stop", runtime.session)
        self._robotharness_end()
''', '''        self._log_takeover_event("stop", runtime.session)
        self._robotharness_end()
        self._yolocue_end()
'''),
    ('''        inference_thread.start()
        command_thread.start()
''', '''        inference_thread.start()
        command_thread.start()
        self._yolocue_preload()
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
print('patched yolocue hooks into', p)
