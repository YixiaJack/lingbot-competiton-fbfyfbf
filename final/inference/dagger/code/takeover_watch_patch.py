"""Auto takeover detection for lingbot_infer (5080), on top of the takeover-trace, robotharness, yolocue, RTC and DAgger
resume patches. Needs takeover_watch.py next to lingbot_infer_server.py (lingbot_infer/takeover_watch.py).

Why: the platform's takeover sends no PAUSE (bags 11/15 04:28), so the policy kept planning, drifted 0.43-0.61 rad from
the arm, and the arm snapped 0.5 rad at 7.19 rad/s at the handback. With this patch:
  - takeover detected (teleop_action rate >= 100 Hz)  -> server.pause() (trace "pause"), session marked follow +
    takeover_auto; while paused + follow, each control tick republishes the latest teleop_action (= the command the
    bridge is sending to the puppet, gripper included) instead of the frozen pre-takeover target;
  - handback detected (< 60 Hz for 0.3 s)            -> server.start() = the DAgger resume path (old plan dropped,
    fresh inference from the current pose, no RTC prefix, yolocue on_resume). Between the handback and the resume the
    tick keeps republishing teleop_action, which is then the bridge's mirror of our own last command -> the arm holds;
  - a takeover while the platform has paused us: follow only, no auto resume (the platform resumes);
  - the platform's PAUSE during an auto takeover is accepted (no error) and turns it into a manual pause (no auto resume).
Off switch: touch ~/takeover_watch/off. Events: ~/takeover_watch/events.jsonl (+ the service log, "takeover_watch:").
usage: python takeover_watch_patch.py <lingbot_infer_server.py>   (idempotent)"""
import shutil, sys, time
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
if '_takeover_watch_attach' in s:
    print('already patched'); sys.exit(0)
assert '_yolocue_resume' in s, 'expects the DAgger resume patch'
assert (p.parent / 'takeover_watch.py').exists(), 'copy takeover_watch.py next to the server first'
bak = Path.home() / f'dagger_patch/{p.name}.before_tw_{time.strftime("%Y%m%d_%H%M%S")}'
bak.parent.mkdir(exist_ok=True)
shutil.copy2(p, bak)
edits = [
    # session flags (slots dataclass: fields must be declared)
    ("""    paused_control: Control | None = None
    paused: bool = False

    def close(self) -> None:
        with ExitStack() as stack:
            stack.callback(self.action_buffer.clear)
""", """    paused_control: Control | None = None
    paused: bool = False
    follow: bool = False  # team patch (takeover_watch): paused ticks republish the human's teleop command
    takeover_auto: bool = False  # team patch (takeover_watch): this pause came from a detected takeover

    def close(self) -> None:
        with ExitStack() as stack:
            stack.callback(self.action_buffer.clear)
"""),
    # resume clears the flags
    ("""                current.paused_control = None
                current.paused = False
                resumed = True
""", """                current.paused_control = None
                current.paused = False
                current.follow = False
                current.takeover_auto = False
                resumed = True
"""),
    # the platform's PAUSE during an auto takeover: accept, becomes a manual pause
    ("""            runtime = self._session_runtime
            if runtime is None or runtime.paused:
                raise RuntimeError("PAUSE requires RUNNING state")
""", """            runtime = self._session_runtime
            if runtime is not None and runtime.paused and runtime.takeover_auto:
                # Team patch (takeover_watch): already auto-paused by a detected takeover; the platform's PAUSE wins,
                # so no auto resume at the handback (the platform will START).
                runtime.takeover_auto = False
                self._logger.info("PAUSE during an auto takeover: now a platform pause (no auto resume)")
                return
            if runtime is None or runtime.paused:
                raise RuntimeError("PAUSE requires RUNNING state")
"""),
    # paused tick: follow the human
    ("""            if runtime.paused:
                control = runtime.paused_control
""", """            if runtime.paused:
                if runtime.follow and self._takeover_follow_send(runtime):
                    return
                control = runtime.paused_control
"""),
    # subscribe on the robot node before it starts spinning
    ("""            communication.connect(lambda message: self._on_robot_msg(runtime, message))
            node = communication.node
""", """            communication.connect(lambda message: self._on_robot_msg(runtime, message))
            node = communication.node
            self._takeover_watch_attach(node)
"""),
    ("""    def _yolocue_resume(self) -> None:
""", """    # ---- team patch: auto takeover detection (lingbot_infer/takeover_watch.py) ----
    def _takeover_watch_attach(self, node) -> None:
        try:
            if getattr(self, "_takeover_watch", None) is None:
                from lingbot_infer.takeover_watch import TakeoverWatch
                self._takeover_watch = TakeoverWatch(self, self._logger)
            self._takeover_watch.attach(node)
        except Exception:
            self._logger.exception("takeover_watch: not available; human takeovers will not be detected")

    def _takeover_watch_start(self) -> None:
        with self._chunk_transition_lock:
            runtime = self._session_runtime
            if runtime is None:
                return
            runtime.follow = True
            if runtime.paused:
                self._logger.info("takeover_watch: takeover while paused by the platform; following the arm, no auto resume")
                return
            runtime.takeover_auto = True
        try:
            self.pause()
        except Exception:
            self._logger.exception("takeover_watch: auto pause failed")
            with self._chunk_transition_lock:
                runtime.takeover_auto = False
                if not runtime.paused:
                    runtime.follow = False

    def _takeover_watch_end(self, auto_resume: bool) -> None:
        with self._chunk_transition_lock:
            runtime = self._session_runtime
            auto = runtime is not None and runtime.paused and runtime.takeover_auto
        if not (auto and auto_resume):
            self._logger.info("takeover_watch: handback; holding where the human left the arm until START")
            return
        try:
            self.start()
        except Exception:
            self._logger.exception("takeover_watch: auto resume failed; the arm holds, send START to continue")

    def _takeover_follow_send(self, runtime) -> bool:
        # called from the paused control tick, under _chunk_transition_lock
        watch = getattr(self, "_takeover_watch", None)
        if watch is None:
            return False
        try:
            msg = watch.follow_message()
            if msg is None:
                return False
            from utils.data_define import OutboundMsg
            channel = getattr(runtime.robot.robot_adapter, "_action_channel", "robot_action")
            runtime.robot.communication.send(OutboundMsg(channel=channel, payload=msg))
            return True
        except Exception:
            if not getattr(self, "_takeover_follow_failed", False):
                self._takeover_follow_failed = True
                self._logger.exception("takeover_watch: follow send failed; sending the frozen target instead")
            return False

    def _yolocue_resume(self) -> None:
"""),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, (n, old[:90])
    s = s.replace(old, new)
p.write_text(s)
print('patched takeover_watch into', p, '; backup', bak)
