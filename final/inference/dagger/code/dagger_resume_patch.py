"""DAgger handback for lingbot_infer (5080), on top of the takeover-trace, robotharness, yolocue and RTC patches.
Official RESUME (start from PAUSED) continues the paused chunk on a shifted timeline. After a human takeover the arms
may be somewhere else, so the robot would snap back towards the old plan, and with async + RTC the next request would
even use that stale remainder as its prefix. This patch makes RESUME:
  - drop the old plan: action_buffer.clear() (current + staged chunk, timeline, last-sent anchor), pending_request_id
    and last_action_index reset; a late inference result is ignored because its request id no longer matches;
  - the next control tick then starts like a new session: inference from the current observation, no RTC prefix
    (rtc_prefix_len 0), first chunk anchored at the current pose;
  - tell yolocue (YoloCue.on_resume): rebuild the box cue from the current frame (placed / fresh cue / grasped).
PAUSE is unchanged (official: freeze, keep sending the frozen target); whether the robot host ignores it during a
teleop takeover must be checked on the robot.
usage: python dagger_resume_patch.py <lingbot_infer_server.py>   (idempotent)"""
import shutil, sys, time
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
if '_yolocue_resume' in s:
    print('already patched'); sys.exit(0)
assert '_yolocue_begin' in s and '_log_takeover_event' in s, 'expects the takeover-trace and yolocue patches'
shutil.copy2(p, Path.home() / f'dagger_patch/{p.name}.before_resume_{time.strftime("%Y%m%d_%H%M%S")}')
edits = [
    ("""                current.action_buffer.resume(self._clock.now())
                current.paused_control = None
                current.paused = False
                resumed = True
""", """                # Team patch (DAgger handback): a human may have moved the arms during the pause, so drop the old
                # plan (current + staged chunk, pending request; a late result no longer matches) and let the next
                # tick start like a new session from the current pose, with no RTC prefix.
                current.action_buffer.clear()
                current.pending_request_id = None
                current.last_action_index = None
                current.paused_control = None
                current.paused = False
                resumed = True
"""),
    ("""            self._logger.info("Inference resumed")
            self._log_takeover_event("resume")
            return
""", """            self._logger.info("Inference resumed (DAgger handback: old plan dropped, fresh start from the current pose)")
            self._log_takeover_event("resume")
            self._yolocue_resume()
            return
"""),
    ("""    def _yolocue_preload(self) -> None:
""", """    def _yolocue_resume(self) -> None:
        cue = getattr(self, "_yolocue", None)
        if cue is not None and hasattr(cue, "on_resume"):
            try:
                cue.on_resume()
            except Exception:
                self._logger.exception("yolocue: resume handling failed; the cue keeps its old state")

    def _yolocue_preload(self) -> None:
"""),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, (n, old[:80])
    s = s.replace(old, new)
p.write_text(s)
print('patched DAgger handback into', p)
