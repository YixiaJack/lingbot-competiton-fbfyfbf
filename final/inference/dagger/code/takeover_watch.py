"""Auto takeover / handback detection for lingbot_infer (team patch, 11/15).

The platform's human takeover does NOT send PAUSE to the inference service (robot bags 11/15 04:26-04:28): the policy
kept planning while the robot bridge executed the master arm, and at the handback the arm snapped 0.5 rad at 7.19 rad/s
towards the stale plan. The robot bridge's /robby/schema/global/teleop_action shows the takeover by its rate:
  - policy mode: a 1:1 mirror of our /robby/schema/global/action, i.e. our send rate (30 Hz), 0 Hz when nothing runs;
  - master-arm teleop: the master command at ~226 Hz (139 msgs in the first second, 118 in the last).
During the takeover its content IS the command the bridge sends to the puppet (arms + gripper openings), so republishing
the latest one on our action topic follows the arm with zero command jump at the handback, and keeps the grip force
(a measured-position target would loosen a gripper that holds something).

TakeoverDetector: messages in a sliding window; takeover when the rate stays >= on_hz for on_hold s, handback when it
stays < off_hz for off_hold s. TakeoverWatch: owns the detector, the latest raw teleop message, a daemon thread that
calls the server's pause()/start() (exactly like the platform's PAUSE/START would) and the off switch.

Off switch: touch ~/takeover_watch/off (checked every second; rm to re-enable). Thresholds: ~/takeover_watch/config.json
(keys of TakeoverDetector.__init__ + "auto_resume": false to only pause/follow and leave the resume to the platform).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

TOPIC = "/robby/schema/global/teleop_action"
HOME = Path.home() / "takeover_watch"


class TakeoverDetector:
    def __init__(self, on_hz=100.0, off_hz=60.0, window=0.25, on_hold=0.1, off_hold=0.3):
        self.on_hz, self.off_hz, self.window = float(on_hz), float(off_hz), float(window)
        self.on_hold, self.off_hold = float(on_hold), float(off_hold)
        self._ts = deque()
        self._lock = threading.Lock()
        self.active = False
        self._since = None   # time the rate first crossed the threshold towards the other state

    def feed(self, t: float) -> None:
        with self._lock:
            self._ts.append(t)

    def rate(self, now: float) -> float:
        with self._lock:
            while self._ts and self._ts[0] <= now - self.window:
                self._ts.popleft()
            n = sum(1 for t in self._ts if t <= now)
        return n / self.window

    def update(self, now: float):
        """-> 'start' | 'end' | None"""
        r = self.rate(now)
        crossing = r < self.off_hz if self.active else r >= self.on_hz
        if not crossing:
            self._since = None
            return None
        if self._since is None:
            self._since = now
        if now - self._since < (self.off_hold if self.active else self.on_hold):
            return None
        self.active = not self.active
        self._since = None
        return "start" if self.active else "end"


class TakeoverWatch:
    """Server side. attach(node) after each robot connect; the server calls follow_message() from its paused tick."""

    def __init__(self, server, logger, period=0.02):
        self.server, self.log, self.period = server, logger, period
        cfg = {}
        try:
            cfg = json.loads((HOME / "config.json").read_text())
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("takeover_watch: bad config.json, using defaults")
        self.auto_resume = bool(cfg.pop("auto_resume", True))
        self.det = TakeoverDetector(**{k: v for k, v in cfg.items() if k in
                                       ("on_hz", "off_hz", "window", "on_hold", "off_hold")})
        self._raw = None          # latest serialized teleop_action (callback thread writes, tick thread reads)
        self._raw_t = None
        self._sub = None
        self.enabled = True
        self._last_off_check = 0.0
        self._last_rate_log = 0.0
        HOME.mkdir(exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="takeover-watch", daemon=True)
        self._thread.start()
        logger.info("takeover_watch: on (on_hz=%.0f off_hz=%.0f window=%.2f on_hold=%.2f off_hold=%.2f auto_resume=%s)",
                    self.det.on_hz, self.det.off_hz, self.det.window, self.det.on_hold, self.det.off_hold,
                    self.auto_resume)

    # ---- ROS side (robot executor thread) ----
    def attach(self, node) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from robby_schema_msgs.msg import RobotAction
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=50)
        self._raw = None
        self._sub = node.create_subscription(RobotAction, TOPIC, self._on_msg, qos, raw=True)

    def _on_msg(self, raw) -> None:
        t = time.monotonic()
        self._raw, self._raw_t = raw, t
        self.det.feed(t)

    # ---- tick side ----
    def follow_message(self):
        """A RobotAction = the latest teleop command with fresh stamps, or None."""
        raw = self._raw
        if raw is None:
            return None
        from builtin_interfaces.msg import Time
        from rclpy.serialization import deserialize_message
        from robby_schema_msgs.msg import RobotAction
        msg = deserialize_message(raw, RobotAction)
        ns = time.time_ns()
        stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        msg.header.stamp = stamp
        for group in msg.joint_groups:
            group.trajectory.header.stamp = stamp
        return msg

    # ---- watcher thread ----
    def _event(self, name, **extra):
        rec = {"wall_time": time.time(), "monotonic": time.monotonic(), "event": name, **extra}
        try:
            with open(HOME / "events.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _loop(self):
        while True:
            try:
                self._step(time.monotonic())
            except Exception:
                self.log.exception("takeover_watch: step failed")
            time.sleep(self.period)

    def _step(self, now):
        if now - self._last_off_check >= 1.0:
            self._last_off_check = now
            on = not (HOME / "off").exists()
            if on != self.enabled:
                self.enabled = on
                self.log.warning("takeover_watch: %s by ~/takeover_watch/off", "enabled" if on else "DISABLED")
                self._event("enabled" if on else "disabled")
        ev = self.det.update(now)
        r = self.det.rate(now)
        if now - self._last_rate_log >= 1.0 and r > 0:
            self._last_rate_log = now
            self._event("rate", hz=round(r, 1), active=self.det.active)
        # off blocks new takeovers only; a takeover already auto-paused still gets its handback (else it would sit
        # paused on the frozen pre-takeover target and snap at the platform's resume)
        if ev is None or (ev == "start" and not self.enabled):
            return
        self.log.info("takeover_watch: %s detected (teleop_action %.0f Hz)", ev, r)
        self._event("takeover_start" if ev == "start" else "takeover_end", hz=round(r, 1))
        if ev == "start":
            self.server._takeover_watch_start()
        else:
            self.server._takeover_watch_end(self.auto_resume)
