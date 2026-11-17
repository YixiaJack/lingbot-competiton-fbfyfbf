"""Runs ON the 5080, fed over ssh stdin by server.py (only while someone has the /topref page open):
    ssh lbreal 'source ~/workspace/lingbot/inference/env.sh >/dev/null 2>&1; exec nice -n 19 python3 -u -' < cam_remote.py
Subscribes read-only (BEST_EFFORT, depth 1) to the head (top) camera JPEG topic the inference service also reads and
prints at most FPS frames per second as "f<TAB>stamp<TAB>base64 jpeg", plus "hb<TAB>wall time<TAB>frames seen" every 2 s.
Writes nothing on the 5080. Exits when the board side goes away (broken pipe) or after MAX_S."""
import base64, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

TOPIC, FPS, MAX_S = '/robby/sensors/camera/head/color/compressed', 4.0, 3 * 3600
rclpy.init()
node = Node('lb_board_cam')
st = {'last': None, 'seen': 0, 'sent_t': 0.0}


def cb(m):
    st['last'] = m; st['seen'] += 1


node.create_subscription(CompressedImage, TOPIC, cb, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                                                 history=HistoryPolicy.KEEP_LAST, depth=1))
t0 = last_hb = time.time()
try:
    while time.time() - t0 < MAX_S:
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.time()
        m = st['last']
        if m is not None and now - st['sent_t'] >= 1.0 / FPS:
            st['last'] = None; st['sent_t'] = now
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            sys.stdout.write(f"f\t{stamp:.3f}\t{base64.b64encode(bytes(m.data)).decode()}\n"); sys.stdout.flush()
        if now - last_hb >= 2:
            last_hb = now
            sys.stdout.write(f"hb\t{now:.3f}\t{st['seen']}\n"); sys.stdout.flush()
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    node.destroy_node(); rclpy.shutdown()
