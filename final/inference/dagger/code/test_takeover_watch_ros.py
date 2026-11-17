"""ROS test of TakeoverWatch on the 5080, isolated (ROS_DOMAIN_ID=77, ROS_LOCALHOST_ONLY=1; the robot uses domain 0).
A fake bridge publishes RobotAction on teleop_action at 30 Hz (3 s) -> 226 Hz (3 s) -> 30 Hz (3 s); a fake server
records the pause/start calls. Checks: one start ~3.2 s, one end ~6.5 s, follow_message() = the last published command
with a fresh stamp. Run with the service's python: source env.sh; ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 python this.py"""
import logging, os, sys, threading, time
from pathlib import Path
assert os.environ.get('ROS_DOMAIN_ID') == '77' and os.environ.get('ROS_LOCALHOST_ONLY') == '1'
os.environ['HOME'] = '/tmp/tw_test_home'; Path('/tmp/tw_test_home').mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent))
import takeover_watch as tw
tw.HOME = Path('/tmp/tw_test_home/takeover_watch')
import rclpy
from rclpy.executors import SingleThreadedExecutor
from robby_schema_msgs.msg import RobotAction, NamedJointTrajectory, GripperOpening
from trajectory_msgs.msg import JointTrajectoryPoint
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('t')

class FakeServer:
    def __init__(self): self.calls = []
    def _takeover_watch_start(self): self.calls.append(('start', time.monotonic()))
    def _takeover_watch_end(self, auto): self.calls.append(('end', time.monotonic()))

ctx = rclpy.context.Context(); rclpy.init(context=ctx)
ex = SingleThreadedExecutor(context=ctx)
sub_node = rclpy.create_node('tw_sub', context=ctx); pub_node = rclpy.create_node('tw_pub', context=ctx)
srv = FakeServer(); w = tw.TakeoverWatch(srv, log); w.attach(sub_node)
ex.add_node(sub_node); threading.Thread(target=ex.spin, daemon=True).start()
pub = pub_node.create_publisher(RobotAction, tw.TOPIC, 10)
time.sleep(2.0)   # discovery

def msg(v):
    m = RobotAction()
    g = NamedJointTrajectory(); g.role = 'left_arm'; g.trajectory.joint_names = [f'joint_{i}' for i in range(1, 7)]
    pt = JointTrajectoryPoint(); pt.positions = [v] * 6; g.trajectory.points = [pt]
    m.joint_groups = [g]; m.grippers = [GripperOpening(role='left_gripper', opening=v / 10)]
    return m

t0 = time.monotonic(); k = 0
for dur, hz in ((3, 30), (3, 226), (3, 30)):
    end = time.monotonic() + dur
    while time.monotonic() < end:
        k += 1; pub.publish(msg(float(k))); time.sleep(1 / hz)
last = k
time.sleep(1.0)
f = w.follow_message()
rel = [(e, round(t - t0, 2)) for e, t in srv.calls]
print('calls:', rel)
fresh = abs(f.header.stamp.sec + f.header.stamp.nanosec / 1e9 - time.time()) < 0.5
print('follow msg left_arm[0] =', f.joint_groups[0].trajectory.points[0].positions[0], 'expected', float(last),
      'gripper', round(f.grippers[0].opening, 3), 'fresh stamp', fresh)
ok = (len(rel) == 2 and rel[0][0] == 'start' and 3.0 < rel[0][1] < 3.8 and rel[1][0] == 'end' and 6.0 < rel[1][1] < 7.0
      and f.joint_groups[0].trajectory.points[0].positions[0] == float(last) and fresh)
print('events.jsonl:', (tw.HOME / 'events.jsonl').read_text().count('\n'), 'lines;',
      [l for l in (tw.HOME / 'events.jsonl').read_text().splitlines() if 'takeover' in l])
print('ROS TEST OK' if ok else 'ROS TEST FAILED')
os._exit(0)
