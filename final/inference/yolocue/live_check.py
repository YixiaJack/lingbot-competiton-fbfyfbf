"""Read one head-camera frame from ROS (read-only), decode it like robot_adapter._rgb_image, run the yolocue detector,
save the frame with every detection >= 0.25 drawn. usage: source env.sh; python live_check.py [instruction]"""
import json, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.append(str(Path.home() / 'yolocue'))
import yolocue_bridge as YC
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

rclpy.init()
node = rclpy.create_node('yolocue_live_check')
got = {}
node.create_subscription(CompressedImage, '/robby/sensors/camera/head/color/compressed', lambda m: got.setdefault('m', m),
                         QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
t0 = time.time()
while 'm' not in got and time.time() - t0 < 10:
    rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node(); rclpy.shutdown()
if 'm' not in got:
    sys.exit('no head-camera frame within 10 s')
rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(got['m'].data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
dets = YC.detector(YC.load_config())(rgb)
shown = [d for d in dets if d[5] >= 0.25]
out = {'format': got['m'].format, 'shape': list(rgb.shape), 'dets': shown}
if len(sys.argv) > 1:
    pr = YC.parse(sys.argv[1]); out['parsed'] = pr
    if pr:
        out['cue'] = YC.select(dets, *pr)
from PIL import Image, ImageDraw
im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
for r in shown:
    d.rectangle(r[1:5], outline=(255, 0, 0), width=1); d.text((r[1] + 2, r[2] + 1), f'{r[0]} {r[5]:.2f}', fill=(255, 0, 0))
im.save(Path.home() / 'yolocue/test_out/live.jpg', quality=92)
print(json.dumps(out, ensure_ascii=False))
