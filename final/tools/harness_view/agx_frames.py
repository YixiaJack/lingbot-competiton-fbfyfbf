"""Runs ON the robot host (<robot-user>@<robot-host>), fed over ssh stdin by mk_topref_t3.py. Read-only.
argv[1] = base64(JSON [[stem, wall_time_s], ...]). For each request, finds the minute recording of the platform recorder
(/mnt/ssd/bags/record/record_<local time>_<uuid>/data/*.mcap) that covers wall_time and prints the first head-camera
JPEG (/robby/sensors/camera/head/color/compressed, the same 640x480 stream the VLA gets) at or after wall_time:
    stem <TAB> frame time <TAB> base64 jpeg        (or: stem <TAB> MISSING <TAB> reason)
Run it as: source /opt/ros/humble/setup.bash; nice -n 19 ionice -c3 taskset -c 18-23 python3 - <b64 json>"""
import base64, datetime, glob, json, os, sys
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

TOPIC = '/robby/sensors/camera/head/color/compressed'
req = json.loads(base64.b64decode(sys.argv[1]))


def dir_time(d):
    try:
        return datetime.datetime.strptime(os.path.basename(d).split('_')[1], '%Y%m%d-%H%M%S').timestamp()
    except (IndexError, ValueError):
        return None


dirs = [(dir_time(d), d) for d in glob.glob('/mnt/ssd/bags/record/record_*')]
dirs = sorted((t, d) for t, d in dirs if t is not None)
for stem, t in req:
    hit = None
    for t0, d in dirs:
        if t0 - 2 <= t <= t0 + 75:
            for f in sorted(glob.glob(os.path.join(d, 'data', '*.mcap'))):
                r = rosbag2_py.SequentialReader()
                r.open(rosbag2_py.StorageOptions(uri=f, storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
                m = r.get_metadata(); s = m.starting_time.nanoseconds / 1e9; e = s + m.duration.nanoseconds / 1e9
                if s <= t < e - 0.2:
                    r.set_filter(rosbag2_py.StorageFilter(topics=[TOPIC])); r.seek(int(t * 1e9))
                    while r.has_next():
                        topic, data, ts = r.read_next()
                        if topic == TOPIC:
                            hit = (ts / 1e9, deserialize_message(data, CompressedImage).data); break
                if hit:
                    break
        if hit:
            break
    if hit:
        sys.stdout.write(f'{stem}\t{hit[0]:.3f}\t{base64.b64encode(bytes(hit[1])).decode()}\n')
    else:
        sys.stdout.write(f'{stem}\tMISSING\tno recording covers {t:.3f}\n')
    sys.stdout.flush()
