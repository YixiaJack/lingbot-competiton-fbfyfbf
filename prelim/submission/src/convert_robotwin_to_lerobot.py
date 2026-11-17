"""Convert one RoboTwin 2.0 task's aloha-agilex_clean_50 demos into a LeRobot dataset for LingBot-VLA 2.0.

Equivalent to the upstream two-step pipeline that LingBot-VLA's experiment/robotwin/README.md prescribes
(RoboTwin policy/pi0/scripts/process_data.py -> examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py),
fused into one pass and ported to lerobot 0.4.2 (the version in the lingbotvla env, writes LeRobot v3.0):

  * state[j]  = [left_arm(6), left_gripper, right_arm(6), right_gripper] from /joint_action at frame j
  * action[j] = state[j + 1]; the last frame is dropped                      (same as process_data.py)
  * cameras head/left/right -> cam_high / cam_left_wrist / cam_right_wrist    (same as process_data.py)
  * one "seen" instruction per episode, chosen at random                       (same as the pi0 converter)
  * fps=50                                                                     (same as the pi0 converter)

Two deliberate differences:
  * images keep their native 320x240 instead of the pi0 script's upsample to 640x480, because the RoboTwin eval
    client sends native-resolution frames to the policy server;
  * cameras are stored as video (lerobot default AV1, keyframe every 2 frames), not as images embedded in parquet.
    LingBot's loader reads action chunks row-first and only fetches future frames for video keys; with embedded
    images every sample decoded ~150 PNGs (3.5 s/sample measured), with video it reads numbers only.

  python src/convert_robotwin_to_lerobot.py --raw_dir $DATA_DIR/raw/adjust_bottle/aloha-agilex_clean_50 \
      --out_dir $DATA_DIR/lerobot/robotwin_clean/adjust_bottle-aloha-agilex_clean_50-50
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate",
    "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate",
    "right_gripper",
]
CAMERAS = {"head_camera": "cam_high", "left_camera": "cam_left_wrist", "right_camera": "cam_right_wrist"}


def load_episode(path: Path):
    with h5py.File(path, "r") as root:
        ja = root["joint_action"]
        state = np.concatenate(
            [ja["left_arm"][()], ja["left_gripper"][()][:, None], ja["right_arm"][()], ja["right_gripper"][()][:, None]],
            axis=1,
        ).astype(np.float32)
        images = {}
        for src, dst in CAMERAS.items():
            images[dst] = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in root[f"observation/{src}/rgb"][()]]
    return state, images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True, help=".../<task>/aloha-agilex_clean_50")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--num_episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert "randomized" not in str(args.raw_dir), "competition rule: clean data only"
    rng = np.random.default_rng(args.seed)

    state0, images0 = load_episode(args.raw_dir / "data" / "episode0.hdf5")
    h, w = images0["cam_high"][0].shape[:2]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
    }
    for cam in CAMERAS.values():
        features[f"observation.images.{cam}"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    ds = LeRobotDataset.create(
        repo_id=f"robotwin_clean/{args.out_dir.name}", root=args.out_dir, fps=50, robot_type="aloha",
        features=features, use_videos=True, image_writer_processes=4, image_writer_threads=4,
    )
    for i in range(args.num_episodes):
        state, images = (state0, images0) if i == 0 else load_episode(args.raw_dir / "data" / f"episode{i}.hdf5")
        with open(args.raw_dir / "instructions" / f"episode{i}.json") as f:
            instruction = str(rng.choice(json.load(f)["seen"]))
        for j in range(len(state) - 1):
            frame = {"observation.state": state[j], "action": state[j + 1], "task": instruction}
            for cam in CAMERAS.values():
                frame[f"observation.images.{cam}"] = images[cam][j]
            ds.add_frame(frame)
        ds.save_episode()
        print(f"episode {i}: {len(state) - 1} frames | {instruction}", flush=True)
    ds.finalize()
    print("done", args.out_dir, "frames", ds.meta.total_frames, "episodes", ds.meta.total_episodes)


if __name__ == "__main__":
    main()
