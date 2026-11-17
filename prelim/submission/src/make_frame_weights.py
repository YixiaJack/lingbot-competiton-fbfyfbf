"""Write meta/frame_weights.npy for every converted clean dataset: one sampling weight per frame (global index order).

motion[t] = || action[t] - state[t] ||_2 over the 14 joint dims (arms + grippers): how far the robot moves between
frame t and t+1 in the demonstration. Frames where the robot does not move (motion < --static_eps; typically the
start / end of an episode and pauses) get weight --low_weight, all others 1. A pure function of the recorded clean
demonstrations (rule 2.1: reweighting by data statistics such as energy / smoothness); used with
LINGBOT_FRAME_WEIGHTS=1 (src/lingbot_sampling.py).

  python src/make_frame_weights.py --lerobot_root $DATA_DIR/lerobot/robotwin_clean
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_root", type=Path, required=True)
    ap.add_argument("--static_eps", type=float, default=1e-3)
    ap.add_argument("--low_weight", type=float, default=0.3)
    args = ap.parse_args()
    summary = {}
    for ds in sorted(args.lerobot_root.glob("*-aloha-agilex_clean_50-50")):
        n = json.load(open(ds / "meta" / "info.json"))["total_frames"]
        motion = np.full(n, np.nan)
        for f in sorted(glob.glob(str(ds / "data" / "*" / "*.parquet"))):
            t = pq.read_table(f, columns=["index", "observation.state", "action"])
            idx = t.column("index").to_numpy()
            s = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
            a = np.stack(t.column("action").to_numpy(zero_copy_only=False))
            motion[idx] = np.linalg.norm(a - s, axis=1)
        assert not np.isnan(motion).any(), f"{ds}: frames without data"
        w = np.where(motion < args.static_eps, args.low_weight, 1.0).astype(np.float32)
        np.save(ds / "meta" / "frame_weights.npy", w)
        task = ds.name.split("-aloha")[0]
        summary[task] = float((motion < args.static_eps).mean())
        print(f"{task:28s} frames {n:6d} static {100 * summary[task]:5.1f}%  effective share {w.sum() / n:.3f}")
    print(f"mean static fraction {100 * np.mean(list(summary.values())):.1f}%")


if __name__ == "__main__":
    main()
