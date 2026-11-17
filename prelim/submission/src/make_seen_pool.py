"""Write meta/seen_instructions.json next to every converted clean dataset: {episode_index: [seen instructions]}.

Used by the patched LeRobot loader (LINGBOT_SEEN_POOL=1) to swap an episode's instruction for another one from the
same episode's `seen` pool on every read. This is rule 2.1 "seen 池内指令替换": release assets only, no environment
calls, the (instruction, trajectory) pairing count is unchanged. `unseen` instructions are never read.

  python src/make_seen_pool.py --raw_root $DATA_DIR/raw --lerobot_root $DATA_DIR/lerobot/robotwin_clean
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, required=True)
    ap.add_argument("--lerobot_root", type=Path, required=True)
    ap.add_argument("--num_episodes", type=int, default=50)
    args = ap.parse_args()
    for ds in sorted(args.lerobot_root.glob("*-aloha-agilex_clean_50-50")):
        task = ds.name.split("-aloha-agilex_clean_50")[0]
        inst_dir = args.raw_root / task / "aloha-agilex_clean_50" / "instructions"
        pool = {}
        for i in range(args.num_episodes):
            with open(inst_dir / f"episode{i}.json") as f:
                pool[str(i)] = list(json.load(f)["seen"])
        with open(ds / "meta" / "seen_instructions.json", "w") as f:
            json.dump(pool, f)
        print(task, len(pool), "episodes,", sum(map(len, pool.values())), "seen instructions")


if __name__ == "__main__":
    main()
