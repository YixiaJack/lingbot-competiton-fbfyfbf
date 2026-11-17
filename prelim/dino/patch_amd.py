"""AMD (ROCm, LeRobot 0.6) compatibility for the patched lingbot-vla-v2 checkout. Idempotent.

The AMD server's lingbot-vla-v2 copy (upstream 951475a + the hosts' ROCm / LeRobot 0.6 edits) reads v3 datasets through
LeRobot 0.6's DatasetReader and returns early in LeRobotDataset.__getitem__:

    if LEROBOT_DATASET_API == "v3" and hasattr(self, "_ensure_reader"):
        return self._ensure_reader().get_item(idx)

so the seen-instruction swap that src/patch_upstream.py appends to the end of __getitem__ never runs. This applies the
same swap on that early path (same pool file meta/seen_instructions.json, same LINGBOT_SEEN_POOL=1 switch).

  python dino/patch_amd.py <lingbot-vla-v2 checkout>     # after src/patch_upstream.py and dino/patch_dino.py
"""

import sys
from pathlib import Path

FILE = "lingbotvla/data/vla_data/base_dataset.py"
OLD = (
    '        if LEROBOT_DATASET_API == "v3" and hasattr(self, "_ensure_reader"):\n'
    "            return self._ensure_reader().get_item(idx)\n"
)
NEW = (
    '        if LEROBOT_DATASET_API == "v3" and hasattr(self, "_ensure_reader"):\n'
    "            item = self._ensure_reader().get_item(idx)\n"
    "            # LingBotRoboTwin (AMD): seen-instruction swap on the LeRobot 0.6 reader path (rule 2.1)\n"
    '            if os.environ.get("LINGBOT_SEEN_POOL") == "1":\n'
    '                if not hasattr(self, "_seen_pool"):\n'
    '                    pool_file = Path(self.root) / "meta" / "seen_instructions.json"\n'
    "                    self._seen_pool = json.loads(pool_file.read_text()) if pool_file.exists() else {}\n"
    '                choices = self._seen_pool.get(str(int(item["episode_index"])))\n'
    "                if choices:\n"
    '                    item["task"] = choices[int(torch.randint(len(choices), ()))]\n'
    "            return item\n"
)


def main():
    root = Path(sys.argv[1])
    p = root / FILE
    code = p.read_text()
    if NEW in code:
        print(f"already patched {FILE}")
        return
    assert code.count(OLD) == 1, f"LeRobot 0.6 reader path not found exactly once in {p}"
    assert "LINGBOT_SEEN_POOL" in code, "apply src/patch_upstream.py first"
    for name in ("import json", "from pathlib import Path", "import torch"):
        assert f"\n{name}" in code or code.startswith(name), f"{FILE} lacks `{name}`"
    p.write_text(code.replace(OLD, NEW))
    print(f"patched {FILE}")


if __name__ == "__main__":
    main()
