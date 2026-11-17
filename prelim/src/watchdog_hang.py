"""Called by src/eval_one_task.sh after its watchdog killed a hung eval client (no progress in the eval log).

The patched client writes <state>.cur ({"now_id", "seed"}) when a policy episode starts and removes it when the
episode ends, so a hang inside an episode is known by episode id and seed. Each hang is appended to <state>.hangs.
The first hang of an episode is only recorded: the item is retried from the state file, i.e. the same episode on the
same seed (a fresh run of the policy usually gets through). A second hang of the same episode records it as a
failure: the state moves to the next seed and episode, and a "Success rate" line in the client's format is appended
to the eval log, so the result and every parser count it like any failed episode. A hang outside an episode (setup or
expert check: no <state>.cur) is only retried.

  python3 src/watchdog_hang.py <state.json> <eval log> <task> <task_config>
"""
import json
import os
import sys

state_file, eval_log, task, cfg = sys.argv[1:5]
cur_file, hangs_file = state_file + ".cur", state_file + ".hangs"


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


cur = load(cur_file)
if cur is None:
    print("watchdog: hang outside a policy episode (scene setup / expert check): retry")
    sys.exit(0)
key = f"{cur['now_id']} {cur['seed']}"
with open(hangs_file, "a") as f:
    f.write(key + "\n")
with open(hangs_file) as f:
    n = sum(1 for line in f if line.strip() == key)
if n < 2:
    print(f"watchdog: episode {cur['now_id']} on seed {cur['seed']} hung ({n}x): retry it")
    sys.exit(0)

state = load(state_file) or {"suc": 0, "test_num": 0}
new = {"now_seed": cur["seed"] + 1, "now_id": cur["now_id"] + 1, "suc": state["suc"],
       "test_num": state["test_num"] + 1}
with open(state_file + ".tmp", "w") as f:
    json.dump(new, f)
os.replace(state_file + ".tmp", state_file)
os.remove(cur_file)
rate = round(new["suc"] / new["test_num"] * 100, 1)
with open(eval_log, "a") as f:
    f.write(f"\n{task} | watchdog | {cfg} | episode {cur['now_id']} on seed {cur['seed']} hung twice: "
            f"counted as a failure\nSuccess rate: {new['suc']}/{new['test_num']} => {rate}%, "
            f"current seed: {cur['seed']}\n")
print(f"watchdog: episode {cur['now_id']} on seed {cur['seed']} hung twice: counted as a failure, state {new}")
