"""One check of the s22k8k_v3uni_15k handoff group against the per-user GPU cap (run on the cluster login node, piped by
scripts/gpu_cap_watch.sh every 5 min). If a member that outranks the running one (or any member when none runs) is held
back by a per-user GPU limit (reason QOSMaxGRESPerUser or similar), cancel up to STEP eval pool workers to make room:
first workers that hold no claim, then the ones with the fewest finished episodes in their current item (their items
are taken over by the remaining workers from the state files). At most MAX_TOTAL workers over the whole watch (counter
in tmp/). Goal (10/25): as many training steps as possible; stopping a few eval workers is acceptable."""

import glob
import json
import os
import subprocess
import sys

TRAIN_JOB = "lb_s22k8k_v3uni_15k"
EVAL_JOB = "lb_evalq"
RANKS = {}  # {"<job id>": handoff rank} of the group's members (filled in by hand for each submission)
STEP, MAX_TOTAL = 4, 8
EVAL = os.path.expandvars("$SCRATCH/lingbot/eval")
COUNTER = os.path.expandvars("$SCRATCH/lingbot/tmp/gpucap_cancelled.json")


def sq():
    r = subprocess.run(["timeout", "120", "squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%j|%T|%r"], capture_output=True, text=True)
    if r.returncode:
        print("squeue failed or timed out; nothing done")
        sys.exit(0)
    return [l.split("|") for l in r.stdout.splitlines() if l.count("|") == 3]


rows = sq()
train = [r for r in rows if r[1] == TRAIN_JOB]
run_rank = max((RANKS.get(r[0], 0) for r in train if r[2] == "RUNNING"), default=0)
blocked = [r for r in train if r[2] == "PENDING" and RANKS.get(r[0], 0) > run_rank
           and any(k in r[3] for k in ("MaxGRES", "GrpGRES", "MaxTRES", "GrpTRES"))]
running = ", ".join(f"{r[0]}(rank {RANKS.get(r[0], 0)})" for r in train if r[2] == "RUNNING") or "none"
if not blocked:
    print(f"ok: training running {running}; no faster member held by a GPU limit")
    sys.exit(0)
try:
    done = json.load(open(COUNTER))
except Exception:
    done = []
left = MAX_TOTAL - len(done)
if left <= 0:
    print(f"blocked {[(r[0], r[3]) for r in blocked]} but {len(done)} workers already cancelled (max {MAX_TOTAL}); nothing done")
    sys.exit(0)
workers = [r[0] for r in rows if r[1] == EVAL_JOB and r[2] == "RUNNING"]
ep = {}  # owner job -> finished episodes of its current item
for d in glob.glob(f"{EVAL}/*/_claims/*"):
    try:
        owner = open(f"{d}/owner").read().strip().split(":")[0]
    except Exception:
        continue
    cfg, _, task = os.path.basename(d).partition("__")
    try:
        ep[owner] = json.load(open(f"{os.path.dirname(os.path.dirname(d))}/{cfg}/state/{task}.json")).get("now_id", 0)
    except Exception:
        ep[owner] = 0  # claimed, no episode finished yet
order = sorted(workers, key=lambda w: (w in ep, ep.get(w, 0)))  # no claim first, then fewest episodes
pick = order[:min(STEP, left)]
if pick:
    subprocess.run(["scancel"] + pick)
    done += pick
    json.dump(done, open(COUNTER, "w"))
print(f"blocked {[(r[0], r[3]) for r in blocked]} while running {running}: cancelled eval workers {pick} "
      f"(episodes {[ep.get(w) for w in pick]}); {len(done)}/{MAX_TOTAL} used")
