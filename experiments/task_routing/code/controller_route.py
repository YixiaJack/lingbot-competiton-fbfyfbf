"""Login-node controller of the routing experiment. Standard library only; one instance (flock).

- Waits for smoke/SMOKE_OK (all smoke checks passed) before submitting any full run.
- Trains the 8 arms in priority order, at most MAX_ACTIVE at once, each as a kit training group (relay by rank).
- A group with no job left in the queue and no DONE is resubmitted (ONLY=missing), at most 4 times.
- When an arm is DONE: cancels its pending members and submits its RoboTwin evaluation (eval_pool.sbatch, 4 tasks x
  25 episodes, demo_clean, shared policy server; A100 / H100 / H200 / L40S, B200 cannot run the RoboTwin eval).
- Writes controller_state.json every loop; stops when PAUSED exists.
  setsid nohup python3 -u route/controller_route.py > R2/logs/controller.log 2>&1 &
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_groups import ARMS, STEPS, short  # noqa: E402

R2 = Path(os.path.expandvars("$SCRATCH/lingbot/final_route_20261112"))
KIT = Path(os.path.expandvars("$SCRATCH/lingbot/tmp/gpu_queue_kit"))
WORK = Path(os.path.expandvars("$SCRATCH/lingbot/LingBotRoboTwin"))
ROUTE_DIR = os.path.expandvars("$SCRATCH/lingbot/dev/lingbot-vla-v2-route")
TASKS = "place_a2b_left,place_a2b_right,put_object_cabinet,dump_bin_bigbin"
EVAL_EPISODES = 25
MAX_ACTIVE = int(os.environ.get("ROUTE_MAX_ACTIVE", "3"))


def sh(cmd, timeout=300):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def jobs(name):
    rc, out = sh(f"squeue -h -u $USER -n {name} -o '%i %T'")
    if rc:
        return None
    return [l.split() for l in out.splitlines() if l.strip()]


def eval_name(arm):
    return f"route_{arm}"


def eval_done(arm):
    d = Path(os.path.expandvars(f"$SCRATCH/lingbot/eval/{eval_name(arm)}/demo_clean"))
    n = 0
    for t in TASKS.split(","):
        f = d / f"{t}.result"
        if f.exists():
            try:
                s, a = map(int, f.read_text().split()[:2])
                n += a >= EVAL_EPISODES
            except ValueError:
                pass
    return n


def main():
    lock = open(R2 / "controller.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another controller runs")
    (R2 / "controller.pid").write_text(str(os.getpid()))
    st_path = R2 / "controller_state.json"
    st = json.loads(st_path.read_text()) if st_path.exists() else {}
    st.setdefault("submitted", []); st.setdefault("resubmits", {}); st.setdefault("eval_submitted", [])
    st.setdefault("failed", []); st.setdefault("log", [])

    def note(msg):
        st["log"] = (st["log"] + [f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"])[-60:]
        print(msg, flush=True)

    while not (R2 / "PAUSED").exists():
        smoke_ok = (R2 / "smoke" / "SMOKE_OK").exists()
        done = [a for a in ARMS if (R2 / "runs" / a / "DONE").exists()]
        for a in done:
            js = jobs(f"lb_route_{short(a)}_1112") or []
            pend = [j for j, s in js if s == "PENDING"]
            if pend:
                sh("scancel " + " ".join(pend)); note(f"{a} DONE: cancelled pending members {pend}")
            if a not in st["eval_submitted"]:
                ck = R2 / "runs" / a / "checkpoints" / f"global_step_{STEPS}" / "hf_ckpt"
                phases = f"{eval_name(a)}|demo_clean|{EVAL_EPISODES}|{TASKS}|{ck}|{ROUTE_DIR}"
                rc, out = sh(f"cd {WORK} && SHARED_SERVER=1 SLOTS=2 PHASES='{phases}' GPU_CONSTRAINT='h100|h200|a100|l40s' "
                             f"bash scripts/hpc/submit.sh scripts/hpc/eval_pool.sbatch --array=0-1 --gres=gpu:1 "
                             f"--cpus-per-task=8 --mem=80G --time=03:00:00 --job-name=lb_route_eval_{short(a)}")
                note(f"{a}: eval submitted rc={rc} {out.strip().splitlines()[-1] if out.strip() else ''}")
                if rc == 0:
                    st["eval_submitted"].append(a)
        active = [a for a in st["submitted"] if a not in done and a not in st["failed"]]
        if smoke_ok:
            for a in ARMS:
                if len(active) >= MAX_ACTIVE:
                    break
                if a in st["submitted"] or a in done:
                    continue
                rc, out = sh(f"cd {KIT} && bash train/submit_group.sh {R2}/configs/group_{a}.sh", timeout=600)
                note(f"{a}: group submitted rc={rc} {out.strip().splitlines()[-1] if out.strip() else ''}")
                if rc == 0:
                    st["submitted"].append(a); active.append(a)
            for a in active:
                js = jobs(f"lb_route_{short(a)}_1112")
                if js is not None and not js and not (R2 / "runs" / a / "DONE").exists():
                    n = st["resubmits"].get(a, 0)
                    if n >= 4:
                        st["failed"].append(a); note(f"{a}: no members left after 4 resubmits -> FAILED, see logs")
                        continue
                    rc, out = sh(f"cd {KIT} && ONLY=missing bash train/submit_group.sh {R2}/configs/group_{a}.sh", 600)
                    st["resubmits"][a] = n + 1; note(f"{a}: no members in queue, resubmitted ({n + 1}) rc={rc}")
        st.update({"updated_utc": datetime.now(timezone.utc).isoformat(), "smoke_ok": smoke_ok, "done": done,
                   "active": active, "evals_complete": {a: eval_done(a) for a in st["eval_submitted"]}})
        tmp = st_path.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1)); tmp.replace(st_path)
        if all(eval_done(a) == 4 for a in ARMS):
            note("all arms trained and evaluated"); break
        time.sleep(120)


if __name__ == "__main__":
    main()
