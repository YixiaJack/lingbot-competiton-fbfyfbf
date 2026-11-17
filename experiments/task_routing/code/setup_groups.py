"""Kit training-group configs for the 8 arms of the routing experiment, B200 only (as requested for this ablation).

Members of one arm relay by rank (the kit's handoff): whichever starts trains; a higher-ranked member that starts
later sends the running one back to the queue and resumes from the newest checkpoint. The global batch stays 128
(gradient accumulation), so changing the GPU count does not change the run.
  python route/setup_groups.py        -> R2/configs/group_<arm>.sh
"""
import os
from pathlib import Path

R2 = Path(os.path.expandvars("$SCRATCH/lingbot/final_route_20261112"))
H = os.path.expandvars("$SCRATCH/lingbot/LingBotRoboTwin/route")
ARMS = ["E1_frozen", "E4_fullsft", "E2_shared_mid", "E3_task_mid", "E2_shared_last", "E3_task_last",
        "E2_shared_all", "E3_task_all"]
STEPS, SAVE = 3000, 1500
ADV = os.environ.get("SLURM_ACCOUNT_ADVANCED", "<slurm-account-advanced>")   # Slurm accounts of your allocation
GEN = os.environ.get("SLURM_ACCOUNT_GENERAL", "<slurm-account-general>")
X4 = "--gres=gpu:b200:4 --constraint=b200 --cpus-per-task=48 --mem=400G --time=12:00:00"
X2 = "--gres=gpu:b200:2 --constraint=b200 --cpus-per-task=24 --mem=200G --time=12:00:00"
X1 = "--gres=gpu:b200:1 --constraint=b200 --cpus-per-task=16 --mem=200G --time=12:00:00"
MEMBERS = [
    ("b200x4_dept_prem", 6, ADV, X4, "MICRO_BS=16 GRAD_CKPT=0 FULL_SHARD=0"),
    ("b200x4", 5, GEN, X4, "MICRO_BS=16 GRAD_CKPT=0 FULL_SHARD=0"),
    ("b200x2_dept_prem", 4, ADV, X2, "MICRO_BS=16 GRAD_CKPT=0 FULL_SHARD=1"),
    ("b200x2", 3, GEN, X2, "MICRO_BS=16 GRAD_CKPT=0 FULL_SHARD=1"),
    ("b200x1_dept_prem", 2, ADV, X1, "MICRO_BS=8 GRAD_CKPT=1 FULL_SHARD=0"),
    ("b200x1", 1, GEN, X1, "MICRO_BS=8 GRAD_CKPT=1 FULL_SHARD=0"),
]


def short(arm):
    return arm.split("_", 1)[0].lower() + "_" + arm.split("_")[-1]   # e1_frozen, e3_mid, ...


def main():
    for arm in ARMS:
        lines = [os.path.expandvars("source $SCRATCH/lingbot/tmp/gpu_queue_kit/config.example.sh"),
                 f"LOG_DIR={R2}/logs", f"GROUP_NAME=lb_route_{short(arm)}_1112", f"GROUP_DIR={R2}/runs/{arm}",
                 f"TRAIN_CMD='bash {H}/train_route.sh {arm} {STEPS} {SAVE}'", "MEMBERS=("]
        lines += [f'  "{n}|{r}|{acct}|1|{sb}|{env}"' for n, r, acct, sb, env in MEMBERS]
        lines.append(")")
        (R2 / "configs" / f"group_{arm}.sh").write_text("\n".join(lines) + "\n")
        (R2 / "runs" / arm).mkdir(parents=True, exist_ok=True)
    print("wrote", len(ARMS), "group configs")


if __name__ == "__main__":
    main()
