#!/usr/bin/env bash
# 10/25: continue the reference model strong_scratch22k_cont8k (22k + 8k = 30k) for MAX_STEPS (default 15000) more steps
# with the AMD +3k/+4k recipe (dino/amd/configs/cont_amd_3kto4k_seed43.yaml + dino/amd/train_amd.sh) but uniform task
# sampling: aug_strong_v3.json, lr 5e-6 -> 2e-6 cosine over the first 75% of the steps then flat (lr_decay_ratio 0.75),
# 3% warmup, seen-pool instructions, proprio noise 0.01, ViT lr x0.1, low-motion frames down-weighted, task sampling
# uniform (the +8k leg's) instead of temp:2. The depth / future-depth loss weights stay at 0.004 (their 1e-8 on AMD was a
# workaround for that box, not part of the recipe).
# Code: the DINO dev checkout (dev/lingbot-vla-v2-dino) without --train.dino_fusion_path, i.e. the plain model (the
# fusion module is only built when that path is set). It has the 10/25 per-rank reseed of the dataloader workers and
# the noise (all ranks drew the same temp:2 samples, a resume replayed the epoch start), time-based resume saves every
# 24 min (LINGBOT_TIME_SAVE_MIN) and resume across GPU counts.
# Members as dino/submit_dino_cont.sh, all on the preemption partitions (this account cannot submit B200 jobs to the
# regular partitions: "GPU job setup is not valid"), ranked handoff (scripts/hpc/train.sbatch HANDOFF_RANK):
# 4xH200 regular 4 > 4xB200 3 = 4xH200 3 > 4xH100 2 = 4xA100 2 > 2xA100 = 4xL40S 1; a faster member that starts
# requeues a slower running one, so only one trains; equal ranks: the lower job id keeps the run. Members added 10/25:
# a100x4 (4xA100), h200 (4xH200, preemptible) and h200t: 4xH200 on the regular h200_dept partition with the
# <advanced-account> account (not preemptible, so it outranks everything: nothing takes a 12-h allocation
# away). H200 (141 GB): micro 8 with full sharding, no gradient checkpointing. h100t: 4xH100 on the regular
# h100_dept partition with the <priority-account> account, rank 4 as well (the earlier h200t keeps the run).
# h200t2 / h100t2: 2xH200 (<account-advanced>) / 2xH100 (<account-priority>) on the same regular partitions, micro 8 with
# gradient checkpointing + full sharding like the 2xA100; rank 2 (fewer GPUs, slower: a 4-GPU member, preemptible or
# not, takes over). The dept partitions have group GPU limits (QOS h200_dept 112, h100_dept 60).
# 10/25 22:25 (user): members on regular, non-preemptible partitions (PreemptMode=OFF) outrank every preemptible one
# and are replaced only by a faster non-preemptible member: 4xH200 15 > 4xH100 14 > 2xH200 13 > 2xH100 12 > 4xA100 11 >
# 2xA100 10; the preemptible ranks below stay 1-5 (an allocation that cannot be preempted is not traded for one that can).
# b200t: 4xB200 preemptible with the <advanced-account> account, which lands on the dept group's own
# b200_dept_prem partition and may preempt guest jobs there (sbatch --test-only 10/25: "Preempts ..."); rank 5, the
# top: fastest card and owner priority on those nodes. a100t2 / a100t4: 2x / 4xA100 on the regular a100_dept partition
# with the <priority-account> account, rank 2, LOWMEM micro 8 (like the general A100 members).
# b200t2 / b2002: 2xB200 preemptible with the <account-advanced> (b200_dept_prem) / general account, rank 3, micro 16
# with full sharding: 2 GPUs fit the per-user limit of the gpu48 QOS (0-2 GPUs on 10/25) that holds the 4-GPU jobs back.
# Global batch 128 everywhere: B200 micro 16 x 4 x 2 (as the +8k leg, 1.9 s/step); H100 micro 16 and the small
# members (A100 micro 8, L40S micro 4) with gradient checkpointing + full sharding.
# Checkpoints every SAVE (1000; 200 on A100 / L40S) steps; the dev trainer deletes older ones except the newest with an
# hf_ckpt and any holding a file named "keep": touch <run>/checkpoints/global_step_N/keep before evaluating step N.
#   scripts/hpc/submit_plain_cont.sh                 # MEMBERS="b200 h100 a100 l40s" by default
#   RESUBMIT=1 MEMBERS="h200 h200t" SIDECAR=0 scripts/hpc/submit_plain_cont.sh   # add members to the running group
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
W="${PWD}"
source scripts/hpc/env.sh
DEV="${PROJ}/dev/lingbot-vla-v2-dino"
INIT="${RUN_DIR}/strong_scratch22k_cont8k/checkpoints/global_step_8000/hf_ckpt"
RUN="${RUN:-s22k8k_v3uni_15k}"
MEMBERS="${MEMBERS:-b200 h100 a100 l40s}"
[[ -f "${INIT}/model.safetensors.index.json" ]] || { echo "missing ${INIT}" >&2; exit 1; }
grep -q "_lb_reseed" "${DEV}/tasks/vla/train_lingbotvla.py" || { echo "dev trainer lacks the per-rank reseed" >&2; exit 1; }
[[ -n "${RESUBMIT:-}" || ! -e "${RUN_DIR}/${RUN}" ]] || { echo "${RUN_DIR}/${RUN} exists; RESUBMIT=1 to add members" >&2; exit 1; }

export LINGBOT_DIR="${DEV}" APPLY_PATCH=0 HANDOFF=0 PYTHONPATH="${DEV}:${W}/src" \
       LINGBOT_AUG_CONFIG="${W}/configs/aug_strong_v3.json" LINGBOT_SEEN_POOL=1 LINGBOT_STATE_NOISE=0.01 \
       LINGBOT_VISION_LR_SCALE=0.1 LINGBOT_TASK_SAMPLING=uniform LINGBOT_FRAME_WEIGHTS=1
ARGS="--model.model_path ${INIT} --data.image_augment true --train.lr 5e-6 --train.lr_min 2e-6 --train.lr_decay_ratio 0.75 --train.lr_warmup_ratio 0.03"
# BRANCH_FROM=<run>: this RUN is a branch of <run>: its first member to train hard-links <run>'s newest complete checkpoint
# into RUN (scripts/hpc/branch_seed.sh via train.sbatch PRE_TRAIN_HOOK) and resumes there (weights, optimizer, lr
# schedule, step); later starts resume from the branch's own checkpoints. BRANCH_SEED (43) keeps its data draws apart
# from the source run's. 10/25: the B200 members run as branch s22k8k_v3uni_15k_b200 next to the 2xH100 main run; the hook
# runs at every start and takes the source's checkpoint whenever it is further than the branch's own.
if [[ -n "${BRANCH_FROM:-}" ]]; then
  [[ -d "${RUN_DIR}/${BRANCH_FROM}/checkpoints" ]] || { echo "no checkpoints in ${RUN_DIR}/${BRANCH_FROM}" >&2; exit 1; }
  export PRE_TRAIN_HOOK="bash ${W}/scripts/hpc/branch_seed.sh ${RUN_DIR}/${BRANCH_FROM} ${RUN_DIR}/${RUN}"
  ARGS="${ARGS} --train.seed ${BRANCH_SEED:-43}"
fi
# ADOPT_FROM=<run>: every start of a member of RUN first takes <run>'s newest complete checkpoint if it is further than
# RUN's own (same hook, without the seed change); 10/25: the main run's members adopt the B200 branch when it is ahead.
if [[ -n "${ADOPT_FROM:-}" ]]; then
  export PRE_TRAIN_HOOK="bash ${W}/scripts/hpc/branch_seed.sh ${RUN_DIR}/${ADOPT_FROM} ${RUN_DIR}/${RUN}"
fi
LOWMEM="--train.enable_gradient_checkpointing true --train.enable_full_shard true"
submit() {  # time gpu_opts...
  local time=$1; shift
  RUN_NAME=${RUN} MAX_STEPS=${MAX_STEPS:-15000} SAVE_STEPS=${SAVE:-1000} MICRO_BS=${MICRO:-16} GLOBAL_BS=128 \
    TRAIN_ARGS="${ARGS} ${EXTRA_TRAIN_ARGS:-}" PREEMPT="${PREEMPT:-1}" \
    scripts/hpc/submit.sh scripts/hpc/train.sbatch --parsable --job-name="lb_${RUN}" --time="${time}" "$@" | tail -1
}
has() { [[ " ${MEMBERS} " == *" $1 "* ]]; }
B=none; H=none; A=none; A4=none; L=none; H2=none; H2T=none; H1T=none; H2T2=none; H1T2=none; BT=none; AT2=none; AT4=none; BT2=none; B2=none
if has b200; then B=$(HANDOFF_RANK=3 submit 12:00:00 --gres=gpu:b200:4 --constraint=b200 --cpus-per-task=48 --mem=400G); fi
if has h100; then H=$(HANDOFF_RANK=2 EXTRA_TRAIN_ARGS="${LOWMEM}" \
                      submit 12:00:00 --gres=gpu:4 --constraint=h100 --cpus-per-task=48 --mem=400G); fi
if has a100; then A=$(HANDOFF_RANK=1 SAVE=200 MICRO=8 EXTRA_TRAIN_ARGS="${LOWMEM}" \
                      submit 12:00:00 --gres=gpu:2 --constraint=a100 --cpus-per-task=24 --mem=200G); fi
if has a100x4; then A4=$(HANDOFF_RANK=2 SAVE=500 MICRO=8 EXTRA_TRAIN_ARGS="${LOWMEM}" \
                      submit 12:00:00 --gres=gpu:a100:4 --constraint=a100 --cpus-per-task=48 --mem=400G); fi
if has h200; then H2=$(HANDOFF_RANK=3 MICRO=8 EXTRA_TRAIN_ARGS="--train.enable_full_shard true" \
                      submit 12:00:00 --gres=gpu:h200:4 --constraint=h200 --cpus-per-task=48 --mem=400G); fi
if has h200t; then H2T=$(HANDOFF_RANK=15 MICRO=8 EXTRA_TRAIN_ARGS="--train.enable_full_shard true" PREEMPT=0 \
                        SLURM_ACCOUNT="${ACCOUNT_ADVANCED:?set ACCOUNT_ADVANCED}" \
                        submit 12:00:00 --gres=gpu:h200:4 --constraint=h200 --cpus-per-task=48 --mem=400G); fi
if has h100t; then H1T=$(HANDOFF_RANK=14 EXTRA_TRAIN_ARGS="${LOWMEM}" PREEMPT=0 SLURM_ACCOUNT="${ACCOUNT_PRIORITY:?set ACCOUNT_PRIORITY}" \
                        submit 12:00:00 --gres=gpu:h100:4 --constraint=h100 --cpus-per-task=48 --mem=400G); fi
if has h200t2; then H2T2=$(HANDOFF_RANK=13 SAVE=500 MICRO=8 EXTRA_TRAIN_ARGS="${LOWMEM}" PREEMPT=0 \
                          SLURM_ACCOUNT="${ACCOUNT_ADVANCED:?set ACCOUNT_ADVANCED}" \
                          submit 12:00:00 --gres=gpu:h200:2 --constraint=h200 --cpus-per-task=24 --mem=200G); fi
if has h100t2; then H1T2=$(HANDOFF_RANK=12 SAVE=500 MICRO=8 EXTRA_TRAIN_ARGS="${LOWMEM}" PREEMPT=0 \
                          SLURM_ACCOUNT="${ACCOUNT_PRIORITY:?set ACCOUNT_PRIORITY}" \
                          submit 12:00:00 --gres=gpu:h100:2 --constraint=h100 --cpus-per-task=24 --mem=200G); fi
if has b200t; then BT=$(HANDOFF_RANK=5 SLURM_ACCOUNT="${ACCOUNT_ADVANCED:?set ACCOUNT_ADVANCED}" \
                       submit 12:00:00 --gres=gpu:b200:4 --constraint=b200 --cpus-per-task=48 --mem=400G); fi
if has a100t2; then AT2=$(HANDOFF_RANK=10 SAVE=200 MICRO=8 EXTRA_TRAIN_ARGS="${LOWMEM}" PREEMPT=0 \
                         SLURM_ACCOUNT="${ACCOUNT_PRIORITY:?set ACCOUNT_PRIORITY}" \
                         submit 12:00:00 --gres=gpu:a100:2 --constraint=a100 --cpus-per-task=24 --mem=200G); fi
if has a100t4; then AT4=$(HANDOFF_RANK=11 SAVE=500 MICRO=8 EXTRA_TRAIN_ARGS="${LOWMEM}" PREEMPT=0 \
                         SLURM_ACCOUNT="${ACCOUNT_PRIORITY:?set ACCOUNT_PRIORITY}" \
                         submit 12:00:00 --gres=gpu:a100:4 --constraint=a100 --cpus-per-task=48 --mem=400G); fi
if has b200t2; then BT2=$(HANDOFF_RANK=3 SAVE=500 MICRO=16 EXTRA_TRAIN_ARGS="--train.enable_full_shard true" \
                         SLURM_ACCOUNT="${ACCOUNT_ADVANCED:?set ACCOUNT_ADVANCED}" \
                         submit 12:00:00 --gres=gpu:b200:2 --constraint=b200 --cpus-per-task=24 --mem=200G); fi
if has b2002; then B2=$(HANDOFF_RANK=3 SAVE=500 MICRO=16 EXTRA_TRAIN_ARGS="--train.enable_full_shard true" \
                       submit 12:00:00 --gres=gpu:b200:2 --constraint=b200 --cpus-per-task=24 --mem=200G); fi
if has l40s; then L=$(HANDOFF_RANK=1 SAVE=200 MICRO=4 EXTRA_TRAIN_ARGS="${LOWMEM}" \
                      submit 12:00:00 --gres=gpu:l40s:4 --constraint=l40s --cpus-per-task=48 --mem=400G); fi
unset HANDOFF APPLY_PATCH PRE_TRAIN_HOOK PYTHONPATH LINGBOT_AUG_CONFIG LINGBOT_SEEN_POOL LINGBOT_STATE_NOISE LINGBOT_VISION_LR_SCALE \
      LINGBOT_TASK_SAMPLING LINGBOT_FRAME_WEIGHTS
[[ "${SIDECAR:-1}" == 1 ]] && RUN_NAME=${RUN} TRAIN_JOB_NAME="lb_${RUN}" scripts/hpc/submit.sh --cpu scripts/hpc/wandb_sync.sbatch >/dev/null
echo "${RUN}: b200=${B} h100=${H} a100=${A} a100x4=${A4} h200=${H2} h200_dept=${H2T} h100_dept=${H1T} h200_dept_x2=${H2T2} h100_dept_x2=${H1T2} b200_dept_prem=${BT} a100_dept_x2=${AT2} a100_dept_x4=${AT4} b200_dept_x2=${BT2} b200_x2=${B2} l40s=${L} (job name lb_${RUN}, run dir ${RUN_DIR}/${RUN})"
