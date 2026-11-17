#!/usr/bin/env bash
# LingBot-VLA 2.0 post-training on the 50 RoboTwin clean datasets (single node, FSDP2, torchrun).
#   RUN_NAME=smoke MAX_STEPS=30 SAVE_STEPS=20 scripts/hpc/submit.sh scripts/hpc/train.sbatch --gres=gpu:4 --time=02:00:00
#   RUN_NAME=main  MAX_STEPS=20000 SAVE_STEPS=1000 scripts/hpc/submit.sh scripts/hpc/train.sbatch --gres=gpu:4 --time=48:00:00
# Resumes from the newest checkpoint in runs/$RUN_NAME (enable_resume), so a preempted + requeued job continues.
# Any extra upstream flags go in TRAIN_ARGS, e.g. TRAIN_ARGS="--train.micro_batch_size 16".
#SBATCH --job-name=lb_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100|h200
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/lingbot/logs/train_%x_%j.log
set -euo pipefail
cd "${WORK_DIR:-/scratch/${USER}/lingbot/LingBotRoboTwin}"
source scripts/hpc/env.sh
RUN_NAME="${RUN_NAME:?RUN_NAME required}"
OUT="${RUN_DIR}/${RUN_NAME}"
mkdir -p "${OUT}"
NGPU=$(nvidia-smi -L | wc -l)
# MICRO_BS=auto: 32 per GPU on >=120 GB cards (B200, H200), 16 on 80 GB cards (H100, A100); the global batch is
# kept by gradient accumulation, which upstream does not infer (global must equal micro * dp * accum).
GPU_MEM_MB=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits)  # no "| head": SIGPIPE + pipefail kills the job
MICRO_BS="${MICRO_BS:-auto}"
if [[ "${MICRO_BS}" == auto ]]; then MICRO_BS=$(( GPU_MEM_MB >= 120000 ? 32 : 16 )); fi
GLOBAL_BS="${GLOBAL_BS:-$(( MICRO_BS * NGPU ))}"
GRAD_ACC=$(( GLOBAL_BS / (MICRO_BS * NGPU) ))
(( GRAD_ACC >= 1 && GRAD_ACC * MICRO_BS * NGPU == GLOBAL_BS )) || { echo "global ${GLOBAL_BS} not divisible by micro ${MICRO_BS} x ${NGPU} GPUs"; exit 1; }
echo "=== train ${RUN_NAME} on $(hostname) $(date) | ${NGPU} GPUs: $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)"
echo "    micro ${MICRO_BS} x ${NGPU} GPUs x accum ${GRAD_ACC} = global ${GLOBAL_BS} max_steps ${MAX_STEPS:?} save_steps ${SAVE_STEPS:?} | requeue count ${SLURM_RESTART_COUNT:-0}"
env | grep -E "^(LINGBOT_|PYTHONPATH=)" | sed "s/^/    /" || true
[[ -f configs/norm_stats/robotwin_clean.json.computed ]] || { echo "clean norm stats missing (norm_stats.sbatch)"; exit 1; }

export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4
# HANDOFF=1: this job takes over RUN_NAME from the other jobs with the same --job-name (e.g. a 4xA100 job keeps the run
# going until 4xH100/B200 frees up). It cancels them (running or pending), waits until they are gone, then resumes from
# the newest checkpoint; upstream enable_resume falls back to an older one if a checkpoint was cut off mid-save.
# All jobs of such a group must use the same RUN_NAME and GLOBAL_BS (lr schedule, dataloader state); MICRO_BS may differ
# (each rank's dataloader yields GLOBAL_BS / NGPU samples per step, which training splits into micro batches).
if [[ "${HANDOFF:-0}" == 1 ]]; then
  SIBLINGS=$(squeue -h -u "${USER}" -n "${SLURM_JOB_NAME}" -o %i | grep -vx "${SLURM_JOB_ID}" | tr '\n' ' ' || true)
  if [[ -n "${SIBLINGS// /}" ]]; then
    echo "    handoff: cancelling ${SIBLINGS}"
    scancel ${SIBLINGS} || true
    for _ in $(seq 1 90); do
      ALIVE=$(squeue -h -u "${USER}" -n "${SLURM_JOB_NAME}" -t R,CG -o %i | grep -vx "${SLURM_JOB_ID}" || true)
      [[ -z "${ALIVE}" ]] && break
      sleep 10
    done
    echo "    handoff: done $(date)"
  fi
fi
# HANDOFF_RANK=<n>: ranked version of HANDOFF for a group with fast and slow members (e.g. B200 3, H100 2, A100/L40S 1).
# On start the job records its rank in <run>/.handoff/<jobid>, then looks at the running siblings: if one has a higher
# rank (or the same rank and a lower job id) it keeps the run, so this job puts itself back in the queue (scontrol
# requeue) and stops; otherwise it requeues the lower-ranked running siblings (they wait in the queue as backups),
# waits until they are gone and resumes from the newest checkpoint. Pending siblings are never touched, so a
# preempted fast job always has queued backups. Two members never train the same run at once.
RESUME_NOTE="${OUT}/.resume_failed/${SLURM_JOB_ID}"  # written by the DINO dev checkout when no checkpoint loads
if [[ -n "${HANDOFF_RANK:-}" ]]; then
  # a member that failed to resume from the current newest checkpoint does not take the run over again (it would
  # requeue the running backup each time it starts) until another member has saved a newer one
  NEWEST=$(ls "${OUT}/checkpoints" 2>/dev/null | grep -E '^global_step_[0-9]+$' | sort -t_ -k3 -n | tail -1 || true)
  if [[ -f "${RESUME_NOTE}" && -n "${NEWEST}" && "$(tail -1 "${RESUME_NOTE}" | cut -d' ' -f1)" == "${NEWEST}" ]]; then
    echo "    resume failed on ${NEWEST} before: back to the queue until a newer checkpoint exists $(date)"
    scontrol requeue "${SLURM_JOB_ID}"; sleep 300; exit 0
  fi
  HD="${OUT}/.handoff"; mkdir -p "${HD}"; echo "${HANDOFF_RANK}" > "${HD}/${SLURM_JOB_ID}"
  sleep 20  # let a sibling that started at the same moment write its rank too
  RUNNING=$(squeue -h -u "${USER}" -n "${SLURM_JOB_NAME}" -t R,CG -o %i | grep -vx "${SLURM_JOB_ID}" || true)
  LOWER=()
  for j in ${RUNNING}; do
    r=$(cat "${HD}/${j}" 2>/dev/null || echo 99)  # unknown rank (old-style member): assume it outranks us
    if (( r > HANDOFF_RANK )) || { (( r == HANDOFF_RANK )) && (( j < SLURM_JOB_ID )); }; then
      echo "    handoff: sibling ${j} (rank ${r}) keeps the run; back to the queue $(date)"
      rm -f "${HD}/${SLURM_JOB_ID}"; scontrol requeue "${SLURM_JOB_ID}"; sleep 300; exit 0
    fi
    LOWER+=("${j}")
  done
  for j in ${LOWER[@]+"${LOWER[@]}"}; do echo "    handoff: requeue lower-ranked sibling ${j}"; scontrol requeue "${j}" || scancel "${j}"; done
  for _ in $(seq 1 90); do
    ALIVE=""; for j in ${LOWER[@]+"${LOWER[@]}"}; do squeue -h -j "${j}" -t R,CG 2>/dev/null | grep -q . && ALIVE+=" ${j}"; done
    [[ -z "${ALIVE}" ]] && break; sleep 10
  done
  echo "    handoff (rank ${HANDOFF_RANK}): running $(date)"
fi
# PRE_TRAIN_HOOK: run once this job holds the run, before training (e.g. scripts/hpc/branch_seed.sh, which seeds a branch
# run from another run's newest checkpoint); if it fails, back to the queue instead of training from scratch
if [[ -n "${PRE_TRAIN_HOOK:-}" ]] && ! bash -c "${PRE_TRAIN_HOOK}"; then
  echo "    pre-train hook failed: back to the queue $(date)"; scontrol requeue "${SLURM_JOB_ID}"; sleep 300; exit 0
fi
# APPLY_PATCH=1: apply this repo's env-gated upstream changes first (idempotent; src/patch_upstream.py)
if [[ "${APPLY_PATCH:-0}" == 1 ]]; then python3 "${WORK_DIR}/src/patch_upstream.py" "${LINGBOT_DIR}"; fi
cd "${LINGBOT_DIR}"
run_train() {
  "${WORK_DIR}/scripts/hpc/run_in_env.sh" torchrun --nnodes=1 --nproc-per-node "${NGPU}" \
    --master-addr=127.0.0.1 --master-port=$(( 20000 + RANDOM % 20000 )) \
    tasks/vla/train_lingbotvla.py "${WORK_DIR}/configs/robotwin_clean.yaml" \
    --data.norm_stats_file "${WORK_DIR}/configs/norm_stats/robotwin_clean.json" \
    --train.output_dir "${OUT}" \
    --train.micro_batch_size "${MICRO_BS}" --train.global_batch_size "${GLOBAL_BS}" \
    --train.gradient_accumulation_steps "${GRAD_ACC}" \
    --train.max_steps "${MAX_STEPS}" --train.save_steps "${SAVE_STEPS}" \
    ${TRAIN_ARGS:-}
}
RC=0; run_train || RC=$?
echo "=== train exit ${RC} $(date)"
# no checkpoint could be loaded: the training script (DINO dev checkout) does not start from scratch by itself but
# leaves its decision in RESUME_NOTE. requeue: hand the run back to the other members (one may read a checkpoint saved
# with another GPU count); restart: keep the unreadable checkpoints aside and train again from the run's initial model
# (DINO: 22k+8k + zero-initialized fusion) in this allocation
if (( RC != 0 )) && [[ -f "${RESUME_NOTE}" ]]; then
  if [[ "$(tail -1 "${RESUME_NOTE}" | cut -d' ' -f2)" == restart ]]; then
    ASIDE="${OUT}/checkpoints_unreadable_$(date +%m%d_%H%M%S)"
    echo "    no checkpoint readable: ${OUT}/checkpoints -> ${ASIDE}, training again from the initial model $(date)"
    mv "${OUT}/checkpoints" "${ASIDE}"; mv "${OUT}/.resume_failed" "${ASIDE}/resume_failed_notes"
    RC=0; run_train || RC=$?
    echo "=== train exit ${RC} $(date)"
  else
    echo "    resume failed: back to the queue"; scontrol requeue "${SLURM_JOB_ID}"; sleep 300; exit 0
  fi
fi
exit "${RC}"
