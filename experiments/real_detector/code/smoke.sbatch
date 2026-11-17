#!/bin/bash
# #SBATCH lines are not shell-expanded: replace <slurm-account-general> and $SCRATCH below with real values.
#SBATCH --job-name=lb_real_smoke_1112
#SBATCH --account=<slurm-account-general>
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --comment=preemption=yes;preemption_partitions_only=yes;requeue=true
#SBATCH --output=$SCRATCH/lingbot/final_real_20261112/logs/%x_%j.log
R=$SCRATCH/lingbot/final_real_20261112
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash $R/code/runtime.sh sim python $R/code/smoke_real.py $R/smoke/${SLURM_JOB_ID} ${N1:-12} ${N2:-9}
