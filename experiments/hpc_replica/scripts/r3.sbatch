#!/bin/bash
#SBATCH --job-name=lb_devrep_r3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
# R3: open-loop eval of the slim hf_ckpt (R2 global_step_16) with the open-source deploy policy (strict weight load).
R=$SCRATCH/lingbot/devrep_20261114; cd $R
for j in $(squeue -u "$USER" -h -n lb_devrep_r3 -o %i); do [ "$j" != "$SLURM_JOB_ID" ] && scancel "$j"; done
H=$R/runs/R2/checkpoints/global_step_16/hf_ckpt
echo "job $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > runs/R3/job.txt
cp $H/lingbotvla_cli.yaml runs/R2/lingbotvla_cli.yaml          # deploy reads <hf_ckpt>/../../../lingbotvla_cli.yaml
mkdir -p code_aug/configs/robot_configs
sed "s#^norm_stats: .*#norm_stats: $H/norm_states.json#" $H/robot_config.yaml > code_aug/configs/robot_configs/agilex_cobot_magic.yaml
cd code_aug
PYTHONPATH=$R/pyextra:$R/code_aug:$R/aux timeout 40m bash $SCRATCH/lingbot/LingBotRoboTwin/scripts/hpc/run_in_env.sh \
  python scripts/open_loop_eval.py --policy qwen3vl --model_path $H --robo_name agilex_cobot_magic \
  --norm_path $H/norm_states.json --data_path $R/data/public_t12 --traj_ids 0 1 --max_infer_time 3 --use_bf16 \
  --save_plot_path $R/runs/R3 > $R/runs/R3/eval.log 2>&1
echo "R3 exit $?" >> $R/runs/R3/job.txt
