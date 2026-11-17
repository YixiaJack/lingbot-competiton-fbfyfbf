#!/bin/bash
#SBATCH --job-name=lb_devrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
# Dev-machine replica checks (items 1-2): timing breakdown (T0-T3) and restart path (R1 save, R2 resume on new data +
# slim checkpoint, R3 open-loop eval of the slim hf_ckpt). Several alternatives are queued; the first to start cancels
# the rest. All cases run sequentially on this one node so the timings are comparable.
R=$SCRATCH/lingbot/devrep_20261114
cd $R; mkdir -p results runs cfg
for j in $(squeue -u "$USER" -h -n lb_devrep -o %i); do [ "$j" != "$SLURM_JOB_ID" ] && scancel "$j"; done
NGPU=$(nvidia-smi -L | wc -l); GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is): $NGPU x $GPU" | tee results/job_info.txt
nvidia-smi topo -m > results/topo.txt 2>&1
RIE=$SCRATCH/lingbot/LingBotRoboTwin/scripts/hpc/run_in_env.sh
# numpy 1.26.4 / pyarrow 19.0.1 as on the dev machine, if the env's compiled packages accept them
if PYTHONPATH=$R/pyextra bash $RIE python -c "import numpy, pyarrow, pandas, cv2, torch, transformers, lerobot; print('pins ok', numpy.__version__, pyarrow.__version__)" > results/pins.txt 2>&1; then
  export PYEXTRA=$R/pyextra; else echo "pins rejected, using env numpy/pyarrow" >> results/pins.txt; fi
GB=$((16 * NGPU * 2))
B="micro_batch_size=16 gradient_accumulation_steps=2 global_batch_size=$GB"
bash $RIE python - <<PY > results/cfg.txt 2>&1
import subprocess, sys
R = '$R'; src = R + '/aux/configs/v2.yaml'; mk = R + '/mkcfg_hpc.py'; b = '$B'.split()
def mk_(out, lst, *kv): subprocess.run([sys.executable, mk, src, R + '/cfg/' + out, R + '/lists/' + lst] + ['train.' + x for x in b] + list(kv), check=True)
mk_('timing.yaml', 'v2like.txt', 'train.max_steps=25', 'train.save_steps=100000')
mk_('timing_noalign.yaml', 'v2like.txt', 'train.max_steps=25', 'train.save_steps=100000', 'train.align_params=DELETE')
mk_('r1.yaml', 'A.txt', 'train.max_steps=10', 'train.save_steps=5')
mk_('r2.yaml', 'B.txt', 'train.max_steps=16', 'train.save_steps=8', 'train.resume_dataloader_state=false',
    'train.load_checkpoint_path=' + R + '/runs/R1/checkpoints/global_step_10')
PY
V2="LB_VIT_BATCHED_SDPA=1 LB_GRID_CACHE=1 LB_FSDP_VISION=1 LB_VIT_FORCE_SDPA=1"
AUG="LINGBOT_AUG_CONFIG=$R/aux/aug_whitelist.json"
run() { echo "== $(date -Is) $1"; env $2 bash $R/run_case.sh $1 $3 $R/cfg/$4 $NGPU "${@:5}" | tee -a results/cases.txt; }
run T0 "$V2 LB_TIME_BREAKDOWN=1 LB_PROFILE_STEPS=18,19" $R/code_opt timing.yaml
run T1 "$V2 LB_TIME_BREAKDOWN=1" $R/code_opt timing.yaml --data.image_augment true
run T2 "$V2 LB_TIME_BREAKDOWN=1 $AUG" $R/code_aug timing.yaml --data.image_augment true
run T3 "$V2 LB_TIME_BREAKDOWN=1" $R/code_opt timing_noalign.yaml
run R1 "$V2 $AUG FORGE_CKPT_KEEP_DCP=1" $R/code_aug r1.yaml --data.image_augment true
run R2 "$V2 $AUG" $R/code_aug r2.yaml --data.image_augment true
for d in runs/R1/checkpoints/* runs/R2/checkpoints/*; do echo "$d: $(ls $d | tr '\n' ' ') $(du -sh $d | cut -f1)"; done > results/ckpt_dirs.txt 2>&1
CK=$(ls -d runs/R2/checkpoints/global_step_*/hf_ckpt 2>/dev/null | tail -1)
echo "== $(date -Is) R3 open-loop on $CK" | tee -a results/cases.txt
mkdir -p runs/R3
PYTHONPATH=${PYEXTRA:+$PYEXTRA:}$R/code_aug:$R/aux timeout 30m bash $RIE python $R/code_aug/scripts/open_loop_eval.py --model_path "$R/$CK" \
  --robo_name agilex_cobot_magic --norm_path $R/aux/rc/agilex_cobot_magic.json --data_path $R/data/public_t12 \
  --traj_ids 0 1 --max_infer_time 3 --use_bf16 --save_plot_path $R/runs/R3 > runs/R3/eval.log 2>&1
echo "R3 exit $?" | tee -a results/cases.txt
echo "done $(date -Is)" | tee -a results/cases.txt
