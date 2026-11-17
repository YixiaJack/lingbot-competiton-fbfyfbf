#!/bin/bash
# stop v1, norm stats over the v2 list, launch v2 (same config as v1, data + task-1 box cues)
W=$HOME/lingbot_train; cd $W; . tools/env.sh
LOG=$W/logs/launch_v2.log; exec >> $LOG 2>&1
echo "== $(date -Is) stop v1 at step $(tr '\r' '\n' < runs/v1/train.log | grep -c StepTime)"
kill $(cat runs/v1/pid); for i in $(seq 1 60); do n=$(pgrep -f train_lingbotvla.py | wc -l); [ "$n" = 0 ] && break; sleep 2; done; echo "train procs left: $n"
mv runs/v1 runs/v1_stopped_for_v2
cp rc/agilex_cobot_magic.json rc/agilex_cobot_magic.v1_norm.json
printf "agilex_cobot_magic $W/data/public_t12\nagilex_cobot_magic $W/data/public_t3_sub\nagilex_cobot_magic $W/data/public_t1_box\n" > lists/v2.txt
echo "== $(date -Is) norm stats"
torchrun --nproc-per-node 8 --master-port 62520 $LB/scripts/compute_norm_stats.py $LB/configs/vla/norm_compute/post_data.yaml \
  --data.robot_name agilex_cobot_magic --data.train_path $W/lists/v2.txt --data.norm_path $W/rc/agilex_cobot_magic.v2_norm.json \
  --data.robot_config_root $W/rc --train.output_dir $W/runs/norm_v2 > logs/norm_v2.log 2>&1
[ -s rc/agilex_cobot_magic.v2_norm.json ] || { echo "norm failed"; exit 1; }
cp rc/agilex_cobot_magic.v2_norm.json rc/agilex_cobot_magic.json
python tools/mkcfg.py configs/v2.yaml data.train_path=$W/lists/v2.txt train.max_steps=6000 train.save_steps=500 train.save_epochs=1000000 \
  train.micro_batch_size=16 train.global_batch_size=256 train.gradient_accumulation_steps=2
echo "== $(date -Is) launch v2"
LB_VIT_BATCHED_SDPA=1 LB_GRID_CACHE=1 LB_FSDP_VISION=1 FORGE_CKPT_KEEP_DCP=1 CODE=$W/code_opt tools/train_run.sh v2 configs/v2.yaml 0,1,2,3,4,5,6,7
(cd code_opt && git log --oneline -1) > runs/v2/code_commit.txt
echo "LB_VIT_BATCHED_SDPA=1 LB_GRID_CACHE=1 LB_FSDP_VISION=1 FORGE_CKPT_KEEP_DCP=1" > runs/v2/env.txt
cp lists/v2.txt runs/v2/; cp rc/agilex_cobot_magic.v2_norm.json runs/v2/norm_stats.json
echo "== $(date -Is) done"
