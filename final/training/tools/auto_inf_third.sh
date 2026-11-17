#!/bin/bash
# 11/16 (data session): once the round-3 inference snapshot is uploaded (/workspace/上传OSS_inference_third, md5-checked)
# AND a PART=weights run has put a checkpoint into /workspace/submit_result/third, run PART=inf for the newest checkpoint
# that is in there (README variant to match). ALLOW_WEIGHT_MISMATCH=1: the 5080 still had 11500 loaded when the snapshot
# was taken (04:38); it is switched to the submitted checkpoint before round 3, and the README says so.
# log: /data/lingbot/tmp_x/auto_inf_third.log
cd ~/lingbot_train/tools || exit 1
U=/workspace/上传OSS_inference_third
ok() { [ -f "$U/$1" ] && [ -f "$U/$1.md5" ] && [ "$(dd if="$U/$1" bs=16M status=none | md5sum | cut -c1-32)" = "$(cut -c1-32 "$U/$1.md5")" ]; }
echo "$(date +%T) waiting for the upload in $U"
until ok lingbot_inference_third.tar && ok data_prep_1115.tar && ok graspnet_src.tar; do sleep 15; done
echo "$(date +%T) upload complete"
while :; do
  last=""; for O in 15200 15100 15000; do grep -q "make_submit third done" /data/lingbot/tmp_x/ms_third_w$O.out 2>/dev/null && { last=$O; break; }; done
  [ -n "$last" ] && ! pgrep -f "[m]ake_submit.sh third" >/dev/null && break; sleep 15
done
V=$((last - 2000)); C=/data/lingbot/runs/v4/checkpoints/global_step_$V/hf_ckpt
RS=$HOME/lingbot_train/submit_src/README_third_$last.md; [ -f $RS ] || RS=$HOME/lingbot_train/submit_src/README_third.md
echo "$(date +%T) PART=inf with $C (overall $last), README $RS"
ALLOW_WEIGHT_MISMATCH=1 README_SRC=$RS PART=inf bash make_submit.sh third $C $HOME/lingbot_train/lists/v4za.txt > /data/lingbot/tmp_x/ms_third_inf.out 2>&1; rc=$?
echo "$(date +%T) rc=$rc $(tail -1 /data/lingbot/tmp_x/ms_third_inf.out)"
