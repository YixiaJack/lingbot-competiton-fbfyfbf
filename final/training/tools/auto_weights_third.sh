#!/bin/bash
# 11/16 (data session): put each round-3 checkpoint into /workspace/submit_result/third as soon as its hf_ckpt is complete
# (PART=weights: weights + SHA-256, logs, README variant), in order 15000 -> 15100 -> 15200, so the folder always holds
# the newest verified one. Waits while another make_submit runs. log: /data/lingbot/tmp_x/auto_weights_third.log
cd ~/lingbot_train/tools || exit 1
for pair in "13000 15000" "13100 15100" "13200 15200"; do
  set -- $pair; V=$1; O=$2; C=/data/lingbot/runs/v4/checkpoints/global_step_$V/hf_ckpt
  echo "$(date +%T) waiting for $C (overall $O)"
  while :; do
    if [ -f $C/lingbotvla_cli.yaml ] && [ -f $C/config.json ]; then
      newest=$(find $C -type f -printf '%T@\n' | sort -n | tail -1); [ $(( $(date +%s) - ${newest%.*} )) -ge 30 ] && break
    fi; sleep 10
  done
  while pgrep -f "[m]ake_submit.sh third" >/dev/null; do sleep 10; done
  RS=$HOME/lingbot_train/submit_src/README_third_$O.md; [ -f $RS ] || RS=$HOME/lingbot_train/submit_src/README_third.md
  echo "$(date +%T) $O: PART=weights from $C, README $RS"
  README_SRC=$RS PART=weights bash make_submit.sh third $C $HOME/lingbot_train/lists/v4za.txt > /data/lingbot/tmp_x/ms_third_w$O.out 2>&1; rc=$?
  echo "$(date +%T) $O: rc=$rc $(tail -1 /data/lingbot/tmp_x/ms_third_w$O.out)"
done
echo "$(date +%T) all done"
