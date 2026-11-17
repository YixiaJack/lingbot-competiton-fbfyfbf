#!/bin/bash
# collect_5080_inference.sh ROUND [--dry-run]      (run on the Mac; ROUND = first | second | third)
# Snapshot of the inference code that actually runs on the 5080 (<infer-user>@<inference-host>), for that round's submission:
#   ~/Desktop/上传OSS_inference_<ROUND>/lingbot_inference_<ROUND>.tar (+ .md5)
# Upload that folder to the OSS workspace as usual (it shows up as /workspace/上传OSS_inference_<ROUND>/ on the dev
# machine), then on the dev machine: ~/lingbot_train/tools/make_submit.sh <ROUND> <HF_CKPT> <LIST>.
# Take the snapshot with the configuration the evaluation will run with (model loaded, cue_prompt, scheduler ...): the
# guide requires model and inference config to stay fixed during a round, and make_submit.sh checks that the weights
# loaded on the 5080 are the HF_CKPT being submitted. Contents: see collect_5080_remote.sh.
# --dry-run builds the snapshot on the 5080, prints the summary and deletes it (nothing downloaded).
# --model=<hf_ckpt path on the 5080> names the round's checkpoint when it is deployed but not loaded yet (default: the loaded one).
R=${1:?round: first, second or third}; DRY=0; MODEL=""
for a in "${@:2}"; do case $a in --dry-run) DRY=1 ;; --model=*) MODEL=${a#--model=} ;; esac; done
case $R in first|second|third) ;; *) echo "round must be first, second or third"; exit 1 ;; esac
H=<infer-user>@<inference-host>; HERE=$(cd "$(dirname "$0")" && pwd); SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
scp -q -o BatchMode=yes "$HERE/collect_5080_remote.sh" $H:/tmp/collect_5080_remote.sh || { echo "cannot reach the 5080"; exit 1; }
out=$($SSH $H "nice -n 19 bash /tmp/collect_5080_remote.sh $R $DRY $MODEL; rc=\$?; rm -f /tmp/collect_5080_remote.sh; exit \$rc"); rc=$?
echo "$out"
[ $rc = 0 ] || { echo "snapshot failed (exit $rc)"; exit $rc; }
[ $DRY = 1 ] && exit 0
TAR=$(echo "$out" | sed -n 's/^TAR //p'); MD5=$(echo "$out" | sed -n 's/^MD5 //p')
[ -n "$TAR" ] && [ -n "$MD5" ] || { echo "no tar reported"; exit 1; }
OUT="$HOME/Desktop/上传OSS_inference_$R"; mkdir -p "$OUT"
scp -q -o BatchMode=yes "$H:$TAR" "$OUT/" && $SSH $H "rm -rf $(dirname "$TAR")"
got=$(md5 -q "$OUT/lingbot_inference_$R.tar")
[ "$got" = "$MD5" ] || { echo "md5 mismatch after download ($got vs $MD5)"; exit 1; }
echo "$MD5  lingbot_inference_$R.tar" > "$OUT/lingbot_inference_$R.tar.md5"
# 11/16: training code + README of the harness grasp-check CNN (the model file itself is in the snapshot)
G=$HOME/LingBotRoboTwin/graspnet
rm -f "$OUT/graspnet_src.tar" "$OUT/graspnet_src.tar.md5"
if echo "$out" | grep -q "graspnet_used=True"; then GU=1; else GU=0; echo "graspnet not used by this harness: graspnet_src.tar not added"; fi
if [ $GU = 1 ] && [ -d "$G" ]; then COPYFILE_DISABLE=1 tar --no-xattrs -cf "$OUT/graspnet_src.tar" -C "$(dirname "$G")" --exclude=.DS_Store graspnet \
  && (cd "$OUT" && md5 -r graspnet_src.tar | awk '{print $1"  graspnet_src.tar"}' > graspnet_src.tar.md5) && echo "graspnet_src.tar added"; fi
echo "ready: $OUT  (md5 $MD5)"
echo "next: upload that folder to the OSS workspace, then on the dev machine: make_submit.sh $R <HF_CKPT> <LIST>"
