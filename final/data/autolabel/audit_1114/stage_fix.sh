#!/bin/bash
# Stage the 11/14 corrected own datasets (lingbot_fix_1114.tar, built off the trainer) next to the live ones; the running
# training is not touched. Fixes: priv_t12 ep51 contour compact name, priv2_t12 ep114 watermelon spelling, priv_t3_sub
# ep80 dustpan arm (right) and boundaries, priv2_t3_sub ep95 dropped (two pours once the dustpan arm is read right).
# Writes lists/v3fix.txt = lists/v3.txt with those four datasets swapped for the *_fix copies. log: logs/stage_fix.log
W=$HOME/lingbot_train; D=/data/lingbot; cd $W; . tools/env.sh
TAR=${TAR:-/workspace/lingbot_fix_1114.tar}; TAR_MD5=a1744e71794a30459fca0d2aec27078f
exec >> $W/logs/stage_fix.log 2>&1
echo "== $(date -Is) waiting for $TAR"
until [ -f "$TAR" ] && [ "$(md5sum < "$TAR" | cut -d' ' -f1)" = "$TAR_MD5" ]; do sleep 20; done
mkdir -p $D/fix_tmp && cp "$TAR" $D/ && rm -rf $D/fix_tmp/* && tar -xf $D/lingbot_fix_1114.tar -C $D/fix_tmp && find $D/fix_tmp -name '._*' -delete
for d in priv_t12 priv_t3_sub priv2_t12 priv2_t3_sub; do
  rm -rf $D/data/${d}_fix; mv $D/fix_tmp/$d $D/data/${d}_fix
  case $d in priv2_*) v=$D/local/priv_4e81/videos ;; *) v=$D/local/priv_b60f/videos ;; esac
  ln -sfn $v $D/data/${d}_fix/videos
  printf "%s " ${d}_fix; CUDA_VISIBLE_DEVICES= nice -n 19 python tools/check_box_ds.py $D/data/${d}_fix 2>&1 | grep -E "items ok|FAIL|Error" | head -2 | tr "\n" " "; echo
done
sed -E 's#/data/lingbot/data/(priv_t12|priv_t3_sub|priv2_t12|priv2_t3_sub)$#/data/lingbot/data/\1_fix#' lists/v3.txt > lists/v3fix.txt
diff lists/v3.txt lists/v3fix.txt
echo "== $(date -Is) STAGED: lists/v3fix.txt ready (apply at the next restart from a checkpoint)"
