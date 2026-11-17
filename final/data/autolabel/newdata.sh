#!/bin/bash
# After team datasets land in ~/locany/data: merge, run the frame-0 jobs for the new episodes, then restart the
# per-frame labelling over the merged view (new episodes' round 1 first). Run while label/PAUSE is set.
cd ~/locany; LOG=label/newdata.log; export LA_DS=merged PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F="Warning\|warn(\|GenerationMixin\|inherit\|auto-classes\|owner\|slow processor\|torch_dtype\|box_avg\|Loading checkpoint\|flash_attn\|magi"
args=(public647:0)
[ -d data/105a2d09-a91b-4ab1-9961-2375dc2bd16e ] && args+=(105a2d09-a91b-4ab1-9961-2375dc2bd16e:1000)
# b60f1e2e is a subset of 105a2d09 (87 identical first frames, 11/14): not merged
touch label/PAUSE   # the v3 wrapper removes PAUSE when training ends; keep the old labeller paused
echo "$(date '+%F %T') merge ${args[*]}" >> $LOG
env/bin/python scripts/make_merged.py "${args[@]}" >> $LOG 2>&1 || exit 1
while pgrep -f "^python scripts/label.py" > /dev/null; do sleep 5; done
LA_IGNORE_PAUSE=1 env/bin/python scripts/label.py --name first0 --frames first 2>&1 | grep -v "$F" | tail -2 >> $LOG
LA_IGNORE_PAUSE=1 env/bin/python scripts/label.py --name first0_t2 --frames first --tasks 2 2>&1 | grep -v "$F" | tail -2 >> $LOG
env/bin/python scripts/ground0.py 2>&1 | grep -v "$F" | tail -1 >> $LOG
env/bin/python scripts/cosm0.py 2>&1 | grep -v "$F" | tail -1 >> $LOG
echo "$(date '+%F %T') frame-0 jobs done; restarting per-frame labelling on merged" >> $LOG
old=$(pgrep -f "^/bin/bash scripts/run.sh main"); [ -n "$old" ] && kill $old
rm -f label/PAUSE
setsid nohup scripts/run.sh main --strides 30,10,5,1 --batch-size 6 < /dev/null > /dev/null 2>&1 &
echo "$(date '+%F %T') labelling restarted (LA_DS=merged)" >> $LOG
