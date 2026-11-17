#!/bin/bash
# Wait for labeling round 1 (stride 30) to finish, pause labeling, build + train YOLO v1, resume labeling.
cd ~/locany; LOG=yolo/v1.log; mkdir -p yolo
echo "$(date '+%F %T') waiting for round 1" >> $LOG
until grep -q '"level": 1' label/main/labels.jsonl; do sleep 30; done
echo "$(date '+%F %T') round 1 done ($(grep -c '"level": 0' label/main/labels.jsonl) rows); pausing labeling" >> $LOG
touch label/PAUSE
while pgrep -f "^python scripts/label.py" > /dev/null; do sleep 5; done
env/bin/python scripts/levels0.py 2>&1 | grep -v "Warning\|warn\|Generation\|inherit\|owner\|slow\|torch_dtype\|Loading" >> $LOG
yolo_env/bin/python scripts/build_yolo.py --out yolo/ds_v1 --levels 0 >> $LOG 2>&1 || { echo "build failed" >> $LOG; rm -f label/PAUSE; exit 1; }
yolo_env/bin/python scripts/train_yolo.py yolo/ds_v1 v1 50 >> yolo/v1_train.log 2>&1; echo "$(date '+%F %T') train rc=$?" >> $LOG
tail -3 yolo/v1_train.log >> $LOG
rm -f label/PAUSE
echo "$(date '+%F %T') labeling resumed" >> $LOG
