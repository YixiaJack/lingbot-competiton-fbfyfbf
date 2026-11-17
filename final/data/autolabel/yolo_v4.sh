#!/bin/bash
# Wait for round 1 on the team episodes, refresh the appearance IDs on the merged view, build + train v4.
cd ~/locany; export LA_DS=merged; LOG=yolo/v4.log
exp=$(env/bin/python -c "
import pandas as pd, math
e=pd.read_parquet('data/merged/meta/episodes/chunk-000/file-000.parquet'); e=e[e.episode_index>=1000]
print(sum(math.ceil(l/30) for l in e['length']))")
echo "$(date '+%F %T') waiting for $exp team round-1 frames" >> $LOG
until [ "$(grep -c '"level": 0' <(grep -E '"ep": 1[0-9]{3},' label/main/labels.jsonl))" -ge "$exp" ]; do sleep 30; done
echo "$(date '+%F %T') team round 1 done" >> $LOG
CUDA_VISIBLE_DEVICES= yolo_env/bin/python scripts/fruit_id.py 2>&1 | grep -v "%\|Warning" | tail -4 >> $LOG
CUDA_VISIBLE_DEVICES= yolo_env/bin/python scripts/cosm_id.py 2>&1 | grep -v "%\|Warning" | tail -4 >> $LOG
yolo_env/bin/python scripts/build_yolo.py --out yolo/ds_v4 --levels 0,1 --t3-levels 0 --regions --t1id 2>&1 | grep -A8 '"images"' | tr -d '\n ' >> $LOG; echo >> $LOG
touch label/PAUSE; while pgrep -f "^python scripts/label.py" > /dev/null; do sleep 5; done
echo "$(date '+%F %T') training v4" >> $LOG
yolo_env/bin/python scripts/train_yolo.py yolo/ds_v4 v4 60 20 > yolo/v4_train.log 2>&1; echo "$(date '+%F %T') train rc=$?" >> $LOG
rm -f label/PAUSE; tail -1 yolo/v4_train.log >> $LOG
