#!/bin/bash
# Run the two v7 candidates one after the other on the same data: v7a continues from v5, v7s starts from COCO weights.
cd ~/locany
nice -n 5 yolo_env/bin/python ~/yolocue/relabel/train_v7.py v7a yolo/ds_v7a.yaml 12 3.0 v5   > yolo/v7a_train.log 2>&1; echo "$(date '+%F %T') v7a rc=$?" >> yolo/v7_queue.log
nice -n 5 yolo_env/bin/python ~/yolocue/relabel/train_v7.py v7s yolo/ds_v7a.yaml 45 3.0 coco > yolo/v7s_train.log 2>&1; echo "$(date '+%F %T') v7s rc=$?" >> yolo/v7_queue.log
