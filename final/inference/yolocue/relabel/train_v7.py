"""Train a v7 YOLO on the corrected + copy-paste data, next to the loaded 6B inference model on the 5080: the process is
capped to a small share of GPU memory (the service keeps its ~12.8 GB) and v7_watch.sh pauses it while an inference
session runs. INIT = v5 (continue from the current team detector, short low-LR run) or coco (fresh from yolo11s COCO
weights, as v1-v5 were trained).
usage (cwd ~/locany): yolo_env/bin/python train_v7.py NAME DATA.yaml EPOCHS GB v5|coco"""
import sys
import torch
from ultralytics import YOLO
name, data, epochs, gb, init = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
torch.cuda.set_per_process_memory_fraction(gb * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
# same augmentation as train_yolo.py: colour is the identity of many classes, so no hue shift
common = dict(data=data, epochs=epochs, imgsz=640, batch=8, device=0, workers=4, cache=False, project='/home/guest/locany/yolo/runs', name=name, exist_ok=True,
              seed=42, plots=False, val=True, patience=100, hsv_h=0.0, hsv_s=0.25, hsv_v=0.35, fliplr=0.5, mosaic=0.5, scale=0.3,
              translate=0.1, degrees=0.0)
if init == 'v5':
    YOLO('yolo/runs/v5/weights/best.pt').train(optimizer='SGD', lr0=0.004, lrf=0.1, warmup_epochs=1.0, close_mosaic=2, **common)
elif init == 'v7a':   # 11/15: v8 = v7a + hand-checked new task-1 frames, short low-LR fine-tune
    YOLO('/home/guest/locany/yolo/runs/v7a/weights/best.pt').train(optimizer='SGD', lr0=0.002, lrf=0.1, warmup_epochs=0.5, close_mosaic=2, **common)
elif init == 'v11':   # 11/15 11:55: v12 = v11 list + by-eye frames of eps 389-411 (straight cucumber vs green chili, mangosteen behind grapes, red pepper)
    YOLO('/home/guest/locany/yolo/runs/v11/weights/best.pt').train(optimizer='SGD', lr0=0.002, lrf=0.1, warmup_epochs=0.5, close_mosaic=2, **common)
elif init == 'v9b':   # 11/15 11:00: v11 = v9 + 249 hand-checked frames of eps 302-388 (confusable objects upweighted), short low-LR fine-tune
    YOLO('/home/guest/locany/yolo/runs/v9/weights/best.pt').train(optimizer='SGD', lr0=0.002, lrf=0.1, warmup_epochs=0.5, close_mosaic=2, **common)
elif init == 'v9':    # 11/15 05:40: v10 = v9 + corrected official labels + own eps 189-194 + task-3 negatives, same short low-LR fine-tune
    YOLO('/home/guest/locany/yolo/runs/v9/weights/best.pt').train(optimizer='SGD', lr0=0.002, lrf=0.1, warmup_epochs=0.5, close_mosaic=2, **common)
elif init == 'v8':    # 11/15 05:00: v9 = v8 + hand-checked newest frames (0308 batch, task 1 + 2), same short low-LR fine-tune
    YOLO('/home/guest/locany/yolo/runs/v8/weights/best.pt').train(optimizer='SGD', lr0=0.002, lrf=0.1, warmup_epochs=0.5, close_mosaic=2, **common)
else:
    YOLO('yolo/weights/yolo11s.pt').train(optimizer='SGD', lr0=0.01, lrf=0.01, warmup_epochs=2.0, close_mosaic=5, **common)
