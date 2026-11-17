"""Train yolo11s (public COCO weights) on a build_yolo.py dataset, then detect on frame 0 of every episode."""
import json, sys, time
from pathlib import Path
from ultralytics import YOLO
H = Path.home() / 'locany'
ds, name = Path(sys.argv[1]), sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
patience = int(sys.argv[4]) if len(sys.argv) > 4 else 12
t0 = time.time()
YOLO(str(H / 'yolo/weights/yolo11s.pt')).train(
    data=str(ds / 'dataset.yaml'), epochs=epochs, patience=patience, imgsz=640, batch=32, device=0, workers=4, cache=False,   # 11/14: 30 GB box hung once; no RAM cache
    project=str(H / 'yolo/runs'), name=name, exist_ok=True, seed=42, plots=True,
    # colour is the identity of many classes (plate colours, red/green fruit): no hue shift
    hsv_h=0.0, hsv_s=0.25, hsv_v=0.35, fliplr=0.5, mosaic=0.5, scale=0.3, translate=0.1, degrees=0.0)
best = H / 'yolo/runs' / name / 'weights/best.pt'
m = YOLO(str(best)); names = m.names
res = {}
eps = sorted(int(p.name[2:]) for p in (H / 'label/first0/images').iterdir())
for i in range(0, len(eps), 32):
    ch = eps[i:i + 32]
    for e, pr in zip(ch, m.predict([str(H / f'label/first0/images/ep{e:03d}/0000.jpg') for e in ch], conf=0.01, verbose=False, device=0)):
        b = pr.boxes
        res[f'first/{e:04d}'] = [[names[int(c)], *[round(v, 1) for v in xy], round(s, 3)] for xy, c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist())]
(H / 'yolo/runs' / name / 'dets_first.json').write_text(json.dumps(res))
print(json.dumps({'train_min': round((time.time() - t0) / 60, 1), 'best': str(best)}))
