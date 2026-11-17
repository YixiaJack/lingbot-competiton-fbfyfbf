"""Train the closed-set competition-object detector (59 classes) on the rendered frame-zero images; export; CPU timing.

Simulator weights are for this pre-event check only. At the venue the detector is retrained on venue data on the
official machines; only this code and the public pretrained weights (yolo11s.pt) go with us.
Usage: train_yolo_real.py --root R [--model yolo11s] [--epochs 120]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO, __version__

from catalog import CLASSES, DET_ITEMS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', default='yolo11s')
    p.add_argument('--epochs', type=int, default=120)
    a = p.parse_args()
    D = a.root / 'det'
    # val = items 90-109 only: images/val also holds renders of items >= DET_ITEMS that are not part of the design
    val_imgs = sorted(str(q) for q in (D / 'images' / 'val').glob('*.png') if int(q.name[:4]) < DET_ITEMS)
    assert len(val_imgs) == 400, len(val_imgs)
    (D / 'val_list.txt').write_text('\n'.join(val_imgs) + '\n')
    spec = {'path': str(D), 'train': 'images/train', 'val': str(D / 'val_list.txt'), 'names': CLASSES}
    (D / 'dataset.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
    run = D / 'runs' / a.model
    last = run / 'weights' / 'last.pt'
    t0 = time.time()
    if not (D / 'TRAIN_DONE').exists() and last.exists():   # preempted mid-training: resume
        YOLO(str(last)).train(resume=True)
    elif not (D / 'TRAIN_DONE').exists():
        YOLO(str(a.root / 'weights' / f'{a.model}.pt')).train(
            data=str(D / 'dataset.yaml'), epochs=a.epochs, imgsz=640, batch=32, device=0, workers=8,
            project=str(D / 'runs'), name=a.model, exist_ok=True, seed=42, deterministic=True, patience=30,
            # colour is the identity of many classes (plate colours, red/green chili, apples): no hue shift
            hsv_h=0.0, hsv_s=0.25, hsv_v=0.35, fliplr=0.5, mosaic=0.5, scale=0.3, translate=0.1, degrees=0.0, plots=True)
    (D / 'TRAIN_DONE').write_text('ok\n')
    best = run / 'weights' / 'best.pt'
    model = YOLO(str(best))
    val = model.val(data=str(D / 'dataset.yaml'), device=0, imgsz=640, plots=True, split='val')
    per_class = {CLASSES[int(c)]: float(m) for c, m in zip(val.box.ap_class_index, val.box.maps[val.box.ap_class_index])}
    try:   # optional deployment format; the pipeline (select_eval, eval_real) uses best.pt. The overlay is read-only,
        exported, export_error = str(model.export(format='onnx', imgsz=640, device='cpu', simplify=False)), None
    except Exception as e:   # so ultralytics cannot auto-install onnx (11/12): record it, do not block
        exported, export_error = None, repr(e)[:300]
    sample = Path(val_imgs[0])
    img = np.asarray(Image.open(sample).convert('RGB'))[..., ::-1].copy()
    times = []
    for i in range(25):
        s = time.perf_counter(); model.predict(img, device='cpu', imgsz=640, verbose=False)
        if i >= 5: times.append(1000 * (time.perf_counter() - s))
    report = {'ultralytics': __version__, 'model': a.model, 'classes': len(CLASSES), 'train_seconds': time.time() - t0,
              'map50': float(val.box.map50), 'map50_95': float(val.box.map), 'per_class_map50_95': per_class,
              'cpu_median_ms': float(np.median(times)), 'cpu_p95_ms': float(np.percentile(times, 95)),
              'export_onnx': exported, 'export_error': export_error, 'weights_sha256': hashlib.sha256(best.read_bytes()).hexdigest(),
              'scope': 'pre-event simulation check only; retrain on venue data at the venue'}
    (D / 'detector_result.json').write_text(json.dumps(report, indent=2))
    (D / 'YOLO_READY').write_text('ok\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_class_map50_95'}, indent=2))


if __name__ == '__main__':
    main()
