"""Single-GPU YOLO rehearsal. Simulator-only weights must stay out of the final submission."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from PIL import Image
import numpy as np
import yaml
from ultralytics import YOLO, __version__


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    r = a.root
    ds = r / 'detector_data'
    for split in ('train','val'):
        (ds / 'images' / split).mkdir(parents=True, exist_ok=True)
        (ds / 'labels' / split).mkdir(parents=True, exist_ok=True)
    # Distinct scenes for detector validation; no VLA evaluation seeds used.
    for i in range(120):
        source = r / 'episodes' / f'{i:04d}'
        meta = json.loads((source / 'complete.json').read_text())
        split = 'val' if i >= 96 else 'train'
        image = Image.open(source / 'frame0.png')
        w,h = image.size
        shutil.copy2(source / 'frame0.png', ds / 'images' / split / f'{i:04d}.png')
        rows = []
        for cls, (x1,y1,x2,y2) in enumerate(meta['boxes']):
            rows.append(f'{cls} {(x1+x2)/2/w:.7f} {(y1+y2)/2/h:.7f} {(x2-x1)/w:.7f} {(y2-y1)/h:.7f}')
        (ds/'labels'/split/f'{i:04d}.txt').write_text('\n'.join(rows)+'\n')
    spec = {'path': str(ds), 'train': 'images/train', 'val': 'images/val',
            'names': ['red_block','green_block','blue_block','yellow_plate','cyan_plate','magenta_plate']}
    (ds/'dataset.yaml').write_text(yaml.safe_dump(spec))
    t0=time.time()
    run=r/'detector'/'yolo11n'
    last=run/'weights'/'last.pt'
    model=YOLO(str(last) if last.exists() else str(r/'weights'/'yolo11n.pt'))
    if last.exists():
        model.train(resume=True)
    else:
        model.train(data=str(ds/'dataset.yaml'), epochs=80, imgsz=640, batch=16, device=0,
                    workers=4, project=str(r/'detector'), name='yolo11n', exist_ok=True,
                    seed=42, deterministic=True, amp=False, hsv_h=0., hsv_s=.1, hsv_v=.15,
                    fliplr=0., mosaic=.5, patience=20, plots=True)
    model=YOLO(str(run/'weights'/'best.pt'))
    val=model.val(data=str(ds/'dataset.yaml'), device=0, plots=True)
    exported=model.export(format='torchscript', imgsz=640, device='cpu')
    sample=np.asarray(Image.open(ds/'images'/'val'/'0096.png'))[...,::-1].copy()
    times=[]
    for i in range(25):
        start=time.perf_counter(); model.predict(sample,device='cpu',verbose=False)
        if i>=5: times.append(1000*(time.perf_counter()-start))
    report={'ultralytics': __version__, 'model':'yolo11n', 'training_seconds':time.time()-t0,
            'validation':val.results_dict, 'cpu_median_ms':float(np.median(times)),
            'cpu_p95_ms':float(np.percentile(times,95)), 'export':str(exported),
            'weights_sha256':hashlib.sha256((run/'weights'/'best.pt').read_bytes()).hexdigest(),
            'purpose':'pre-event simulation rehearsal only; not final competition weights'}
    (r/'detector_result.json').write_text(json.dumps(report, indent=2))
    (r/'YOLO_READY').write_text('complete\n')
    print(json.dumps(report, indent=2))


if __name__=='__main__':
    main()
