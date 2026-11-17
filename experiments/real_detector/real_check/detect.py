import sys, json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
w, out = sys.argv[1], sys.argv[2]; dirs = sys.argv[3:]
m = YOLO(w); names = m.names
res = {}
for d in dirs:
    files = sorted(Path(d).glob('*.png'))
    for i in range(0, len(files), 32):
        ims = [Image.open(f).convert('RGB') for f in files[i:i+32]]
        for f, pr in zip(files[i:i+32], m.predict(ims, conf=0.01, verbose=False, device='mps')):
            b = pr.boxes
            res[f'{d}/{f.stem}'] = [[names[int(c)], *[round(v, 1) for v in xy], round(s, 3)] for xy, c, s in zip(b.xyxy.tolist(), b.cls.tolist(), b.conf.tolist())]
Path(out).write_text(json.dumps(res))
print(len(res))
