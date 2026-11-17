"""Export an ultralytics YOLO .pt to TorchScript for yolocue (run with ~/locany/yolo_env/bin/python on the 5080).

The inference env has torch 2.8 but no ultralytics; TorchScript carries the network, yolocue_bridge.Detector repeats
ultralytics 8.4 predict around it. Fixed input 480x640 = what predict feeds a 640x480 frame at imgsz 640 (rect, no pad).
  export_ts.py SRC.pt NAME   ->  ~/yolocue/weights/NAME.pt (copy), NAME.torchscript, NAME.names.json
"""
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

src, name = Path(sys.argv[1]).expanduser(), sys.argv[2]
out = Path.home() / 'yolocue' / 'weights'
out.mkdir(parents=True, exist_ok=True)
dst = out / f'{name}.pt'
shutil.copy2(src, dst)
m = YOLO(str(dst))
ts = m.export(format='torchscript', imgsz=(480, 640), device='cpu', half=False, optimize=False, batch=1)
names = [m.names[i] for i in range(len(m.names))]
(out / f'{name}.names.json').write_text(json.dumps({'names': names, 'imgsz': [480, 640], 'source': str(src)}, indent=1))
print(json.dumps({'torchscript': str(ts), 'classes': len(names)}))
