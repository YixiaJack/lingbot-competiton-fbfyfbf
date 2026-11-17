"""v13 proposals on today's live no-cue raw frames (5080 yolocue logs): v12 detections >= 0.25 (cross-class NMS 0.6),
low boxes 0.05-0.25 (class unknown, hint shown), colour blobs nothing covers. Numbered review sheets -> yolo13/live_prop/."""
import json, glob, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path.home() / 'lingbot_v7work')); sys.path.insert(0, str(Path.home() / 'lingbot_v7work/yolo12'))
import yolocue_bridge as YC
H = Path.home() / 'lingbot_v7work/yolo13'; L = H / 'live'; O = H / 'live_prop'; O.mkdir(exist_ok=True)
import importlib.util
spec = importlib.util.spec_from_file_location('p12', str(Path.home() / 'lingbot_v7work/yolo12/propose.py'))
src = open(spec.origin).read(); blobs_src = src[src.index('def blobs('):src.index('def blocked(')]
ns = {}; exec('import cv2, numpy as np\n' + blobs_src, ns); blobs = ns['blobs']
det = YC.Detector(YC.ROOT / 'weights/v12.torchscript', json.loads((YC.ROOT / 'weights/v12.names.json').read_text())['names'], conf=0.05)
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]); inter = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
iou = lambda a, b: inter(a, b) / (area(a) + area(b) - inter(a, b) + 1e-9)
KEEP = ['172346', '172432', '172549', '173022', '175132', '175219', '175700', '180359', '103619', '112038', '080608']
F = ImageFont.load_default(size=13); FB = ImageFont.load_default(size=15); props, cells = {}, []
for f in sorted(glob.glob(str(L / '*_nocue_raw.jpg'))):
    b = Path(f).name.replace('_nocue_raw.jpg', '')
    if b[9:15] not in KEEP: continue
    task = json.loads(open(L / f'{b}.jsonl').readline())['task']; rgb = np.asarray(Image.open(f).convert('RGB'))
    d = det(rgb); P = []
    for x in sorted((x for x in d if x[5] >= 0.25), key=lambda x: -x[5]):
        if all(iou(x[1:5], p['box']) < 0.6 for p in P): P.append({'box': [float(v) for v in x[1:5]], 'src': 'yolo', 'cls': x[0], 'conf': round(x[5], 2)})
    for x in sorted((x for x in d if 0.05 <= x[5] < 0.25 and not x[0].startswith(('plate_', 'organizer_'))), key=lambda x: -x[5]):
        if all(iou(x[1:5], p['box']) < 0.3 and inter(x[1:5], p['box']) / max(area(x[1:5]), 1) < 0.5 for p in P): P.append({'box': [float(v) for v in x[1:5]], 'src': 'low', 'cls': None, 'hint': x[0], 'conf': round(x[5], 2)})
    for bb in blobs(rgb, [p['box'] for p in P]): P.append({'box': [float(v) for v in bb], 'src': 'blob', 'cls': None, 'conf': None})
    for i, p in enumerate(P): p['id'] = i
    props[b] = {'task': task, 'parsed': YC.parse(task), 'props': P}
    im = Image.fromarray(rgb).resize((960, 720)); g = ImageDraw.Draw(im); k = 1.5
    for p in P:
        col = (120, 200, 255) if (p['cls'] or '').startswith(('plate_', 'organizer_')) else {'yolo': (255, 235, 0), 'low': (255, 150, 220), 'blob': (255, 60, 60)}[p['src']]
        bx = [v * k for v in p['box']]; g.rectangle(bx, outline=col, width=2)
        lab = f"{p['id']}:{p['cls'] or '?' + (p.get('hint') or '')}" + (f" {p['conf']:.2f}" if p['conf'] else '')
        tw = g.textlength(lab, font=F); ty = bx[1] - 15 if bx[1] >= 15 else bx[3] + 1; g.rectangle([bx[0], ty, bx[0] + tw + 4, ty + 14], fill=col); g.text((bx[0] + 2, ty), lab, font=F, fill=(0, 0, 0))
    cell = Image.new('RGB', (960, 750), 'white'); cell.paste(im, (0, 30)); ImageDraw.Draw(cell).text((4, 6), f'{b[9:15]}  {task}', font=FB, fill=(0, 0, 160)); cells.append(cell)
for k in range(0, len(cells), 2):
    cc = cells[k:k + 2]; sh = Image.new('RGB', (1920, 750), 'white')
    for i, c in enumerate(cc): sh.paste(c, (i * 960, 0))
    sh.save(O / f'live_{k // 2 + 1}.jpg', quality=85)
json.dump(props, open(O / 'props.json', 'w'), indent=1, ensure_ascii=False); print(len(props), 'frames', sum(len(v['props']) for v in props.values()), 'proposals')
