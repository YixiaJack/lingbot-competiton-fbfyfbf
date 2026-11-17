"""Compare detectors on frames none of the v7 runs trained on, with the cue rule used online (yolocue_bridge.select).

  new held-out   the 23 new team episodes kept out of training (every third); truth = the target crop verified by eye
  new trained    the other 46 new episodes (their crops were pasted / the frame was used) -- optimistic, for reference
  old val        the 22 team episodes of the old val split, frame 0; truth = the corrected label of the target class
  old train      the 49 old team train episodes -- seen, for reference
A cue is "right" when its white box sits on the true target box. Sheets with the cue drawn are written for a look.
usage: python -I eval_v7.py OUT MODEL [MODEL ...]      (MODEL = name of ~/yolocue/weights/<name>.torchscript)
"""
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.home() / 'yolocue'))
import ff_trigger as FT  # noqa: E402
import yolocue_bridge as YC  # noqa: E402
from ff_paste import holdout  # noqa: E402

H = Path.home(); Y = H / 'locany/yolo'; T = H / 'yolocue/test_out'
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
same = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(min(area(a), area(b)), 1) > 0.5


def cases():
    out = []
    rows = {r['uid']: r for r in json.loads((T / 'firstframes_1114/index.json').read_text())}
    crops = json.loads((T / 'targets/crops.json').read_text())
    ho = holdout({m['uid'] for m in crops})
    for m in crops:
        if m['role'] != 'target':
            continue
        pr, _ = FT.parse_row(rows[m['uid']])
        out.append({'set': 'new held-out' if m['uid'] in ho else 'new trained', 'id': m['uid'], 'img': T / 'firstframes_1114/frames' / f"{m['uid']}_first.jpg",
                    'instruction': rows[m['uid']]['instruction'], 'pr': pr, 'truth': [m['box']]})
    props = json.loads((T / 'old/old_proposals.json').read_text()); final = json.loads((Y / 'ds_v7_team/final_labels.json').read_text())
    for e, v in props.items():
        if not v['target_cls']:
            continue
        pr = (1 if v['dest_cls'].startswith('plate_') else 2, v['target_cls'], v['dest_cls'])
        out.append({'set': 'old ' + v['split'], 'id': 'ep' + e, 'img': Y / 'ds_v6/images' / v['split'] / f"ep{e}_{v['frame0']:04d}.jpg",
                    'instruction': v['instruction'], 'pr': pr, 'truth': [b for c, b in final[e]['labels'] if c == pr[1]]})
    return out


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); models = sys.argv[2:]
    det = {m: YC.Detector(YC.ROOT / f'weights/{m}.torchscript', json.loads((YC.ROOT / f'weights/{m}.names.json').read_text())['names']) for m in models}
    C = cases(); F = ImageFont.load_default(size=13)
    for c in C:
        rgb = cv2.cvtColor(cv2.imread(str(c['img'])), cv2.COLOR_BGR2RGB); c['res'] = {}
        for m in models:
            tb, db, reason = YC.select(det[m](rgb), *c['pr'])
            c['res'][m] = {'reason': reason, 'tb': tb, 'db': db, 'right': (any(same(tb, t) for t in c['truth']) if (tb and c['truth']) else None)}
    summ = {}
    for s in ('new held-out', 'old val', 'new trained', 'old train'):
        P = [c for c in C if c['set'] == s]; summ[s] = {}
        for m in models:
            ok = [c for c in P if c['res'][m]['reason'] == 'ok']
            summ[s][m] = {'episodes': len(P), 'cue': len(ok), 'rate': round(len(ok) / max(len(P), 1), 3), 'right': sum(c['res'][m]['right'] is True for c in ok),
                          'wrong': sum(c['res'][m]['right'] is False for c in ok), 'no_truth': sum(c['res'][m]['right'] is None for c in ok),
                          'task1': '%d/%d' % (sum(c['pr'][0] == 1 for c in ok), sum(c['pr'][0] == 1 for c in P)), 'task2': '%d/%d' % (sum(c['pr'][0] == 2 for c in ok), sum(c['pr'][0] == 2 for c in P)),
                          'reasons': dict(collections.Counter(c['res'][m]['reason'] for c in P))}
            print(f"{s:13s} {m:5s}", json.dumps(summ[s][m]))
    last = models[-1]
    for s in ('new held-out', 'old val'):
        P = [c for c in C if c['set'] == s]
        print(f'-- {s}: per episode ({" / ".join(models)})')
        for c in P:
            print('  ', c['id'], c['pr'][1], '->', c['pr'][2], '|', ' / '.join(f"{c['res'][m]['reason']}{'' if c['res'][m]['right'] is not False else ' WRONG'}" for m in models))
        cells = []
        for n, c in enumerate(P, 1):
            r = c['res'][last]; rgb = np.asarray(Image.open(c['img']).convert('RGB'))
            im = Image.fromarray(YC.draw(rgb, r['tb'], r['db'], False)) if r['reason'] == 'ok' else Image.fromarray(rgb)
            cell = Image.new('RGB', (480, 394), (255, 255, 255)); cell.paste(im.resize((480, 360), Image.LANCZOS), (0, 34)); g = ImageDraw.Draw(cell)
            g.text((4, 1), f"#{n} {c['id']} {c['pr'][1]} -> {c['pr'][2]}   {last}: {r['reason']}" + ('' if r['right'] is not False else '  (auto: WRONG)'), font=F, fill=(0, 0, 0) if r['reason'] == 'ok' else (200, 0, 0))
            g.text((4, 17), (c['instruction'] or '')[:70], font=F, fill=(0, 0, 160)); cells.append(cell)
        for k in range(0, len(cells), 9):
            cc = cells[k:k + 9]; sheet = Image.new('RGB', (1440, ((len(cc) + 2) // 3) * 394), (255, 255, 255))
            for i, im in enumerate(cc):
                sheet.paste(im, ((i % 3) * 480, (i // 3) * 394))
            sheet.save(out / f"{s.replace(' ', '_')}_{last}_{k // 9 + 1}.jpg", quality=88)
    (out / f'eval_{"_".join(models)}.json').write_text(json.dumps({'summary': summ, 'cases': [{k: (str(v) if k == 'img' else v) for k, v in c.items()} for c in C]}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
