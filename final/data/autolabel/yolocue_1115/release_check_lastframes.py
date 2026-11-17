"""False-regrasp rate of the yolocue regrasp insurance on recorded demos (5080, CPU YOLO; the service is not touched).
Every demo ends with the object placed, so any 'regrasp' decision on its LAST frame is a false regrasp.
Per episode: instruction parses (task 1/2) -> frame 0: detect + select (the online cue rule, with the plate-colour rescue)
-> cue ok -> last frame: detect, best target box (conf >= 0.5) -> placed (YoloCue._on_destination) / regrasp / unseen.
usage: cd ~/yc_test && python -I release_check_lastframes.py ROOT [ROOT ...] [--out out.json] [--save-dir imgs]"""
import argparse, json, sys, time, types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolocue_bridge as YC  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('roots', nargs='+'); ap.add_argument('--out', default='release_check.json'); ap.add_argument('--save-dir', default='release_false')
a = ap.parse_args()
cfg = YC.load_config()
det = YC.detector(cfg)
print('detector', cfg['weights'])
Path(a.save_dir).mkdir(exist_ok=True)
rows, t0 = [], time.time()
for root in a.roots:
    try:
        ds = LeRobotDataset('local/' + Path(root).name, root=root, video_backend='pyav')
    except Exception as e:
        print('SKIP', root, repr(e)[:200]); continue
    name = Path(root).name[:12]
    for e in range(ds.meta.total_episodes):
        ep = ds.meta.episodes[e]
        instr = ep['tasks'][0] if len(ep['tasks']) else ''
        p = YC.parse(instr)
        if not p:
            continue
        rgb = lambda i: (ds[int(i)]['observation.images.camera_top'].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        f0, fl = rgb(ep['dataset_from_index']), rgb(ep['dataset_to_index'] - 1)
        d0 = det(f0)
        tb, db, why = YC.select(d0, *p, conf=cfg['conf'], margin=cfg['margin'], rgb=f0)
        r = {'src': name, 'ep': e, 'task': p[0], 'target': p[1], 'instr': instr, 'cue': why}
        if why == 'ok':
            dl = det(fl)
            # same rules as YoloCue._check_release: placed if any target-class instance (>= 0.25) is on a plate /
            # in the organizer; otherwise a regrasp only on a target-class box that is not a never-moved look-alike
            fake = types.SimpleNamespace(parsed=p)
            on = any(YC.YoloCue._on_destination(fake, d[1:5], db, dl) for d in dl if d[0] == p[1] and d[5] >= 0.25)
            static = [d[1:5] for d in d0 if d[5] >= 0.25 and YC._iou(d[1:5], tb) < 0.5]
            t = [(d[5], d[1:5]) for d in sorted(dl, key=lambda d: -d[5]) if d[0] == p[1] and d[5] >= cfg['conf']
                 and not any(YC._iou(d[1:5], b) > 0.5 for b in static)]
            if on:
                r['decision'] = 'placed'
            elif not t:
                r['decision'] = 'unseen'
            else:
                r['decision'] = 'regrasp'
                r['tb_last'], r['db'], r['tb0'] = [round(v, 1) for v in t[0][1]], db, tb
                r['iou_last_vs_cue'] = round(YC._iou(t[0][1], tb), 2)   # ~1: the boxed object never moved
                if True:
                    from PIL import Image, ImageDraw
                    im = Image.fromarray(YC.draw(fl, t[0][1], db, False)); d = ImageDraw.Draw(im)
                    for q in dl:
                        if q[5] >= 0.25:
                            d.rectangle(q[1:5], outline=(255, 0, 0), width=1); d.text((q[1] + 2, q[2] + 1), f'{q[0]} {q[5]:.2f}', fill=(255, 0, 0))
                    im.save(f'{a.save_dir}/{name}_ep{e:04d}_t{p[0]}.jpg', quality=88)
        rows.append(r)
    print(root, 'done', len(rows), f'{time.time() - t0:.0f}s', flush=True)

summ = {}
for task in (1, 2):
    rr = [r for r in rows if r['task'] == task]
    cued = [r for r in rr if r['cue'] == 'ok']
    c = {k: sum(r.get('decision') == k for r in cued) for k in ('placed', 'regrasp', 'unseen')}
    summ[f'task{task}'] = {'parsed': len(rr), 'cue_ok': len(cued), **c,
                           'false_regrasp_rate': round(c['regrasp'] / max(1, c['placed'] + c['regrasp']), 3),
                           'false_eps': [f"{r['src']}:{r['ep']}" for r in cued if r.get('decision') == 'regrasp']}
Path(a.out).write_text(json.dumps({'summary': summ, 'rows': rows}, ensure_ascii=False, indent=1))
print('SUMMARY', json.dumps(summ, ensure_ascii=False))
