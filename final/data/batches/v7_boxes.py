"""Box cues for VLA training with YOLO v7a, run where the videos are (the Mac): same rules as autolabel/t1_boxes.py,
t2_boxes.py and own_boxes_v5.py, but the detector is the online one (yolocue_bridge.Detector on the decoded frame 0 of
camera_top, exactly what the inference service sees) and the instruction is parsed with yolocue_bridge.parse.

  task 1: target/destination by the cue rule; grasp frame = first close of a gripper COMMAND after it opened
          (t1_boxes.grasp_frame); no cue when a box is missing or the target is ambiguous.
  task 2: as task 1, plus t2_boxes' checks: one acting arm, the measured gripper holds something 0.5-1 s after the close
          (> 0.02 above its start) and that arm lifts > 3 cm.
  --own:  own data is checked like task 2 for both tasks (own_boxes_v5 / private_prep grasp_rule_ok).
Writes OUT_t1.json / OUT_t2.json in the derive_box_dataset.py format ({"summary", "episodes": {ep: {...}}}) and a
contact sheet per task for a visual check.
usage: python -I v7_boxes.py V1 OUT_PREFIX [--episodes e1,e2,...] [--instructions instr.json] [--own]
       instr.json = {episode: instruction} used instead of the dataset's own text (rewritten prompts for own data)
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolocue_bridge as YC  # noqa: E402

CAM = 'observation.images.camera_top'


def grasp_frame(cmd):
    """t1_boxes.grasp_frame: first frame on which some gripper command closes after having opened."""
    best = None
    for a in (0, 1):
        opened = False
        for j, v in enumerate(cmd[:, a]):
            if not opened and v > YC.CMD_OPEN:
                opened = True
            elif opened and v < YC.CMD_CLOSED:
                best = j if best is None else min(best, j)
                break
    return best


def frame0(v1, m):
    path = v1 / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"
    t0 = m[f'videos/{CAM}/from_timestamp']
    with av.open(str(path)) as ic:
        s = ic.streams.video[0]
        ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in ic.decode(s):
            if float(fr.pts * s.time_base) + 1e-3 >= t0:
                return fr.to_ndarray(format='rgb24')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('v1', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--episodes'); ap.add_argument('--instructions', type=Path); ap.add_argument('--own', action='store_true')
    a = ap.parse_args()
    det = YC.detector(YC.load_config())
    meta = {m['episode_index']: m for m in pq.read_table(sorted((a.v1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
    lab = {}
    for line in open(a.v1 / 'meta/labels.jsonl'):
        r = json.loads(line); L = (r.get('labels') or [{}])[0]
        cf = {c.get('name'): c.get('value') for c in (L.get('custom_fields') or [])}
        lab[r['episode_index']] = (cf.get('id') or f"ep{r['episode_index']:04d}", L.get('task_instruction') or '', L.get('result'))
    instr = {int(k): v for k, v in json.loads(a.instructions.read_text()).items()} if a.instructions else {}
    if a.instructions and not a.episodes:
        a.episodes = ','.join(str(k) for k in sorted(instr))
    want = [int(x) for x in a.episodes.split(',')] if a.episodes else sorted(meta)
    t = pq.read_table(sorted((a.v1 / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'action.effector.position',
                      'observation.state.effector.position', 'observation.state.ee.position'])
    ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
    C = np.array(t.column('action.effector.position').to_pylist(), float); G = np.array(t.column('observation.state.effector.position').to_pylist(), float)
    E = np.array(t.column('observation.state.ee.position').to_pylist(), float)
    out = {1: {}, 2: {}}; why = {1: collections.Counter(), 2: collections.Counter()}; cells = {1: [], 2: []}
    F = ImageFont.load_default(size=12)
    for e in want:
        if e not in meta:
            continue
        did, text, _ = lab.get(e, (f'ep{e:04d}', '', None))
        text = instr.get(e, text)
        pr = YC.parse(text)
        if pr is None:
            continue
        task, tcls, dcls = pr
        m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; c, g, ee = C[m], G[m], E[m]
        gf = grasp_frame(c)
        if task == 2 or a.own:
            arms = [x for x in (0, 1) if (c[:, x] > YC.CMD_OPEN).any()]
            ok = gf is not None and len(arms) == 1
            if ok:
                x = arms[0]
                hold = float(np.median(g[gf + 15:gf + 30, x]) - np.median(g[:15, x])); lift = float(ee[gf:gf + 60, 7 * x + 2].max() - ee[gf, 7 * x + 2])
                ok = hold > 0.02 and lift > 0.03
            if not ok:
                why[task]['grasp rule does not fit'] += 1; continue
        rgb = frame0(a.v1, meta[e])
        tb, db, reason = YC.select(det(rgb), task, tcls, dcls, rgb=rgb)
        why[task][reason] += 1
        if reason != 'ok':
            continue
        out[task][str(e)] = {'id': did, 'instruction': text, 'target': tcls, 'dest': dcls, 'target_box': tb, 'dest_box': db,
                             'grasp_frame': gf, 'T': int(len(m))}
        cell = Image.fromarray(YC.draw(rgb, tb, db, False)).resize((320, 240)); d = ImageDraw.Draw(cell)
        d.rectangle([0, 0, 320, 15], fill=(255, 255, 255)); d.text((2, 1), f'{did} {tcls}->{dcls}', font=F, fill=(0, 0, 0))
        cells[task].append(cell)
    for task in (1, 2):
        n = sum(why[task].values())
        summ = {'episodes_considered': n, 'with_cue': len(out[task]), 'reasons': dict(why[task]), 'yolo': 'v7a', 'mode': 'boxgrasp', 'line_px': YC.LINE,
                'colors': {'target': list(YC.WHITE), 'dest': list(YC.ORANGE)}, 'own': a.own}
        Path(f'{a.out}_t{task}.json').write_text(json.dumps({'summary': summ, 'episodes': out[task]}, indent=1, ensure_ascii=False))
        print(f'task {task}:', json.dumps(summ, ensure_ascii=False))
        for k in range(0, len(cells[task]), 30):
            cc = cells[task][k:k + 30]; sheet = Image.new('RGB', (6 * 320, ((len(cc) + 5) // 6) * 240), (255, 255, 255))
            for i, im in enumerate(cc):
                sheet.paste(im, ((i % 6) * 320, (i // 6) * 240))
            sheet.save(f'{a.out}_t{task}_sheet{k // 30 + 1:02d}.jpg', quality=85)


if __name__ == '__main__':
    main()
