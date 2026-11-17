"""Head-camera first + last frame of every unique team episode in the private datasets (run on the dev machine).

Latest snapshot of each dataset under /mnt/forge/datasets/private (newest meta/info.json, as field_board_probe does).
An episode is identified by its first 30 measured states (timestamp_ns is relative, 0 at frame 0), so the same
recording in two datasets is kept once. Writes OUT/frames/<uid>_first.jpg, <uid>_last.jpg and OUT/index.json (instruction, result, score,
custom fields, recording date in CST).
usage: python3 ff_extract.py OUT
"""
import hashlib
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path('/mnt/forge/datasets/private')
CAM = 'observation.images.camera_top'


def grab(path, t_first, t_last, fps):
    """(first, last) RGB frames of one episode inside a concatenated mp4."""
    out = {}
    with av.open(str(path)) as ic:
        s = ic.streams.video[0]
        for key, t in (('first', t_first), ('last', t_last)):
            ic.seek(int(max(t - 0.5, 0) / s.time_base), stream=s, backward=True)
            best = None
            for fr in ic.decode(s):
                ft = float(fr.pts * s.time_base)
                if ft + 1e-3 < t:
                    best = fr if key == 'last' else best
                    continue
                best = fr
                break
            out[key] = best.to_ndarray(format='rgb24')
    return out['first'], out['last']


def main():
    out = Path(sys.argv[1]); (out / 'frames').mkdir(parents=True, exist_ok=True)
    latest = {}
    for info in ROOT.glob('*/snapshots/*/v1/meta/info.json'):
        ds, rank = info.parents[4].name, info.stat().st_mtime_ns
        if ds not in latest or rank > latest[ds][0]:
            latest[ds] = (rank, info.parent.parent)
    rows, seen = [], {}
    for ds, (_, base) in sorted(latest.items(), key=lambda x: -x[1][0]):   # newest snapshot first
        fps = json.loads((base / 'meta/info.json').read_text())['fps']
        lab = {}
        for line in open(base / 'meta/labels.jsonl'):
            r = json.loads(line)
            lab[r['episode_index']] = (r.get('labels') or [{}])[0]
        t = pq.read_table(sorted((base / 'data').glob('*/*.parquet'))[0],
                          columns=['episode_index', 'frame_index', 'observation.state.arm.position', 'observation.state.effector.position'])
        ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
        arm = np.array(t.column('observation.state.arm.position').to_pylist(), dtype=np.float64)
        eff = np.array(t.column('observation.state.effector.position').to_pylist(), dtype=np.float64)
        start = {}
        for e in np.unique(ep):   # identity = the first 30 measured states (the same recording gives the same numbers)
            m = np.where((ep == e) & (fi < 30))[0]; m = m[np.argsort(fi[m])]
            start[int(e)] = hashlib.sha256(np.round(np.hstack([arm[m], eff[m]]), 4).tobytes()).hexdigest()[:16]
        meta = pq.read_table(sorted((base / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()
        for m in meta:
            e = int(m['episode_index']); uid = start.get(e)
            if uid is None:
                continue
            if uid in seen:
                seen[uid]['also_in'].append(f'{ds[:8]}/{e}')
                continue
            L = lab.get(e, {})
            vid = base / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"
            t0, t1 = m[f'videos/{CAM}/from_timestamp'], m[f'videos/{CAM}/to_timestamp']
            name = f'{ds[:8]}_{e:04d}'
            try:
                f0, f1 = grab(vid, t0, t1 - 1.0 / fps, fps)
                Image.fromarray(f0).save(out / 'frames' / f'{name}_first.jpg', quality=93)
                Image.fromarray(f1).save(out / 'frames' / f'{name}_last.jpg', quality=88)
                err = None
            except Exception as ex:   # keep going; the row says what failed
                err = repr(ex)
            row = {'uid': name, 'ds': ds, 'snapshot': base.parent.name, 'ep': e, 'trajectory_id': uid,
                   'length': int(m['length']), 'tasks': list(m.get('tasks') or []),
                   'instruction': L.get('task_instruction'), 'result': L.get('result'), 'score': L.get('score'),
                   'custom': {c.get('name'): c.get('value') for c in (L.get('custom_fields') or [])},
                   'also_in': [], 'error': err}
            seen[uid] = row; rows.append(row)
            print(name, (row['instruction'] or '')[:50], err or '', flush=True)
    (out / 'index.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print('SUMMARY', json.dumps({'unique': len(rows), 'duplicates': sum(len(r['also_in']) for r in rows),
                                 'errors': sum(1 for r in rows if r['error'])}))


if __name__ == '__main__':
    main()
