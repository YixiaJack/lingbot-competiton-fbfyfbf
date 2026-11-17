"""Build LeRobot v3 training datasets from the official real-robot dataset without re-encoding any video.

Each output episode is a frame range [start, end) of a source episode with its own task string. The output shares the
source videos (videos/ is a symlink) and only rewrites the data table, the per-episode metadata (frame ranges, video
from/to timestamps, per-episode stats), tasks, info and stats.

Modes:
  --mode t3_subtasks --labels real_t3_subtasks.json
        every task-3 demo becomes 4 sub-episodes (grasp / sweep / dump / home), each with that subtask's instruction
        (real_split.instruction, the exact string the harness sends)
  --mode whole --prefix T1_,T2_
        whole episodes whose demo id starts with one of the prefixes, original instruction
Writes OUT/meta/source_map.json: output episode -> (source episode, demo id, start, end, subtask).
Usage: derive_dataset.py --src <.../v1> --out OUT --mode ... [--exclude-ids ids.txt]
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

NUMERIC_SKIP = {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}


def load_src(src):
    info = json.loads((src / 'meta/info.json').read_text())
    eps = pq.read_table(sorted((src / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()
    eps = {e['episode_index']: e for e in eps}
    ids = {}
    for line in open(src / 'meta/labels.jsonl'):
        d = json.loads(line)
        cf = (d['labels'][0].get('custom_fields') or []) if d.get('labels') else []
        ids[d['episode_index']] = next((f['value'] for f in cf if f['name'] == 'id'), f"ep{d['episode_index']:04d}")   # our own data has no id
    return info, eps, ids


def stats_of(col):
    a = np.asarray(col, dtype=np.float64)
    if a.ndim == 1: a = a[:, None]
    return {'min': a.min(0).tolist(), 'max': a.max(0).tolist(), 'mean': a.mean(0).tolist(), 'std': a.std(0).tolist(),
            'count': [int(len(a))]}


def build(src, out, items, fps):
    """items: [{'src_ep', 'start', 'end', 'task', ...}] in output order."""
    info, eps, ids = load_src(src)
    data_files = sorted((src / 'data').glob('*/*.parquet'))
    assert len(data_files) == 1, 'expects one data file (the official snapshot has one)'
    table = pq.read_table(data_files[0])
    numeric = [n for n in table.schema.names if n not in NUMERIC_SKIP]
    tasks = []
    for it in items:
        if it['task'] not in tasks: tasks.append(it['task'])
    task_idx = {t: i for i, t in enumerate(tasks)}
    vkeys = [k for k, f in info['features'].items() if f.get('dtype') == 'video']
    parts, ep_rows, smap, gidx = [], [], [], 0
    for new_ep, it in enumerate(items):
        e = eps[it['src_ep']]
        a, b = e['dataset_from_index'] + it['start'], e['dataset_from_index'] + it['end']
        n = b - a
        t = table.slice(a, n)
        assert t.column('episode_index')[0].as_py() == it['src_ep'] and t.column('frame_index')[0].as_py() == it['start']
        fi = np.arange(n, dtype=np.int64)
        cols = {c: t.column(c) for c in table.schema.names}
        cols['frame_index'] = pa.array(fi)
        cols['timestamp'] = pa.array((fi / fps).astype(np.float32))
        cols['episode_index'] = pa.array(np.full(n, new_ep, dtype=np.int64))
        cols['index'] = pa.array(np.arange(gidx, gidx + n, dtype=np.int64))
        cols['task_index'] = pa.array(np.full(n, task_idx[it['task']], dtype=np.int64))
        parts.append(pa.table([cols[c] for c in table.schema.names], schema=table.schema))
        row = {'episode_index': new_ep, 'data/chunk_index': 0, 'data/file_index': 0,
               'dataset_from_index': gidx, 'dataset_to_index': gidx + n}
        for k in vkeys:
            t0 = e[f'videos/{k}/from_timestamp']
            row[f'videos/{k}/chunk_index'] = e[f'videos/{k}/chunk_index']
            row[f'videos/{k}/file_index'] = e[f'videos/{k}/file_index']
            row[f'videos/{k}/from_timestamp'] = t0 + it['start'] / fps
            row[f'videos/{k}/to_timestamp'] = t0 + it['end'] / fps
        row.update({'length': n, 'tasks': [it['task']], 'task_index': task_idx[it['task']]})
        for c in numeric:
            for s, v in stats_of(t.column(c).to_pylist()).items():
                row[f'stats/{c}/{s}'] = v
        row.update({'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0})
        ep_rows.append(row)
        smap.append({'episode_index': new_ep, 'src_episode': it['src_ep'], 'id': ids[it['src_ep']], 'start': it['start'],
                     'end': it['end'], **{k: v for k, v in it.items() if k not in ('src_ep', 'start', 'end')}})
        gidx += n
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data/chunk-000').mkdir(parents=True, exist_ok=True)
    (out / 'meta/episodes/chunk-000').mkdir(parents=True, exist_ok=True)
    data = pa.concat_tables(parts)
    pq.write_table(data, out / 'data/chunk-000/file-000.parquet', row_group_size=20000)
    pq.write_table(pa.Table.from_pylist(ep_rows), out / 'meta/episodes/chunk-000/file-000.parquet')
    pd.DataFrame({'task_index': list(range(len(tasks)))}, index=pd.Index(tasks)).to_parquet(out / 'meta/tasks.parquet')
    stats = {c: stats_of(data.column(c).to_pylist()) for c in numeric}
    (out / 'meta/stats.json').write_text(json.dumps(stats))
    info = dict(info)
    info.update(total_episodes=len(items), total_frames=int(gidx), total_tasks=len(tasks), splits={'train': f'0:{len(items)}'})
    (out / 'meta/info.json').write_text(json.dumps(info))
    (out / 'meta/source_map.json').write_text(json.dumps(smap, ensure_ascii=False))
    v = out / 'videos'
    if v.is_symlink() or v.exists():
        v.unlink()
    os.symlink((src / 'videos').resolve(), v)
    return {'episodes': len(items), 'frames': int(gidx), 'tasks': len(tasks)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['t3_subtasks', 'whole'], required=True)
    p.add_argument('--labels', type=Path); p.add_argument('--prefix', default='T1_,T2_')
    p.add_argument('--exclude-ids', type=Path, help='demo ids to leave out (one per line), e.g. a val split')
    a = p.parse_args()
    info, eps, ids = load_src(a.src)
    fps = info['fps']
    skip = set(a.exclude_ids.read_text().split()) if a.exclude_ids else set()
    items = []
    if a.mode == 't3_subtasks':
        for r in json.loads(a.labels.read_text())['episodes']:
            if r['id'] in skip: continue
            for s in r['segments']:
                items.append({'src_ep': r['episode_index'], 'start': s['start'], 'end': s['end'], 'task': s['instruction'],
                              'subtask': s['subtask'], 'pan_arm': r['pan_arm']})
    else:
        pre = tuple(a.prefix.split(','))
        for e in sorted(eps):
            if ids[e].startswith(pre) and ids[e] not in skip:
                items.append({'src_ep': e, 'start': 0, 'end': eps[e]['length'], 'task': eps[e]['tasks'][0]})
    print(json.dumps(build(a.src, a.out, items, fps)))


if __name__ == '__main__':
    main()
