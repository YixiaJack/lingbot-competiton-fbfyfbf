"""Split a task-3 subtask dataset (LeRobot v3, made by derive_dataset.py --mode t3_subtasks) into one dataset per
subtask, so a training list can weight grasp / sweep / dump / home separately. No video is re-encoded: videos/ points
at the source's video directory and every episode keeps its video file + from/to timestamps; only the data table,
episodes metadata (indices, row ranges), tasks and info are rewritten. stats.json is copied (normalization uses the
robot norm file, not per-dataset stats).
Subtask = prefix of the episode's instruction: Grasp -> grasp, Sweep -> sweep, Lift -> dump, Put the dustpan -> home.
usage: python3 split_t3_subtasks.py SRC_DIR [SRC_DIR ...]   -> SRC_DIR__grasp, __sweep, __dump, __home"""
import glob, json, os, shutil, sys
import pandas as pd

PREFIX = {'grasp': 'grasp ', 'sweep': 'sweep ', 'dump': 'lift ', 'home': 'put the dustpan'}


def subtask(t):
    t = t.lower()
    return next((k for k, p in PREFIX.items() if t.startswith(p)), None)


for src in sys.argv[1:]:
    src = os.path.abspath(src).rstrip('/')
    info = json.load(open(f'{src}/meta/info.json'))
    ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{src}/meta/episodes/*/*.parquet'))]).sort_values('episode_index')
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{src}/data/*/*.parquet'))]).sort_values('index')
    kinds = ep['tasks'].map(lambda a: subtask(a[0]))
    if kinds.isna().any():
        print('SKIP', src, 'unknown subtask:', ep.loc[kinds.isna(), 'tasks'].head(2).tolist()); continue
    vids = os.path.realpath(f'{src}/videos')
    for k in PREFIX:
        sel = ep[kinds == k].copy()
        if sel.empty:   # 11/15: takeover / follow-up DAgger sets do not cover every subtask
            print(f'{os.path.basename(src)}__{k}: no episodes, not written'); continue
        dst = f'{src}__{k}'; tmp = dst + '.tmp'
        shutil.rmtree(tmp, ignore_errors=True); shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(f'{tmp}/meta/episodes/chunk-000'); os.makedirs(f'{tmp}/data/chunk-000')
        os.symlink(vids, f'{tmp}/videos')
        tasks = sorted({a[0] for a in sel['tasks']})
        tix = {t: i for i, t in enumerate(tasks)}
        rows, start = [], 0
        for new, (_, e) in enumerate(sel.iterrows()):
            d = data[(data['index'] >= e['dataset_from_index']) & (data['index'] < e['dataset_to_index'])].copy()
            assert len(d) == e['length'] == e['dataset_to_index'] - e['dataset_from_index'], (src, k, e['episode_index'])
            d['episode_index'] = new
            d['index'] = range(start, start + len(d))
            d['task_index'] = tix[e['tasks'][0]]
            rows.append(d)
            sel.loc[e.name, ['episode_index', 'dataset_from_index', 'dataset_to_index', 'task_index']] = \
                [new, start, start + len(d), tix[e['tasks'][0]]]
            start += len(d)
        for c in ('data/chunk_index', 'data/file_index', 'meta/episodes/chunk_index', 'meta/episodes/file_index'):
            if c in sel.columns:
                sel[c] = 0
        out = pd.concat(rows)
        out.to_parquet(f'{tmp}/data/chunk-000/file-000.parquet', index=False)
        sel.to_parquet(f'{tmp}/meta/episodes/chunk-000/file-000.parquet', index=False)
        pd.DataFrame({'task_index': range(len(tasks))}, index=pd.Index(tasks, name='task')).to_parquet(f'{tmp}/meta/tasks.parquet')
        inf = dict(info, total_episodes=len(sel), total_frames=len(out), total_tasks=len(tasks), splits={'train': f'0:{len(sel)}'})
        json.dump(inf, open(f'{tmp}/meta/info.json', 'w'), indent=4)
        for f in ('stats.json', 'source_map.json'):
            if os.path.exists(f'{src}/meta/{f}'):
                shutil.copyfile(f'{src}/meta/{f}', f'{tmp}/meta/{f}')
        os.rename(tmp, dst)
        print(f'{os.path.basename(dst):34s} {len(sel):4d} eps {len(out):7d} frames')
    print(f'  source {os.path.basename(src)}: {len(ep)} eps {len(data)} frames')
