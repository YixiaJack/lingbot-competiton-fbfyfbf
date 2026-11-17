"""Copy a LeRobot v3 dataset without some episodes (metadata filter, 11/15: mislabelled demos found after staging).
The output gets its own data table and episodes metadata (episode_index / index / dataset_from/to renumbered, info
totals, source_map filtered and renumbered); videos/ points at the source's video directory and every kept episode
keeps its video file + from/to timestamps; tasks.parquet is copied unchanged (task_index values stay valid).
The source is not modified (a running training may be reading it).
usage: python3 drop_episodes.py SRC_DIR OUT_DIR EP_INDEX [EP_INDEX ...]"""
import glob, json, os, shutil, sys
import pandas as pd

src, dst, drop = os.path.abspath(sys.argv[1]).rstrip('/'), os.path.abspath(sys.argv[2]).rstrip('/'), {int(x) for x in sys.argv[3:]}
info = json.load(open(f'{src}/meta/info.json'))
ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{src}/meta/episodes/*/*.parquet'))]).sort_values('episode_index')
data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{src}/data/*/*.parquet'))]).sort_values('index')
assert drop <= set(ep['episode_index']), f'not in {src}: {drop - set(ep["episode_index"])}'
tmp = dst + '.tmp'
shutil.rmtree(tmp, ignore_errors=True)
assert not os.path.exists(dst), f'{dst} exists'
os.makedirs(f'{tmp}/meta/episodes/chunk-000'); os.makedirs(f'{tmp}/data/chunk-000')
os.symlink(os.path.realpath(f'{src}/videos'), f'{tmp}/videos')
keep = ep[~ep['episode_index'].isin(drop)].copy()
old2new, rows, start = {}, [], 0
for new, (_, e) in enumerate(keep.iterrows()):
    d = data[data['episode_index'] == e['episode_index']].copy()
    assert len(d) == e['length'] == e['dataset_to_index'] - e['dataset_from_index'], (src, e['episode_index'])
    old2new[int(e['episode_index'])] = new
    d['episode_index'] = new
    d['index'] = range(start, start + len(d))
    rows.append(d)
    keep.loc[e.name, ['episode_index', 'dataset_from_index', 'dataset_to_index']] = [new, start, start + len(d)]
    start += len(d)
for c in ('data/chunk_index', 'data/file_index', 'meta/episodes/chunk_index', 'meta/episodes/file_index'):
    if c in keep.columns:
        keep[c] = 0
out = pd.concat(rows)
out.to_parquet(f'{tmp}/data/chunk-000/file-000.parquet', index=False)
keep.to_parquet(f'{tmp}/meta/episodes/chunk-000/file-000.parquet', index=False)
shutil.copyfile(f'{src}/meta/tasks.parquet', f'{tmp}/meta/tasks.parquet')
json.dump(dict(info, total_episodes=len(keep), total_frames=len(out), splits={'train': f'0:{len(keep)}'}), open(f'{tmp}/meta/info.json', 'w'), indent=4)
if os.path.exists(f'{src}/meta/stats.json'):
    shutil.copyfile(f'{src}/meta/stats.json', f'{tmp}/meta/stats.json')
if os.path.exists(f'{src}/meta/source_map.json'):
    sm = json.load(open(f'{src}/meta/source_map.json'))
    sm = [dict(r, episode_index=old2new[r['episode_index']]) for r in sm if r['episode_index'] not in drop]
    json.dump(sm, open(f'{tmp}/meta/source_map.json', 'w'), ensure_ascii=False)
os.rename(tmp, dst)
dropped = ep[ep['episode_index'].isin(drop)]
print(f'{os.path.basename(dst)}: {len(keep)} eps {len(out)} frames (dropped {sorted(drop)}: ' +
      '; '.join(str(t[0])[:60] for t in dropped['tasks']) + ')')
