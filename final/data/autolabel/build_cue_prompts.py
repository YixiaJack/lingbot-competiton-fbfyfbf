"""Box-cue datasets with box-referencing prompts (11/15). For each source box dataset X (LeRobot v3, camera_top already
carries the boxgrasp cue) this writes two siblings that differ ONLY in the instruction text:
  X_named  'Pick up the {obj} in the white box and place it on the {color} plate in the orange box.'  (task 1)
           'Put the {item} in the white box into the {level} compartment in the orange box.'          (task 2)
  X_gen    'Pick up the object in the white box and place it on the plate in the orange box.'          (task 1)
           'Put the item in the white box into the compartment in the orange box.'                     (task 2)
The templates come from yolocue_bridge.cue_prompts (same file as online), so training and inference use identical text.
Only meta/ is written: tasks.parquet keeps one row per original task (same row order; the loaders look tasks up with
tasks.iloc[task_index], so duplicate strings are fine) and episodes/*.parquet get the new 'tasks' lists; data/ and
videos/ are symlinks to the source. Any instruction that does not parse aborts the dataset (nothing half-written).
usage: python3 build_cue_prompts.py --bridge ~/lingbot_train/tools/yolocue_bridge.py SRC_DIR [SRC_DIR ...]"""
import argparse
import glob
import importlib.util
import json
import os
import shutil

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--bridge', required=True)
ap.add_argument('--force', action='store_true', help='rebuild variants that already exist')
ap.add_argument('src', nargs='+')
a = ap.parse_args()
spec = importlib.util.spec_from_file_location('yolocue_bridge', os.path.expanduser(a.bridge))
YC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(YC)

for src in a.src:
    src = os.path.abspath(os.path.expanduser(src)).rstrip('/')
    tasks = pd.read_parquet(f'{src}/meta/tasks.parquet')
    assert list(tasks['task_index']) == list(range(len(tasks))), f'{src}: task_index is not the row position'
    prompts = {t: YC.cue_prompts(t) for t in tasks.index}
    bad = [t for t, p in prompts.items() if p is None]
    if bad:
        print(f'SKIP {src}: {len(bad)} instructions do not parse, e.g. {bad[:3]}')
        continue
    for k, variant in ((0, 'named'), (1, 'gen')):
        dst = f'{src}_{variant}'
        if os.path.exists(dst):
            if not a.force:
                print('exists, skipped (use --force):', dst)
                continue
            shutil.rmtree(dst) if not os.path.islink(dst) else os.unlink(dst)
        tmp = dst + '.tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(f'{tmp}/meta')
        for sub in ('data', 'videos'):
            os.symlink(f'{src}/{sub}', f'{tmp}/{sub}')
        for f in os.listdir(f'{src}/meta'):
            if f not in ('tasks.parquet', 'episodes') and os.path.isfile(f'{src}/meta/{f}'):
                shutil.copyfile(f'{src}/meta/{f}', f'{tmp}/meta/{f}')
        new = [prompts[t][k] for t in tasks.index]
        out = pd.DataFrame({'task_index': list(range(len(new)))}, index=pd.Index(new, name=tasks.index.name))
        out.to_parquet(f'{tmp}/meta/tasks.parquet')
        n_ep = 0
        for f in glob.glob(f'{src}/meta/episodes/**/*.parquet', recursive=True):
            ep = pd.read_parquet(f)
            ep['tasks'] = ep['tasks'].map(lambda arr: [prompts[t][k] for t in arr])
            rel = os.path.relpath(f, f'{src}/meta')
            os.makedirs(os.path.dirname(f'{tmp}/meta/{rel}'), exist_ok=True)
            ep.to_parquet(f'{tmp}/meta/{rel}')
            n_ep += len(ep)
        # check: read back, every row maps to the right prompt
        back = pd.read_parquet(f'{tmp}/meta/tasks.parquet')
        assert [back.iloc[i].name for i in range(len(back))] == new
        os.rename(tmp, dst)
        print(f'wrote {dst}: {len(new)} task rows ({len(set(new))} distinct), {n_ep} episodes; e.g. {new[0]!r}')
