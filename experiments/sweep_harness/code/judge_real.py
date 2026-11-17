"""Offline test of the default harness (Fixed, four subtasks) on the official real-robot task-3 demos. No VLA involved.

Each demo is replayed as the harness would see it on the robot: one query every --every frames (60 = 2 s at 30 fps),
three cameras + gripper readings, the same agent prompt and append-only conversation as harness.Fixed, closed loop (the
"running now" line is the model's own previous answer). Truth = segment_real.py labels at the queried frame:
  exact    answer == label at that frame
  lenient  answer == label at that frame or at the next query's frame (switching one query early is fine)
  side     the locked dustpan side == the demo's dustpan arm
  stop     when harness.Fixed would stop the robot: from the LLM's task_complete on, arm joints sampled every 0.5 s
           (a chunk) must stay still for Fixed.FINISH_STILL_S; compared with the moment the arms are really home
Usage: judge_real.py --data <.../v1> --labels real_t3_subtasks.json --out DIR [--n 24] [--every 60] [--workers 4]
Needs: numpy, pyarrow, av, pillow, openai.
"""
import argparse, json, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow.parquet as pq
import av
import harness as H
import real_split as R

CAMS = {'top': 'camera_top', 'left': 'camera_wrist_left', 'right': 'camera_wrist_right'}


def episode_meta(data):
    t = pq.read_table(sorted((data / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()
    return {r['episode_index']: r for r in t}


def decode(data, meta, cam, idx, size):
    """RGB frames at in-episode indices idx (ascending) of one camera, resized to size."""
    key = f'observation.images.{CAMS[cam]}'
    fi, t0 = int(meta[f'videos/{key}/file_index']), float(meta[f'videos/{key}/from_timestamp'])
    out, want = [], list(idx)
    with av.open(str(data / f'videos/{key}/chunk-000/file-{fi:03d}.mp4')) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 0.05, 0) / s.time_base), stream=s, backward=True)
        k = 0
        for fr in c.decode(s):
            if k >= len(want): break
            if float(fr.pts * s.time_base) + 1e-3 >= t0 + want[k] / 30.0:
                out.append(np.asarray(fr.to_image().resize(size))); k += 1
    return out


def simulate_stop(complete_t, gr, arm, step=15):
    """Time (s) at which harness.Fixed stops after the LLM's task_complete, replaying recorded joints; None if never."""
    if complete_t is None:
        return None
    f = SimpleNamespace(arm_hist=[], FINISH_STILL_S=H.Fixed.FINISH_STILL_S, FINISH_TOL=H.Fixed.FINISH_TOL)
    for j in range(int(complete_t * R.FPS), len(arm), step):
        t = j / R.FPS
        f.arm_hist = [(u, q) for u, q in f.arm_hist if u >= t - f.FINISH_STILL_S - 1.0] + [(t, arm[j])]
        if max(gr[j]) < 0.05 and H.Fixed.arms_still(f, t):
            return round(t, 2)
    return None


def arms_home_t(arm, tol=0.02):
    """Last moment any joint is more than tol away from its final value (the arms are home and still after it)."""
    far = np.where(np.abs(arm - arm[-1]).max(1) > tol)[0]
    return round((far[-1] + 1) / R.FPS, 2) if len(far) else 0.0


def run_episode(rec, data, meta, gr, every, model, size, arm):
    T, ep = rec['T'], rec['episode_index']
    idx = list(range(0, T, every))
    imgs = {c: decode(data, meta[ep], c, idx, size) for c in CAMS}
    labels = R.frame_labels(rec)
    sched = H.Fixed(model=model, img_size=size)
    clock = H.Clock(limit=1e9)
    rows = []
    for q, j in enumerate(idx):
        clock.actions, clock.blocked = 0, j / R.FPS   # Clock.now = j / 30 s
        obs = {'images': {c: imgs[c][q] for c in CAMS}, 'grippers': tuple(gr[j])}
        content, text = sched.user_turn(obs, clock, first=(q == 0))
        parsed, rec_ = sched.chat.ask(content, text)
        ok = sched.apply(parsed, clock.now, rec_)
        if not ok and q == 0:
            sched.fallback_start(0.0)
        tgt, nxt = str(labels[j]), str(labels[min(j + every, T - 1)])
        rows.append({'frame': j, 't': round(j / R.FPS, 2), 'target': tgt, 'answer': sched.subtask, 'exact': sched.subtask == tgt,
                     'lenient': sched.subtask in (tgt, nxt), 'valid': bool(ok), 'task_complete': sched.llm_complete_t is not None,
                     'latency_s': rec_['latency_s'], 'prompt_tokens': rec_['prompt_tokens'],
                     'cache_hit_tokens': rec_['cache_hit_tokens'], 'reply': rec_['reply'][:400], 'error': rec_['error']})
    s = sched.summary(); sched.close()
    first_done = next((r['t'] for r in rows if r['task_complete']), None)
    stop_t, home_t = simulate_stop(first_done, gr, arm), arms_home_t(arm)
    return {'episode_index': ep, 'id': rec['id'], 'pan_arm': rec['pan_arm'], 'side_answer': s['pan_side'],
            'side_ok': s['pan_side'] == rec['pan_arm'], 'b': [rec['b1'], rec['b2'], rec['b3'], T], 'rows': rows,
            'exact': float(np.mean([r['exact'] for r in rows])), 'lenient': float(np.mean([r['lenient'] for r in rows])),
            'first_complete_t': first_done, 'stop_t': stop_t, 'arms_home_t': home_t, 'home_start_t': round(rec['b3'] / R.FPS, 2), 'release_t': round(rec['release'] / R.FPS, 2),
            'latency_mean_s': s['latency_mean_s'], 'cache_hit_rate': s['cache_hit_rate'], 'n_invalid': s['n_invalid'],
            'n_side_mismatch': s['n_side_mismatch'], 'prompt_tokens': s['prompt_tokens'], 'cache_hit_tokens': s['cache_hit_tokens']}


def load_states(data, eps, key):
    """{episode: (T, d) array} of one state column, in frame order."""
    out = {e: [] for e in eps}
    for f in sorted((data / 'data').glob('*/*.parquet')):
        t = pq.read_table(f, columns=['episode_index', 'frame_index', key])
        ep = t.column('episode_index').to_numpy(); keep = np.isin(ep, list(eps))
        if not keep.any(): continue
        fi = t.column('frame_index').to_numpy()[keep]
        g = np.array(t.column(key).to_pylist(), dtype=float)[keep]
        for e in np.unique(ep[keep]):
            m = ep[keep] == e; out[int(e)].append((fi[m], g[m]))
    res = {}
    for e, parts in out.items():
        fi = np.concatenate([p[0] for p in parts]); g = np.concatenate([p[1] for p in parts])
        res[e] = g[np.argsort(fi)]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--n', type=int, default=24)
    p.add_argument('--episodes', default=''); p.add_argument('--every', type=int, default=60)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--model', default='deepseek-flash')
    p.add_argument('--size', default='320x240'); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    size = tuple(int(x) for x in a.size.split('x'))
    recs = {r['episode_index']: r for r in json.loads(a.labels.read_text())['episodes']}
    if a.episodes:
        eps = [int(x) for x in a.episodes.split(',')]
    else:   # balanced: half with the dustpan on the left, half on the right
        rng = random.Random(a.seed)
        eps = sorted(rng.sample([e for e, r in recs.items() if r['pan_arm'] == 'left'], a.n // 2)
                     + rng.sample([e for e, r in recs.items() if r['pan_arm'] == 'right'], a.n - a.n // 2))
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eps if not (a.out / f'{e:04d}.json').exists()]
    meta = episode_meta(a.data)
    gr = load_states(a.data, todo, 'observation.state.effector.position') if todo else {}
    arm = load_states(a.data, todo, 'observation.state.arm.position') if todo else {}

    def job(e):
        try:
            res = run_episode(recs[e], a.data, meta, gr[e], a.every, a.model, size, arm[e])
        except Exception as ex:
            print(json.dumps({'episode_index': e, 'error': repr(ex)[:300]}), flush=True); return
        (a.out / f'{e:04d}.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps({k: res[k] for k in ('episode_index', 'id', 'pan_arm', 'side_ok', 'exact', 'lenient', 'first_complete_t',
                                              'stop_t', 'arms_home_t', 'latency_mean_s', 'cache_hit_rate', 'n_invalid')}), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(job, todo))
    summarize(a.out, eps, a)


def summarize(out, eps, a):
    res = [json.loads((out / f'{e:04d}.json').read_text()) for e in eps if (out / f'{e:04d}.json').exists()]
    rows = [r for x in res for r in x['rows']]
    conf = {t: {s: 0 for s in R.SUBTASK_IDS + ['None']} for t in R.SUBTASK_IDS}
    for r in rows:
        conf[r['target']][str(r['answer'])] += 1
    per = {t: {'n': sum(conf[t].values()), 'exact': round(conf[t][t] / max(1, sum(conf[t].values())), 3)} for t in R.SUBTASK_IDS}
    early_done = [x for x in res if x['first_complete_t'] is not None and x['first_complete_t'] < x['release_t']]
    stops = [x for x in res if x.get('stop_t') is not None]
    pt = sum(x['prompt_tokens'] for x in res); ch = sum(x['cache_hit_tokens'] for x in res)
    s = {'episodes': len(res), 'queries': len(rows), 'every_frames': a.every, 'image_size': a.size, 'model': a.model,
         'exact': round(float(np.mean([r['exact'] for r in rows])), 3), 'lenient': round(float(np.mean([r['lenient'] for r in rows])), 3),
         'side_ok': sum(x['side_ok'] for x in res), 'per_subtask': per, 'confusion_target_to_answer': conf,
         'complete_before_release': len(early_done), 'never_complete': sum(x['first_complete_t'] is None for x in res),
         'llm_complete_before_arms_home': sum(x['first_complete_t'] is not None and x['first_complete_t'] < x.get('arms_home_t', 0) for x in res),
         'stop_before_arms_home': sum(x['stop_t'] < x['arms_home_t'] for x in stops),
         'stop_fired_within_demo': len(stops),
         'latency_mean_s': round(float(np.mean([r['latency_s'] for r in rows])), 3),
         'latency_p95_s': round(float(np.percentile([r['latency_s'] for r in rows], 95)), 3),
         'cache_hit_rate': round(ch / pt, 3) if pt else None, 'invalid': sum(x['n_invalid'] for x in res),
         'side_mismatch_answers': sum(x['n_side_mismatch'] for x in res), 'episode_list': eps}
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s))


if __name__ == '__main__':
    main()
