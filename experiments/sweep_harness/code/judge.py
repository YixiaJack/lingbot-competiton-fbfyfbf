"""Offline test of the harness LLM's progress judgement on held-out expert demonstrations (no VLA involved).

Each demo is replayed as a harness episode would see it: one query every --every frames (50 = one VLA chunk, 3 s),
the same agent prompts and append-only conversation as harness.py, closed loop (the "running now" line is the model's
own previous answer). Compared with the expert's skill label at the queried frame:
  exact    answer == label at that frame
  lenient  answer == label at that frame or at the next query's frame (switching one query early is fine)
v2 is scored on subtasks (prompts.SUBTASK_OF_SKILL); its written instructions are kept for review, with a keyword check
that the instruction names the right object (dustpan / brush / the colour of a ball still outside / basket).
Usage: judge.py --root R --version v1|v2 [--items 150,151] [--every 50] [--model deepseek-flash]
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import harness as H
from prompts import SUBTASK_OF_SKILL
from skills import relabel

DT = H.DT


class Replay:
    """Minimal obs dict for harness.LLMScheduler.user_turn from a recorded frame."""
    def __init__(self, data, j):
        self.obs = {'joint_action': {'vector': data['state'][j]},
                    'observation': {'head_camera': {'rgb': data['cam_high'][j]}, 'left_camera': {'rgb': data['cam_left_wrist'][j]},
                                    'right_camera': {'rgb': data['cam_right_wrist'][j]}}}


def keyword_ok(instr, label, flags):
    s = instr.lower()
    if label == 'grasp_dustpan': return 'dustpan' in s and 'brush' not in s.split('dustpan')[0]
    if label == 'grasp_brush': return 'brush' in s and 'dustpan' not in s
    if label.startswith('sweep'):
        out = [c for c, inside in (('red', flags[0]), ('white', flags[1])) if not inside]
        return any(c in s for c in out) or ('both' in s or 'two' in s or 'all' in s)
    if label in ('carry_to_basket', 'pour'): return 'basket' in s
    return 'home' in s or 'down' in s or 'place' in s or 'put' in s or 'release' in s


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--version', choices=['v1', 'v2'], required=True); p.add_argument('--items', default='')
    p.add_argument('--every', type=int, default=50); p.add_argument('--model', default='deepseek-flash')
    p.add_argument('--thinking', type=int, default=0); p.add_argument('--tag', default='')
    a = p.parse_args()
    items = [int(x) for x in (a.items.split(',') if a.items else (a.root / 'lists/val_items.txt').read_text().split())]
    outdir = a.root / 'judge' / f'{a.version}_{a.model}{a.tag}'; outdir.mkdir(parents=True, exist_ok=True)
    for item in items:
        out = outdir / f'{item:04d}.json'
        if out.exists():
            continue
        d = a.root / 'episodes' / f'{item:04d}'
        with np.load(d / 'episode.npz') as z:
            data = {k: z[k] for k in z.files}
        labels = relabel(data['skill'], data['flags'])   # same boundaries as the training segments
        sched = H.make_scheduler(a.version, model=a.model, thinking=bool(a.thinking))
        clock = H.Clock(limit=1e9)
        rows, n = [], len(data['skill'])
        for j in range(0, n, a.every):
            clock.actions, clock.blocked = j, 0.0
            o = Replay(data, j).obs
            if j == 0:
                content, text = sched.user_turn(o, clock, first=True)
            else:
                content, text = sched.user_turn(o, clock, first=False)
            parsed, rec = sched.chat.ask(content, text)
            ok = sched.apply(parsed, clock.now, rec)
            if not ok and j == 0:
                sched.fallback_start(0.0)
            label = str(labels[j]); nxt = str(labels[min(j + a.every, n - 1)])
            if a.version == 'v1':
                ans, tgt, tgt2 = sched.current, label, nxt
            else:
                ans, tgt, tgt2 = sched.subtask, SUBTASK_OF_SKILL[label], SUBTASK_OF_SKILL[nxt]
            rows.append({'frame': j, 't': round(j * DT, 2), 'label': label, 'target': tgt, 'answer': ans,
                         'exact': ans == tgt, 'lenient': ans in (tgt, tgt2), 'valid': bool(ok),
                         'instruction': sched.current if a.version == 'v2' else None,
                         'keyword_ok': keyword_ok(sched.current, label, data['flags'][j]) if a.version == 'v2' else None,
                         'task_complete': sched.done, 'latency_s': rec['latency_s'],
                         'prompt_tokens': rec['prompt_tokens'], 'cache_hit_tokens': rec['cache_hit_tokens'],
                         'reply': rec['reply'][:400], 'error': rec['error']})
            sched.done = False   # keep replaying to the end even if the model says complete early
        summ = sched.summary(); sched.close()
        res = {'item': item, 'version': a.version, 'model': a.model, 'every': a.every, 'rows': rows,
               'exact': float(np.mean([r['exact'] for r in rows])), 'lenient': float(np.mean([r['lenient'] for r in rows])),
               'keyword_ok': float(np.mean([r['keyword_ok'] for r in rows])) if a.version == 'v2' else None,
               'latency_mean_s': summ['latency_mean_s'], 'cache_hit_rate': summ['cache_hit_rate'],
               'prompt_tokens': summ['prompt_tokens'], 'cache_hit_tokens': summ['cache_hit_tokens'], 'n_invalid': summ['n_invalid']}
        out.write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ('item', 'version', 'exact', 'lenient', 'keyword_ok', 'latency_mean_s', 'cache_hit_rate', 'n_invalid')}), flush=True)


if __name__ == '__main__':
    main()
