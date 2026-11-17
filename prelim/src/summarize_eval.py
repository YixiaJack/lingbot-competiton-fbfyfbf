"""Summarize one or more eval runs ($PROJ/eval/<name>/<setting>/): per-setting success rate, finished tasks, and a
per-task table. Unfinished tasks count with the episodes done so far (state/<task>.json from the resumable client).

  python3 src/summarize_eval.py --eval_root $PROJ/eval q50x10_scratch22k strong10k [--tasks 10] [--per_task]
"""

import argparse
import json
import os


def load(root, name, setting, task):
    r = os.path.join(root, name, setting, f"{task}.result")
    if os.path.exists(r):
        s, a = map(int, open(r).read().split())
        return s, a, True
    st = os.path.join(root, name, setting, "state", f"{task}.json")
    if os.path.exists(st):
        d = json.load(open(st))
        return d["suc"], d["test_num"], False
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--eval_root", default=os.path.expandvars("${PROJ}/eval") if os.environ.get("PROJ") else "eval")
    ap.add_argument("--tasks_file", default=os.path.join(os.path.dirname(__file__), "..", "configs", "tasks50.json"))
    ap.add_argument("--tasks", type=int, default=50, help="only the first N tasks of tasks50.json")
    ap.add_argument("--per_task", action="store_true")
    args = ap.parse_args()
    tasks = json.load(open(args.tasks_file))[: args.tasks]
    for setting in ("demo_clean", "demo_randomized"):
        print(f"== {setting}")
        rows = {t: [load(args.eval_root, n, setting, t) for n in args.names] for t in tasks}
        for i, n in enumerate(args.names):
            v = [r[i] for r in rows.values() if r[i]]
            s, a = sum(x[0] for x in v), sum(x[1] for x in v)
            done = sum(x[2] for x in v)
            rate = f"{100 * s / a:5.1f}%" if a else "   - "
            print(f"  {n:24s} {rate}  ({s}/{a} episodes, {done}/{len(tasks)} tasks finished, {len(v)} started)")
        if args.per_task:
            print("  " + " " * 24 + "".join(f"{n[:14]:>16s}" for n in args.names))
            for t, r in rows.items():
                cells = "".join(f"{(f'{x[0]}/{x[1]}' + ('' if x[2] else '*')) if x else '-':>16s}" for x in r)
                print(f"  {t:24s}{cells}")


if __name__ == "__main__":
    main()
