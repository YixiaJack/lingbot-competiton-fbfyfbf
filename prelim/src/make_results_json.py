"""Collect per-task eval results into the competition's results.json template.

  python3 src/make_results_json.py --eval_dir /scratch/$USER/lingbot/eval/<EVAL_NAME> --team_id <id> --out results.json

Reads <eval_dir>/{demo_clean,demo_randomized}/<task>.result ("successes attempts", written by eval_one_task.sh).
Fails loudly if any of the 100 (task, setting) cells is missing or did not run 100 episodes.
"""

import argparse
import json
from pathlib import Path

SETTINGS = {"clean": "demo_clean", "randomized": "demo_randomized"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, required=True)
    ap.add_argument("--team_id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args()
    tasks = json.loads((Path(__file__).resolve().parent.parent / "configs" / "tasks50.json").read_text())

    results, missing = {}, []
    for key, setting in SETTINGS.items():
        results[key] = {}
        for t in tasks:
            f = args.eval_dir / setting / f"{t}.result"
            if not f.exists():
                missing.append(f"{setting}/{t}")
                results[key][t] = {"attempts": 0, "successes": 0}
                continue
            succ, att = map(int, f.read_text().split())
            if att != 100:
                missing.append(f"{setting}/{t} ({att} attempts)")
            results[key][t] = {"attempts": att, "successes": succ}
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} cells missing/incomplete: {missing[:10]} ...")
    args.out.write_text(json.dumps({"schema_version": 1, "team_id": args.team_id, "results": results}, indent=2, ensure_ascii=False))
    for key in SETTINGS:
        s = sum(v["successes"] for v in results[key].values())
        a = sum(v["attempts"] for v in results[key].values())
        print(f"{key:10s} {s}/{a} = {100 * s / max(a, 1):.2f}%")


if __name__ == "__main__":
    main()
