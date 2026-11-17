"""results.json over the first N episodes of every (setting, task), from the evaluation client logs.

  python3 src/make_results_first_n.py --eval_dir /path/to/eval/<name> --episodes 10 --team_id <id> --out results.json

src/make_results_json.py reads the <task>.result files ("successes attempts"), i.e. all episodes run so far. When an
evaluation has gone on past N episodes (or stopped at different counts for different tasks), this script takes, for
each (setting, task), the client's "Success rate: k/N ..." line from <eval_dir>/<setting>/eval_logs/<task>.log: k is
the number of successes among the first N episodes (the episode order is fixed by the seeds, see README). Fails if any
of the 100 cells has fewer than N episodes. Output format and task order as the official template.
"""

import argparse
import json
import re
from pathlib import Path

SETTINGS = {"clean": "demo_clean", "randomized": "demo_randomized"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RATE = re.compile(r"Success rate: (\d+)/(\d+) => [\d.]+%, current seed: (\d+)")


def successes_at(log, n):
    """k of the last "Success rate: k/n" record in a client log (a resumed client repeats no episode), or None."""
    k = None
    for line in open(log, errors="ignore"):
        if "Success" in line:
            for m in RATE.finditer(ANSI.sub("", line)):
                if int(m.group(2)) == n:
                    k = int(m.group(1))
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--team_id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tasks = json.loads((Path(__file__).resolve().parent.parent / "configs" / "tasks50.json").read_text())
    results, missing = {}, []
    for key, setting in SETTINGS.items():
        results[key] = {}
        for t in tasks:
            log = args.eval_dir / setting / "eval_logs" / f"{t}.log"
            k = successes_at(log, args.episodes) if log.exists() else None
            if k is None:
                missing.append(f"{setting}/{t}")
            results[key][t] = {"attempts": args.episodes, "successes": k or 0}
    if missing:
        raise SystemExit(f"{len(missing)} cells have fewer than {args.episodes} episodes: {missing[:10]} ...")
    args.out.write_text(json.dumps({"schema_version": 1, "team_id": args.team_id, "results": results}, indent=2,
                                   ensure_ascii=False))
    for key in results:
        s = sum(v["successes"] for v in results[key].values())
        print(f"{key}: {s}/{args.episodes * len(tasks)} = {100 * s / (args.episodes * len(tasks)):.1f}%")


if __name__ == "__main__":
    main()
