# Paired comparison of two evaluations on the same seeds (e.g. the cluster's 22k+8k vs a model evaluated on AMD with
# EVAL_SEED_LIST = the cluster run's seeds): per task and in total, how often both / only A / only B succeeded, and an exact
# sign test on the episodes where they differ.
#   python3 compare_paired.py A B [--names A_name B_name] [--n 20] [--out paired.csv]
# A / B: an eval dir (demo_clean/ and demo_randomized/ with eval_logs/, as written by src/eval_one_task.sh) or a
# 每回合明细.csv written by src/episode_table.py. Episodes are matched by (setting, task, seed). An episode with no
# runnable scene on one side only (the other side ran it) is a machine difference: listed, left out of the comparison.
import argparse, csv, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode_table import chain, records  # noqa: E402

CFGS = ("demo_clean", "demo_randomized")


def from_dir(root, tasks, n_max):
    out = {}
    for cfg in CFGS:
        for t in tasks:
            log = f"{root}/{cfg}/eval_logs/{t}.log"
            if not os.path.exists(log):
                continue
            recs, resumes = records(log)
            if not recs:
                continue
            N = min(n_max, max(r["n"] for r in recs))
            end_k = next((k for n, k, s in reversed(resumes) if n == N), None)
            rf = f"{root}/{cfg}/{t}.result"
            if end_k is None and os.path.exists(rf):
                a, b = map(int, open(rf).read().split())
                end_k = a if b == N else None
            eps = chain(recs, resumes, N, end_k) or []
            prev = 0
            for r in eps:
                out[cfg, t, r["seed"]] = dict(ok=r["k"] - prev, no_scene=r["no_scene"], n=r["n"])
                prev = r["k"]
    return out


def from_csv(path, n_max):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if int(r["回合"]) <= n_max and r["种子"]:
            out[r["setting"], r["任务"], int(r["种子"])] = dict(ok=int(r["结果"] == "成功"), n=int(r["回合"]),
                                                              no_scene="场景无法运行" in r["备注"])
    return out


def sign_test(b, c):  # two-sided exact binomial p for b vs c discordant pairs
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--n", type=int, default=20)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--tasks", default=next(p for p in (os.path.join(here, "tasks50.json"),
                                                        os.path.join(here, "..", "configs", "tasks50.json"))
                                            if os.path.exists(p) or p.endswith("configs/tasks50.json")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tasks = json.load(open(args.tasks))
    load = lambda p: from_csv(p, args.n) if p.endswith(".csv") else from_dir(p, tasks, args.n)
    A, B = load(args.a), load(args.b)
    na, nb = args.names
    rows, machine, tot = [], [], {c: [0, 0, 0, 0] for c in CFGS}  # both, only A, only B, neither
    for cfg in CFGS:
        for t in tasks:
            keys = sorted(k for k in A if k[:2] == (cfg, t) and k in B)
            cnt = [0, 0, 0, 0]
            for k in keys:
                a, b = A[k], B[k]
                if a["no_scene"] != b["no_scene"]:
                    machine.append((cfg, t, k[2], na if a["no_scene"] else nb))
                    continue
                cnt[{(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}[a["ok"], b["ok"]]] += 1
            n = sum(cnt)
            if n:
                rows.append([cfg, t, n, cnt[0] + cnt[1], cnt[0] + cnt[2], cnt[0], cnt[1], cnt[2],
                             f"{sign_test(cnt[1], cnt[2]):.3f}"])
                tot[cfg] = [x + y for x, y in zip(tot[cfg], cnt)]
    head = ["setting", "任务", "配对回合", f"{na} 成功", f"{nb} 成功", "都成功", f"只有 {na} 成功", f"只有 {nb} 成功",
            "符号检验 p"]
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(head); w.writerows(rows)
    for r in rows:
        print("  ".join(map(str, r)))
    for cfg in CFGS:
        both, oa, ob, nei = tot[cfg]
        n = both + oa + ob + nei
        if n:
            print(f"{cfg}: {n} paired episodes | {na} {both + oa}/{n} = {100 * (both + oa) / n:.1f}% | "
                  f"{nb} {both + ob}/{n} = {100 * (both + ob) / n:.1f}% | only {na} {oa}, only {nb} {ob}, "
                  f"sign test p = {sign_test(oa, ob):.3f}")
    if machine:
        print(f"left out ({len(machine)}): no runnable scene on one side only:",
              ", ".join(f"{c}/{t}/{s} ({who})" for c, t, s, who in machine))


if __name__ == "__main__":
    main()
