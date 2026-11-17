# Per-episode table + per-task summary of the first N episodes from RoboTwin eval client logs.
#   python3 episodes20.py <eval dir with demo_clean/ demo_randomized/> <tasks50.json> <out dir> [N=20]
# A log can hold several client runs: resumed ones (preemption; "resumed from <state>: {now_seed, now_id, suc}") and,
# when two pool workers claimed the same item at once, two clients writing interleaved. Each client prints
# "Success rate: k/n ..., current seed: s" after every episode, with k / n counted from the shared state file.
# The official sequence is the chain r_1..r_N (n = 1..N, seeds increasing, k_n - k_{n-1} = outcome in {0, 1}) that
# passes through every resume point and ends at the official N-episode count (the resume point at n = N, else the
# .result file, else the last n = N line); among equal choices the later-written record wins (it updated the state).
import csv, json, os, re, sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"step: (\d+) / (\d+)")
RATE = re.compile(r"Success rate: (\d+)/(\d+) => [\d.]+%, current seed: (\d+)")
RESUME = re.compile(r"resumed from .*'now_seed': (\d+), 'now_id': (\d+), 'suc': (\d+)")
ST_SEED = 100000  # 100000 * (1 + seed), seed 0


def records(path):
    recs, resumes, cur_step, limit, forced, outcome, no_scene = [], [], 0, None, False, None, False
    if not os.path.exists(path):  # item not evaluated yet
        return recs, resumes
    for line in open(path, errors="ignore"):
        line = ANSI.sub("", line)
        m = RESUME.search(line)
        if m:
            s, n, k = map(int, m.groups())
            resumes.append((n, k, s - 1))  # state after episode n: k successes, next seed s
            cur_step, forced, outcome, no_scene = 0, False, None, False
            continue
        for s, lim in STEP.findall(line):
            cur_step, limit = max(cur_step, int(s)), int(lim)
        if "expert failed 5 times" in line:
            forced = True
        if "episode counted as failure" in line:
            outcome, no_scene = 0, True
        elif "Success!" in line:
            outcome = 1
        elif "Fail!" in line:
            outcome = 0
        m = RATE.search(line)
        if m:
            k, n, seed = map(int, m.groups())
            recs.append(dict(k=k, n=n, seed=seed, ok=outcome, steps=None if no_scene else cur_step, limit=limit,
                             forced=forced, no_scene=no_scene, pos=len(recs)))
            cur_step, forced, outcome, no_scene = 0, False, None, False
    return recs, resumes


def chain(recs, resumes, N, end_k):
    """Official episodes 1..N (list of records) or None."""
    by_n = {}
    for r in recs:
        by_n.setdefault(r["n"], []).append(r)
    fixed = {}  # n -> (k, last seed) from resume points; the seed of a resume point is the last tried seed, which is
    for n, k, s in resumes:  # >= the episode's seed (an expert failure after the episode is not possible: seeds move
        fixed[n] = (k, s)    # on only when an episode starts), so check k and seed <= s
    def ok(r, prev_k):
        d = r["k"] - prev_k
        return d in (0, 1) and (r["ok"] is None or r["ok"] == d)

    def back(n, nxt):  # choose r_n given r_{n+1} (None for n = N)
        cands = sorted(by_n.get(n, []), key=lambda r: -r["pos"])
        if not cands:  # line lost (two clients writing one log): an episode known only from its neighbours' counts
            top = end_k if nxt is None else nxt["k"]
            cands = [dict(n=n, k=kk, seed=None, ok=None, steps=None, limit=None, forced=False, no_scene=False,
                          pos=-1, lost=True) for kk in ((top, top - 1) if top is not None else ()) if 0 <= kk <= n]
        for r in cands:
            if n in fixed and (r["k"] != fixed[n][0] or (r["seed"] is not None and r["seed"] > fixed[n][1])):
                continue
            if nxt is None:
                if end_k is not None and r["k"] != end_k:
                    continue
            elif not ((r["seed"] is None or nxt["seed"] is None or r["seed"] < nxt["seed"]) and ok(nxt, r["k"])):
                continue
            if n == 1:
                if ok(r, 0):
                    return [r]
                continue
            rest = back(n - 1, r)
            if rest is not None:
                return rest + [r]
        return None
    eps = back(N, None)
    for i, r in enumerate(eps or []):  # a lost episode's seed is known when its neighbours are two seeds apart
        if r.get("lost") and 0 < i < len(eps) - 1 and None not in (eps[i - 1]["seed"], eps[i + 1]["seed"]) \
                and eps[i + 1]["seed"] - eps[i - 1]["seed"] == 2:
            r["seed"] = eps[i - 1]["seed"] + 1
    return eps


def main():
    root, tasks_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    tasks = json.load(open(tasks_json))
    cfgs = ("demo_clean", "demo_randomized")
    res, problems, itype, multi = {}, [], {}, {}
    for cfg in cfgs:
        for t in tasks:
            recs, resumes = records(f"{root}/{cfg}/eval_logs/{t}.log")
            end_k = next((k for n, k, s in reversed(resumes) if n == N), None)
            rf = f"{root}/{cfg}/{t}.result"
            if end_k is None and os.path.exists(rf):
                a, b = map(int, open(rf).read().split())
                end_k = a if b == N else None
            eps = chain(recs, resumes, N, end_k)
            if eps is None:
                problems.append(f"{cfg}/{t}: no consistent {N}-episode sequence"); eps = []
            dup_n = {n for n in range(1, N + 1) if sum(r["n"] == n for r in recs) > 1}
            if dup_n:
                multi[cfg, t] = sorted(dup_n)
            out_eps = []
            prev_k = 0
            for r in eps:
                e = dict(r, ok=r["k"] - prev_k, dup=r["n"] in dup_n and not r.get("lost"))
                prev_k = r["k"]
                out_eps.append(e)
            res[cfg, t] = out_eps
            rt = f"{root}/{cfg}/eval_results/{t}/_result.txt"
            m = re.search(r"Instruction Type: (\w+)", open(rt).read()) if os.path.exists(rt) else None
            itype[cfg, t] = m.group(1) if m else "?"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/每回合明细.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["setting", "任务", "回合", "种子", "结果", "用了多少步", "步数上限", "这一回合前跳过的种子数（专家失败）",
                    "专家5次都失败后照样评测", "备注"])
        for cfg in cfgs:
            for t in tasks:
                prev = ST_SEED - 1
                for e in res[cfg, t]:
                    note = "场景无法运行，直接计为失败" if e["no_scene"] else ""
                    if e["dup"]:
                        note = (note + "；" if note else "") + "这一回合有两个评测进程同时在跑，步数可能混记"
                    if e.get("lost"):
                        note = "日志里这一回合的记录被另一个进程覆盖，结果由前后两回合的成功计数推出" + \
                               ("" if e["seed"] is not None else "，种子无法确定")
                    gap = "" if e["seed"] is None or prev is None else e["seed"] - prev - 1
                    w.writerow([cfg, t, e["n"], "" if e["seed"] is None else e["seed"], "成功" if e["ok"] else "失败",
                                "" if e["steps"] is None or e["dup"] else e["steps"], e["limit"] or "",
                                gap, "是" if e["forced"] else "", note])
                    prev = e["seed"]
    win = lambda eps, a, b: sum(e["ok"] for e in eps[a - 1:b])
    with open(f"{out}/汇总.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["任务", "clean 成功/次数", "clean 成功率", "clean 第1-7回合", f"clean 第8-{N}回合",
                    "randomized 成功/次数", "randomized 成功率", "randomized 第1-7回合", f"randomized 第8-{N}回合"])
        tot = {c: [0, 0] for c in cfgs}
        for t in tasks:
            row = [t]
            for cfg in cfgs:
                eps = res[cfg, t]
                s = sum(e["ok"] for e in eps)
                tot[cfg][0] += s; tot[cfg][1] += len(eps)
                row += [f"{s}/{len(eps)}", f"{100 * s / max(1, len(eps)):.0f}%", f"{win(eps, 1, 7)}/7",
                        f"{win(eps, 8, N)}/{N - 7}"]
            w.writerow(row)
        w.writerow(["合计"] + sum(([f"{tot[c][0]}/{tot[c][1]}", f"{100 * tot[c][0] / max(1, tot[c][1]):.1f}%",
                                    f"{sum(win(res[c, t], 1, 7) for t in tasks)}/{7 * len(tasks)}",
                                    f"{sum(win(res[c, t], 8, N) for t in tasks)}/{(N - 7) * len(tasks)}"]
                                   for c in cfgs), []))
    stats = {}
    for cfg in cfgs:
        all_eps = [e for t in tasks for e in res[cfg, t]]
        skipped = 0
        for t in tasks:
            prev = ST_SEED - 1
            for e in res[cfg, t]:
                if e["seed"] is not None and prev is not None:
                    skipped += e["seed"] - prev - 1
                prev = e["seed"]
        forced = [e for e in all_eps if e["forced"]]
        stats[cfg] = dict(
            succ=sum(e["ok"] for e in all_eps), n=len(all_eps),
            w17=sum(win(res[cfg, t], 1, 7) for t in tasks), w810=sum(win(res[cfg, t], 8, 10) for t in tasks),
            w1120=sum(win(res[cfg, t], 11, N) for t in tasks), skipped=skipped, forced=len(forced),
            forced_ok=sum(e["ok"] for e in forced), no_scene=sum(e["no_scene"] for e in all_eps),
            itype=sorted({itype[cfg, t] for t in tasks}))
    print(json.dumps(stats, ensure_ascii=False))
    print("problems:", problems or "none")
    print("lost lines:", [f"{c}/{t} ep{e['n']} seed {e['seed']}" for (c, t), v in res.items() for e in v if e.get("lost")])
    print("items with two clients:", {f"{c}/{t}": v for (c, t), v in multi.items()} or "none")


if __name__ == "__main__":
    main()
