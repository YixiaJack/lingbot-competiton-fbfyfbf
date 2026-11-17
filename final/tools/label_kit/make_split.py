#!/usr/bin/env python3
"""把标好的图按 episode/布局分组、按任务分层，切成 YOLO 训练集和测试集。

用法:
    python make_split.py --frames frames/ --labels labels/ --classes classes.txt \
        --out yolo_ds/ --test-frac 0.25 [--holdout holdout.txt] [--group-col layout]
--holdout: 和 LingBot 验证集共用的留出条目，每行 "数据集名 episode_index"，这些一律进测试集
--group-col: 默认按 (dataset, episode_index) 分组；同一摆法录了多条时，在 manifest 的 layout 列
             填同一个编号再用 --group-col layout，保证同一布局不会同时出现在训练和测试里
输出 YOLO 目录: images/{train,val}, labels/{train,val}, data.yaml, split.csv
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True, help="extract_frames.py 的输出目录（含 manifest.csv）")
    ap.add_argument("--labels", type=Path, required=True, help="YOLO txt 标注目录（X-AnyLabeling 导出）")
    ap.add_argument("--classes", type=Path, required=True, help="classes.txt，每行一个类名")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--holdout", type=Path, default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = [l.strip() for l in args.classes.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = list(csv.DictReader(open(args.frames / "manifest.csv", encoding="utf-8")))
    holdout = set()
    if args.holdout:
        for l in args.holdout.read_text(encoding="utf-8").splitlines():
            if l.strip():
                ds, ep = l.split()
                holdout.add((ds, int(ep)))

    missing = [r["image"] for r in rows if not (args.labels / (Path(r["image"]).stem + ".txt")).exists()]
    if missing:
        print(f"警告: {len(missing)} 张没有标注文件，跳过（前 5 个: {missing[:5]}）")
    rows = [r for r in rows if r["image"] not in set(missing)]

    def gkey(r):
        if args.group_col and r.get(args.group_col):
            return ("layout", r[args.group_col])
        return (r["dataset"], int(r["episode_index"]))

    groups = defaultdict(list)
    for r in rows:
        groups[gkey(r)].append(r)

    # 分层: 每个 task_type 内部按组随机抽 test_frac 进测试集；holdout 条目所在的组强制进测试集
    rng = random.Random(args.seed)
    by_task = defaultdict(list)
    for k, rs in groups.items():
        by_task[rs[0]["task_type"]].append(k)
    test_groups = {k for k, rs in groups.items() if any((r["dataset"], int(r["episode_index"])) in holdout for r in rs)}
    for t, ks in by_task.items():
        ks = sorted(ks)
        rng.shuffle(ks)
        need = max(1, round(len(ks) * args.test_frac)) if len(ks) > 1 else 0
        have = sum(1 for k in ks if k in test_groups)
        for k in ks:
            if have >= need:
                break
            if k not in test_groups:
                test_groups.add(k)
                have += 1

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: [0, 0])
    with open(args.out / "split.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "split", "task_type", "dataset", "episode_index"])
        for k, rs in groups.items():
            split = "val" if k in test_groups else "train"
            for r in rs:
                stem = Path(r["image"]).stem
                shutil.copy2(args.frames / r["image"], args.out / f"images/{split}" / r["image"])
                shutil.copy2(args.labels / f"{stem}.txt", args.out / f"labels/{split}" / f"{stem}.txt")
                w.writerow([r["image"], split, r["task_type"], r["dataset"], r["episode_index"]])
                stats[r["task_type"]][split == "val"] += 1

    yaml_lines = [f"path: {args.out.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    for t in sorted(stats):
        print(f"{t}: train {stats[t][0]}  test {stats[t][1]}")
    print(f"-> {args.out/'data.yaml'}")


if __name__ == "__main__":
    main()
