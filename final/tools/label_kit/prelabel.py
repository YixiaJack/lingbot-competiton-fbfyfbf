#!/usr/bin/env python3
"""用现有检测器给抽出来的图做预标注，输出 YOLO txt，X-AnyLabeling 可以直接导入后人工修正。

用法（需要 ultralytics）:
    python prelabel.py --weights best.pt --images frames/ --out frames/ --conf 0.25
输出:
    <out>/<图片同名>.txt    每行: class cx cy w h（归一化）
    <out>/classes.txt       类别名，按 id 顺序，X-AnyLabeling 导入 YOLO 标注时要用
没有可用的权重时（比如规则不允许带赛前训的 YOLO），可以用 YOLO-World 预标:
    python prelabel.py --weights yolov8s-worldv2.pt --world-classes classes.txt ...
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--world-classes", type=Path, default=None,
                    help="YOLO-World 用: 每行一个英文类名，顺序就是 class id")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    if args.world_classes:
        names = [l.strip() for l in args.world_classes.read_text(encoding="utf-8").splitlines() if l.strip()]
        model.set_classes(names)
    names = model.names if isinstance(model.names, list) else [model.names[i] for i in sorted(model.names)]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    for i in range(0, len(imgs), 32):
        for r in model.predict([str(p) for p in imgs[i:i + 32]], conf=args.conf, imgsz=args.imgsz, verbose=False):
            lines = []
            for c, xywhn in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                lines.append(f"{int(c)} " + " ".join(f"{v:.6f}" for v in xywhn))
            (args.out / (Path(r.path).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"预标注 {len(imgs)} 张 -> {args.out}，类别 {len(names)} 个")


if __name__ == "__main__":
    main()
