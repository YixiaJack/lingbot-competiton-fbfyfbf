#!/usr/bin/env python3
"""标注质检：每个类别拼一张裁剪图墙（标错的一眼能看出来），每 N 张原图拼一张带框缩略图，再出一份问题清单。

用法:
    python qa_gallery.py --images frames/ --labels labels/ --classes classes.txt --out qa/
输出:
    qa/crops/<id>_<类名>.jpg     该类所有框的裁剪（带少量上下文），下方小字是来源图片编号
    qa/sheets/sheet_000.jpg      原图缩略图 + 框 + 类名
    qa/report.txt                每类数量、没有框的图、过小的框、重复框、越界框
把 crops/ 下的图发给 Claude 帮忙查错也可以。
"""
import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)]


def font(size, path=None):
    for p in ([path] if path else []) + FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, TypeError):
            continue
    return ImageFont.load_default()


def read_labels(p: Path):
    out = []
    if not p.exists():
        return None
    for l in p.read_text().splitlines():
        s = l.split()
        if len(s) >= 5:
            out.append((int(s[0]), *map(float, s[1:5])))
    return out


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("qa"))
    ap.add_argument("--tile", type=int, default=112, help="裁剪图墙每格边长")
    ap.add_argument("--per-sheet", type=int, default=24, help="每张缩略图拼图放几张原图")
    ap.add_argument("--min-px", type=int, default=6, help="短边小于这个像素的框记为过小")
    ap.add_argument("--font", default=None)
    args = ap.parse_args()

    names = [l.strip() for l in args.classes.read_text(encoding="utf-8").splitlines() if l.strip()]
    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    (args.out / "crops").mkdir(parents=True, exist_ok=True)
    (args.out / "sheets").mkdir(parents=True, exist_ok=True)
    f_small, f_mid = font(11, args.font), font(14, args.font)

    crops = defaultdict(list)  # cls -> [(crop, short_name)]
    report = defaultdict(list)
    thumbs = []
    for idx, ip in enumerate(imgs):
        labels = read_labels(args.labels / (ip.stem + ".txt"))
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        if labels is None:
            report["没有标注文件"].append(ip.name)
            labels = []
        if not labels:
            report["标注为空（确认图里真的没有目标物体）"].append(ip.name)
        boxes = []
        for c, cx, cy, w, h in labels:
            if not 0 <= c < len(names):
                report["类别 id 越界"].append(f"{ip.name}: id={c}")
                continue
            x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
            if x0 < -1 or y0 < -1 or x1 > W + 1 or y1 > H + 1:
                report["框超出图像"].append(f"{ip.name}: {names[c]}")
            if min(x1 - x0, y1 - y0) < args.min_px:
                report["框过小"].append(f"{ip.name}: {names[c]} ({x1 - x0:.0f}x{y1 - y0:.0f}px)")
            for c2, b2 in boxes:
                v = iou((x0, y0, x1, y1), b2)
                if v > 0.85:
                    kind = "同类重复框" if c2 == c else "两个不同类别的框几乎重合（可能标错类）"
                    report[kind].append(f"{ip.name}: {names[c2]} / {names[c]} IoU={v:.2f}")
            boxes.append((c, (x0, y0, x1, y1)))
            pad_w, pad_h = 0.15 * (x1 - x0) + 4, 0.15 * (y1 - y0) + 4
            crop = im.crop((max(0, x0 - pad_w), max(0, y0 - pad_h), min(W, x1 + pad_w), min(H, y1 + pad_h)))
            crops[c].append((crop, f"#{idx}"))

        th = im.copy()
        d = ImageDraw.Draw(th)
        for c, (x0, y0, x1, y1) in boxes:
            col = PALETTE[c % len(PALETTE)]
            d.rectangle((x0, y0, x1, y1), outline=col, width=2)
            d.text((x0 + 2, max(0, y0 - 13)), names[c], fill=col, font=f_small)
        d.text((3, 3), f"#{idx} {ip.name[:40]}", fill=(255, 255, 255), font=f_small)
        thumbs.append(th)

    # 每类裁剪图墙
    T = args.tile
    for c in sorted(crops):
        items = crops[c]
        cols = min(10, len(items))
        rows = (len(items) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * T, rows * (T + 14) + 22), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        d.text((4, 3), f"[{c}] {names[c]}  共 {len(items)} 个框", fill=(255, 255, 255), font=f_mid)
        for i, (cr, tag) in enumerate(items):
            s = (T - 4) / max(cr.width, cr.height)  # 小物体放大、大物体缩小，统一铺满格子
            cr = cr.resize((max(1, round(cr.width * s)), max(1, round(cr.height * s))), Image.BICUBIC)
            x, y = (i % cols) * T, 22 + (i // cols) * (T + 14)
            sheet.paste(cr, (x + (T - cr.width) // 2, y + (T - cr.height) // 2))
            d.text((x + 3, y + T), tag, fill=(200, 200, 200), font=f_small)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in names[c])
        sheet.save(args.out / "crops" / f"{c:02d}_{safe}.jpg", quality=90)

    # 带框缩略图拼图
    if thumbs:
        tw = 320
        scaled = [t.resize((tw, round(t.height * tw / t.width))) for t in thumbs]
        th_h = max(s.height for s in scaled)
        cols = 4
        for s in range(0, len(scaled), args.per_sheet):
            group = scaled[s:s + args.per_sheet]
            rows = (len(group) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * tw, rows * th_h), (0, 0, 0))
            for i, t in enumerate(group):
                sheet.paste(t, ((i % cols) * tw, (i // cols) * th_h))
            sheet.save(args.out / "sheets" / f"sheet_{s // args.per_sheet:03d}.jpg", quality=88)

    with open(args.out / "report.txt", "w", encoding="utf-8") as fh:
        fh.write(f"图片 {len(imgs)} 张，框 {sum(len(v) for v in crops.values())} 个\n\n每类框数:\n")
        for c, n in enumerate(names):
            k = len(crops.get(c, []))
            fh.write(f"  [{c:2d}] {n}: {k}{'   <-- 没有样本' if k == 0 else ''}\n")
        for kind, items in report.items():
            fh.write(f"\n{kind}（{len(items)}）:\n")
            for it in items[:200]:
                fh.write(f"  {it}\n")
    print(f"-> {args.out}/crops ({len(crops)} 类), {args.out}/sheets, {args.out}/report.txt")


if __name__ == "__main__":
    main()
