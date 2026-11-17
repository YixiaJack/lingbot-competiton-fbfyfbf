#!/usr/bin/env python3
"""从 LeRobot 数据集里抽头顶相机帧（默认每条 episode 的第 0 帧），给检测器标注用。

用法（开发机 lingbotvla 环境里有 pyarrow / torchcodec）:
    python extract_frames.py --out frames/ <数据集根目录>...
    python extract_frames.py --inspect <数据集根目录>     # 先看一眼 meta 结构对不对
可选:
    --key observation.images.camera_top   相机键
    --frames 0                            要抽的帧号，可多个: --frames 0 30
    --backend auto|torchcodec|ffmpeg      视频解码方式（auto: 有 torchcodec 就用，否则 ffmpeg）

输出:
    <out>/<数据集名>__ep000012__f0000.png
    <out>/manifest.csv   列: image,dataset,episode_index,frame_index,task,task_type,layout
      task_type 按指令关键词判断: t1 单臂抓放 / t2 化妆品收纳 / t3 垃圾清扫 / unknown
      layout 留空，同一摆法录了多条时手动填同一个编号，划分测试集时按它分组
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# 注意: 任务 2 的物体里有 "makeup brush"，所以任务 3 不能用 brush 判断，只用 dustpan / sweep
TASK_PATTERNS = [
    ("t3", re.compile(r"\b(dustpan|sweep)\b", re.I)),
    ("t2", re.compile(r"\b(compartment|organizer|section)\b", re.I)),
    ("t1", re.compile(r"\bplate\b", re.I)),
]


def task_type(text: str) -> str:
    for name, pat in TASK_PATTERNS:
        if pat.search(text or ""):
            return name
    return "unknown"


def load_info(root: Path) -> dict:
    p = root / "meta" / "info.json"
    if not p.is_file():
        sys.exit(f"找不到 {p}，不是 LeRobot 数据集根目录")
    return json.loads(p.read_text())


def read_parquet_rows(path: Path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pylist()


def load_episodes(root: Path, info: dict):
    """返回 [{episode_index, tasks:[...], **video 定位字段}]，兼容 v3.0 和 v2.1。"""
    ver = str(info.get("codebase_version", ""))
    if ver.startswith("v3"):
        files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        rows = []
        for f in files:
            rows += read_parquet_rows(f)
        return rows
    # v2.1: meta/episodes.jsonl
    p = root / "meta" / "episodes.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def video_location(root: Path, info: dict, ep: dict, key: str):
    """返回 (视频文件, 该 episode 第 0 帧在文件里的时间戳秒)。"""
    ver = str(info.get("codebase_version", ""))
    tmpl = info["video_path"]
    if ver.startswith("v3"):
        chunk = ep[f"videos/{key}/chunk_index"]
        file = ep[f"videos/{key}/file_index"]
        t0 = float(ep.get(f"videos/{key}/from_timestamp", 0.0))
        rel = tmpl.format(video_key=key, chunk_index=chunk, file_index=file)
    else:
        chunks_size = int(info.get("chunks_size", 1000))
        epi = int(ep["episode_index"])
        rel = tmpl.format(video_key=key, episode_chunk=epi // chunks_size, episode_index=epi)
        t0 = 0.0
    return root / rel, t0


class Decoder:
    def __init__(self, backend: str):
        if backend == "auto":
            try:
                import torchcodec  # noqa: F401
                backend = "torchcodec"
            except ImportError:
                backend = "ffmpeg"
        self.backend = backend
        self._cache = {}

    def frame(self, video: Path, t: float, fps: float, out_png: Path):
        if self.backend == "torchcodec":
            from torchcodec.decoders import VideoDecoder
            from PIL import Image
            dec = self._cache.get(video)
            if dec is None:
                self._cache.clear()  # 一次只开一个文件，省内存
                dec = self._cache[video] = VideoDecoder(str(video))
            fr = dec.get_frame_played_at(t + 0.5 / fps)  # 取这一帧显示区间的中点，避免浮点误差跳帧
            Image.fromarray(fr.data.permute(1, 2, 0).cpu().numpy()).save(out_png)
        else:
            ss = max(0.0, t - 0.5 / fps)  # ffmpeg 取 pts >= ss 的第一帧
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{ss:.6f}", "-i", str(video),
                   "-frames:v", "1", str(out_png)]
            subprocess.run(cmd, check=True)


def extract_image_dtype(root: Path, key: str, frames: set, out: Path, dsname: str, tasks_by_ep: dict, writer):
    """相机以 image 类型存在 parquet 里（不是视频）时走这里。"""
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    n = 0
    for f in files:
        for r in read_parquet_rows(f, columns=[key, "episode_index", "frame_index"]):
            if int(r["frame_index"]) not in frames:
                continue
            img = r[key]
            data = img.get("bytes") if isinstance(img, dict) else None
            if data is None and isinstance(img, dict) and img.get("path"):
                data = (root / img["path"]).read_bytes()
            ep, fi = int(r["episode_index"]), int(r["frame_index"])
            name = f"{dsname}__ep{ep:06d}__f{fi:04d}.png"
            from PIL import Image
            import io
            Image.open(io.BytesIO(data)).convert("RGB").save(out / name)
            task = tasks_by_ep.get(ep, "")
            writer.writerow([name, dsname, ep, fi, task, task_type(task), ""])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("frames"))
    ap.add_argument("--key", default="observation.images.camera_top")
    ap.add_argument("--frames", type=int, nargs="+", default=[0])
    ap.add_argument("--backend", default="auto", choices=["auto", "torchcodec", "ffmpeg"])
    ap.add_argument("--inspect", action="store_true", help="只打印 meta 结构，不抽帧")
    args = ap.parse_args()

    if args.inspect:
        for root in args.roots:
            info = load_info(root)
            print(f"== {root}\ncodebase_version: {info.get('codebase_version')}  fps: {info.get('fps')}")
            print("video_path:", info.get("video_path"))
            feats = info.get("features", {})
            print("camera features:", {k: v.get("dtype") for k, v in feats.items() if "image" in k})
            eps = load_episodes(root, info)
            print(f"episodes: {len(eps)}; 第一条的字段: {sorted(eps[0].keys()) if eps else []}")
            if eps:
                print("第一条示例:", {k: eps[0][k] for k in list(eps[0])[:12]})
        return

    args.out.mkdir(parents=True, exist_ok=True)
    dec = Decoder(args.backend)
    manifest = args.out / "manifest.csv"
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["image", "dataset", "episode_index", "frame_index", "task", "task_type", "layout"])
        for root in args.roots:
            info = load_info(root)
            fps = float(info.get("fps", 30))
            dsname = root.name if root.name not in ("v1", "") else root.parent.name  # 平台快照路径以 v1 结尾
            eps = load_episodes(root, info)
            tasks_by_ep = {int(e["episode_index"]): " | ".join(e.get("tasks") or []) for e in eps}
            dtype = info.get("features", {}).get(args.key, {}).get("dtype")
            if dtype is None:
                sys.exit(f"{root} 里没有 {args.key}，可用的: {[k for k in info.get('features', {}) if 'image' in k]}")
            if dtype == "image":
                n = extract_image_dtype(root, args.key, set(args.frames), args.out, dsname, tasks_by_ep, w)
                print(f"{root}: {n} 张 (image dtype)")
                continue
            n = 0
            for e in eps:
                ep = int(e["episode_index"])
                video, t0 = video_location(root, info, e, args.key)
                length = int(e.get("length", 10**9))
                for fi in args.frames:
                    if fi >= length:
                        continue
                    name = f"{dsname}__ep{ep:06d}__f{fi:04d}.png"
                    dec.frame(video, t0 + fi / fps, fps, args.out / name)
                    w.writerow([name, dsname, ep, fi, tasks_by_ep[ep], task_type(tasks_by_ep[ep]), ""])
                    n += 1
            print(f"{root}: {n} 张 (video, backend={dec.backend})")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
