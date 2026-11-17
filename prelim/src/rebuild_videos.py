"""Rebuild the camera videos of our clean LeRobot dataset from RoboTwin raw demos, next to its uploaded meta/ + data/.

Used on the AMD server, whose upload link is too slow for the 2.25 GB of videos but which has the official clean_50
demos repackaged (XPolicyLab format: /models/robotwin-persistent/data/demo_clean/<task>/aloha_agilex/data/
episode_0000000.hdf5, vision/cam_head|cam_left_wrist|cam_right_wrist/colors = the same JPEG bytes as the original
observation/<camera>/rgb, without the last frame, which src/convert_robotwin_to_lerobot.py drops anyway).

Reproduces what convert_robotwin_to_lerobot.py + lerobot 0.4.2 did for every episode and camera: cv2.imdecode of
each JPEG (used as is), encoded on its own with libsvtav1 (g=2, crf=30, preset=12, yuv420p, 50 fps; lerobot's
encode_video_frames defaults), then appended to the camera's file with the concat demuxer (stream copy, as lerobot's
concatenate_video_files), following the chunk / file layout and episode order in meta/episodes. With the same PyAV
build the result is the original file byte for byte (checked on cluster); with another encoder build the frames differ
by the codec's rounding only, which --verify measures (PSNR against the JPEGs, frame count, every frame's timestamp
against meta/episodes from_timestamp + j / fps).

  python src/rebuild_videos.py --dataset_root <dir with <task>-aloha-agilex_clean_50-50/{meta,data}> \
      --raw_root /models/robotwin-persistent/data/demo_clean --raw_format amd [--tasks a,b] [--workers 32] [--verify]
"""

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

CAMS = {  # dataset video key -> raw dataset path (amd / cluster)
    "observation.images.cam_high": ("vision/cam_head/colors", "observation/head_camera/rgb"),
    "observation.images.cam_left_wrist": ("vision/cam_left_wrist/colors", "observation/left_camera/rgb"),
    "observation.images.cam_right_wrist": ("vision/cam_right_wrist/colors", "observation/right_camera/rgb"),
}
FPS = 50
OPTIONS = {"g": "2", "crf": "30", "preset": "12"}


def raw_episode(raw_root, task, i, fmt):
    if fmt == "amd":
        return Path(raw_root) / task / "aloha_agilex" / "data" / f"episode_{i:07d}.hdf5"
    return Path(raw_root) / task / "aloha-agilex_clean_50" / "data" / f"episode{i}.hdf5"


def frames_of(path, key, fmt, n):
    with h5py.File(path, "r") as f:
        jpgs = f[CAMS[key][0 if fmt == "amd" else 1]][()]
    assert len(jpgs) >= n, f"{path}: {len(jpgs)} frames < {n}"
    return [cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR) for b in jpgs[:n]]


def encode_episode(frames, path):
    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as out:
        st = out.add_stream("libsvtav1", FPS, options=dict(OPTIONS))
        st.pix_fmt, st.width, st.height = "yuv420p", w, h
        for fr in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


def concat(paths, out_path):
    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False) as lst:
        lst.write("ffconcat version 1.0\n" + "".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    tmp_out = str(out_path) + ".tmp.mp4"
    src = av.open(lst.name, mode="r", format="concat", options={"safe": "0"})
    dst = av.open(tmp_out, mode="w", options={"movflags": "faststart"})
    smap = {}
    for s in src.streams:
        if s.type in ("video", "audio", "subtitle"):
            smap[s.index] = dst.add_stream_from_template(template=s, opaque=True)
            smap[s.index].time_base = s.time_base
    for pkt in src.demux():
        if pkt.stream.index not in smap or pkt.dts is None:
            continue
        pkt.stream = smap[pkt.stream.index]
        dst.mux(pkt)
    src.close(); dst.close()
    os.replace(tmp_out, out_path)
    os.unlink(lst.name)


def episodes_meta(ds_dir):
    files = sorted((ds_dir / "meta" / "episodes").rglob("*.parquet"))
    rows = []
    for f in files:
        rows += pq.read_table(f).to_pylist()
    return sorted(rows, key=lambda r: r["episode_index"])


def video_groups(ds_dir, key):
    """(chunk, file) -> [episode rows in order] for one camera, plus the dataset's video path template."""
    info = json.loads((ds_dir / "meta" / "info.json").read_text())
    groups = {}
    for r in episodes_meta(ds_dir):
        groups.setdefault((r[f"videos/{key}/chunk_index"], r[f"videos/{key}/file_index"]), []).append(r)
    return info["video_path"], groups


def rebuild(job):
    ds_dir, raw_root, fmt, key = job
    ds_dir = Path(ds_dir)
    task = ds_dir.name.split("-aloha-agilex")[0]
    tmpl, groups = video_groups(ds_dir, key)
    work = Path(tempfile.mkdtemp(prefix=f"rv_{task}_{key.split('.')[-1]}_", dir=os.environ.get("TMPDIR")))
    try:
        for (chunk, fidx), eps in groups.items():
            parts = []
            for r in eps:
                p = work / f"ep{r['episode_index']:03d}.mp4"
                encode_episode(frames_of(raw_episode(raw_root, task, r["episode_index"], fmt), key, fmt, r["length"]), p)
                parts.append(p)
            out = ds_dir / tmpl.format(video_key=key, chunk_index=chunk, file_index=fidx)
            out.parent.mkdir(parents=True, exist_ok=True)
            if len(parts) == 1:
                shutil.move(str(parts[0]), out)
            else:
                concat(parts, out)
        return f"{ds_dir.name} {key}: {sum(len(e) for e in groups.values())} episodes"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def verify(job):
    """Decode every frame; compare with the JPEG (PSNR) and its timestamp with meta/episodes."""
    ds_dir, raw_root, fmt, key = job
    ds_dir = Path(ds_dir)
    task = ds_dir.name.split("-aloha-agilex")[0]
    tmpl, groups = video_groups(ds_dir, key)
    psnrs, worst_dt, n = [], 0.0, 0
    for (chunk, fidx), eps in groups.items():
        path = ds_dir / tmpl.format(video_key=key, chunk_index=chunk, file_index=fidx)
        with av.open(str(path)) as c:
            dec = ((fr.pts * fr.time_base, fr.to_ndarray(format="rgb24")) for fr in c.decode(video=0))
            for r in eps:
                ref = frames_of(raw_episode(raw_root, task, r["episode_index"], fmt), key, fmt, r["length"])
                t0 = r[f"videos/{key}/from_timestamp"]
                for j, img in enumerate(ref):
                    t, got = next(dec)
                    worst_dt = max(worst_dt, abs(float(t) - (t0 + j / FPS)))
                    mse = np.mean((got.astype(np.float64) - img.astype(np.float64)) ** 2)
                    psnrs.append(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse))
                    n += 1
            left = sum(1 for _ in dec)
    return (f"{ds_dir.name} {key}: {n} frames, extra frames {left}, worst timestamp error {worst_dt * 1e3:.3f} ms, "
            f"PSNR mean {np.mean(psnrs):.2f} min {np.min(psnrs):.2f} dB"), left == 0 and worst_dt < 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--raw_format", choices=("amd", "cluster"), required=True)
    ap.add_argument("--tasks", default="", help="comma-separated task names (default: every dataset dir)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify", action="store_true", help="only check the existing videos")
    args = ap.parse_args()
    dirs = sorted(d for d in Path(args.dataset_root).iterdir() if (d / "meta" / "info.json").exists())
    if args.tasks:
        want = set(args.tasks.split(","))
        dirs = [d for d in dirs if d.name.split("-aloha-agilex")[0] in want]
    jobs = [(str(d), args.raw_root, args.raw_format, k) for d in dirs for k in CAMS]
    print(f"{len(dirs)} datasets, {len(jobs)} camera files, {args.workers} workers, PyAV {av.__version__}", flush=True)
    ok = True
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(verify if args.verify else rebuild, jobs):
            if args.verify:
                res, good = res
                ok &= good
            print(res, flush=True)
    if args.verify:
        print("VERIFY OK" if ok else "VERIFY FAILED", flush=True)


if __name__ == "__main__":
    main()
