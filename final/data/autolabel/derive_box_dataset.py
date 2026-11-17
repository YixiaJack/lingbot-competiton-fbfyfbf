"""Task-1 box-cue training dataset (LeRobot v3) from the official demos: only the head camera is re-encoded.

For every demo in t1_boxes.json (YOLO v1 frame-0 boxes that passed the cue rule), the whole episode is copied with its
original instruction; camera_top gets the boxgrasp cue drawn on every frame (white target box until grasp_frame, orange
destination box always, LINE px wide so it survives the resize to the model's input size); the two wrist cameras point
at the original video files. Data table / tasks / stats come from derive_dataset.build (same frames, same actions).
Train it together with the plain dataset that holds the same demos (public_t12): the 50% box / 50% plain mix of the
real_detector experiment.
Usage: derive_box_dataset.py --src <public v1> --videos <local copy of v1/videos> --boxes t1_boxes.json --out OUT
"""
import argparse, json, os, shutil, sys
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import av
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sweep_harness/code'))
import derive_dataset as DD

CAM = 'observation.images.camera_top'
WHITE, ORANGE, LINE = (255, 255, 255), (255, 128, 0), 5


def draw(rgb, tb, db, grasped):
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
    for box, col, skip in ((tb, WHITE, grasped), (db, ORANGE, False)):
        if skip:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        d.rectangle([x1, y1, x2 - 1, y2 - 1], outline=col, width=LINE)
    return np.asarray(im)


def encode_group(args):
    k, jobs, videos, out_dir, fps, crf = args
    path = Path(out_dir) / f'file-{k:03d}.mp4'
    oc = av.open(str(path), mode='w')
    # options must go to add_stream (assigning st.options afterwards is silently ignored by PyAV 15)
    st = oc.add_stream('libx264', rate=fps, options={'crf': str(crf), 'preset': 'medium', 'g': '80'})   # originals: h264 High, keyframe every 80
    st.width, st.height, st.pix_fmt = 640, 480, 'yuv420p'
    st.time_base = Fraction(1, fps)
    placed, n = [], 0
    for j in jobs:
        src = Path(videos) / CAM / f"chunk-{j['chunk']:03d}" / f"file-{j['file']:03d}.mp4"
        t0, T = j['from_ts'], j['T']
        got = 0
        with av.open(str(src)) as ic:
            s = ic.streams.video[0]
            ic.seek(int(max(t0 - 0.1, 0) / s.time_base), stream=s, backward=True)
            for fr in ic.decode(s):
                ft = float(fr.pts * s.time_base)
                if ft + 1e-3 < t0 + got / fps:
                    continue
                rgb = draw(fr.to_ndarray(format='rgb24'), j['target_box'], j['dest_box'], got >= j['grasp_frame'])
                vf = av.VideoFrame.from_ndarray(rgb, format='rgb24'); vf.pts = n + got
                if got == 0:
                    vf.pict_type = av.video.frame.PictureType.I   # every episode starts on a keyframe
                for pkt in st.encode(vf):
                    oc.mux(pkt)
                got += 1
                if got == T:
                    break
        assert got == T, (j['ep'], got, T)
        placed.append((j['new_ep'], k, n / fps, (n + T) / fps)); n += T
    for pkt in st.encode():
        oc.mux(pkt)
    oc.close()
    return placed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, required=True); p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--boxes', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--crf', type=int, default=17, help='libx264 crf of the re-encoded head camera')
    p.add_argument('--instructions', type=Path, help='{episode: {"instruction": ...}} to use instead of the dataset task')
    a = p.parse_args()
    info, eps, ids = DD.load_src(a.src); fps = info['fps']
    B = json.loads(a.boxes.read_text())['episodes']
    sel = sorted(int(e) for e in B)
    ins = json.loads(a.instructions.read_text())['episodes'] if a.instructions else {}
    items = [{'src_ep': e, 'start': 0, 'end': eps[e]['length'], 'task': ins.get(str(e), {}).get('instruction', eps[e]['tasks'][0])} for e in sel]
    print(json.dumps(DD.build(a.src, a.out, items, fps)), flush=True)
    # replace the videos symlink: wrist cameras -> original files, head camera -> re-encoded with cues
    v = a.out / 'videos'
    if v.is_symlink() or v.is_file():
        v.unlink()
    elif v.exists():
        shutil.rmtree(v)
    v.mkdir()
    for key in [k for k, f in info['features'].items() if f.get('dtype') == 'video' and k != CAM]:
        os.symlink(a.videos.resolve() / key, v / key)
    cam_dir = v / CAM / 'chunk-000'; cam_dir.mkdir(parents=True)
    jobs = []
    for new_ep, e in enumerate(sel):
        r = B[str(e)]; m = eps[e]
        assert r['T'] == m['length'], (e, r['T'], m['length'])
        jobs.append({'ep': e, 'new_ep': new_ep, 'chunk': m[f'videos/{CAM}/chunk_index'], 'file': m[f'videos/{CAM}/file_index'],
                     'from_ts': m[f'videos/{CAM}/from_timestamp'], 'T': m['length'], 'target_box': r['target_box'],
                     'dest_box': r['dest_box'], 'grasp_frame': r['grasp_frame'] if r['grasp_frame'] is not None else 10 ** 9})
    groups = [(k, jobs[k::a.workers], str(a.videos), str(cam_dir), fps, a.crf) for k in range(min(a.workers, len(jobs)))]
    with Pool(len(groups)) as pool:
        placed = [x for g in pool.map(encode_group, groups) for x in g]
    where = {ne: (k, f, t) for ne, k, f, t in placed}
    epf = sorted((a.out / 'meta/episodes').glob('*/*.parquet'))[0]
    rows = pq.read_table(epf).to_pylist()
    for r in rows:
        k, f, t = where[r['episode_index']]
        r[f'videos/{CAM}/chunk_index'], r[f'videos/{CAM}/file_index'] = 0, k
        r[f'videos/{CAM}/from_timestamp'], r[f'videos/{CAM}/to_timestamp'] = f, t
    pq.write_table(pa.Table.from_pylist(rows), epf)
    smap = json.loads((a.out / 'meta/source_map.json').read_text())
    for s in smap:
        r = B[str(s['src_episode'])]
        s.update(cue='boxgrasp', target=r['target'], dest=r['dest'], target_box=r['target_box'], dest_box=r['dest_box'],
                 grasp_frame=r['grasp_frame'], line_px=LINE)
    (a.out / 'meta/source_map.json').write_text(json.dumps(smap, ensure_ascii=False))
    print(json.dumps({'box_episodes': len(rows), 'video_files': len(groups)}))


if __name__ == '__main__':
    main()
