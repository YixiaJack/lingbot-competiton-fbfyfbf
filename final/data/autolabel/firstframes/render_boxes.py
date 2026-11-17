"""Frame 0 of every box-cue training episode, exactly as the model saw it (boxes are burned into the re-encoded head
camera), with the cue facts written above it. Run on the dev machine.
  OUT/<dataset>/<ep>.jpg, OUT/sheets/<dataset>_NN.jpg (4x5 thumbnails), OUT/index.json (source_map rows + files)
usage: python3 render_boxes.py OUT [DATA_DIR=/data/lingbot/data]
"""
import json
import sys
from pathlib import Path

import av
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

CAM = 'observation.images.camera_top'
HEAD = 58


def frame0(path, t0):
    with av.open(str(path)) as ic:
        s = ic.streams.video[0]
        ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in ic.decode(s):
            if float(fr.pts * s.time_base) + 1e-3 >= t0:
                return fr.to_image()


def main():
    out = Path(sys.argv[1]); data = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/data/lingbot/data')
    font = ImageFont.load_default()
    rows = []
    for ds in sorted(p for p in data.iterdir() if 'box' in p.name and (p / 'meta/source_map.json').exists()):
        smap = {r['episode_index']: r for r in json.loads((ds / 'meta/source_map.json').read_text())}
        meta = pq.read_table(sorted((ds / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()
        (out / ds.name).mkdir(parents=True, exist_ok=True)
        thumbs = []
        for m in meta:
            e = m['episode_index']; r = smap.get(e, {})
            vid = ds / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"
            try:
                im = frame0(vid, m[f'videos/{CAM}/from_timestamp'])
            except Exception as ex:
                print('FAIL', ds.name, e, repr(ex), flush=True); continue
            can = Image.new('RGB', (im.width, im.height + HEAD), (255, 255, 255)); can.paste(im, (0, HEAD))
            d = ImageDraw.Draw(can)
            d.text((6, 4), f"{ds.name}  ep{e}  src {r.get('id')} (#{r.get('src_episode')})  grasp_frame {r.get('grasp_frame')}", font=font, fill=(0, 0, 0))
            d.text((6, 22), f"WHITE target = {r.get('target')} {r.get('target_box')}", font=font, fill=(90, 90, 90))
            d.text((330, 22), f"ORANGE dest = {r.get('dest')} {r.get('dest_box')}", font=font, fill=(200, 100, 0))
            d.text((6, 40), (r.get('task') or '')[:100], font=font, fill=(0, 0, 160))
            f = out / ds.name / f'{e:04d}.jpg'; can.save(f, quality=82)
            thumbs.append(can.resize((320, (im.height + HEAD) // 2)))
            rows.append({**r, 'dataset': ds.name, 'file': f'{ds.name}/{e:04d}.jpg'})
        (out / 'sheets').mkdir(exist_ok=True)
        for k in range(0, len(thumbs), 20):
            th = thumbs[k:k + 20]; w, h = th[0].size
            sheet = Image.new('RGB', (4 * w, ((len(th) + 3) // 4) * h), (255, 255, 255))
            for i, t in enumerate(th):
                sheet.paste(t, ((i % 4) * w, (i // 4) * h))
            sheet.save(out / 'sheets' / f'{ds.name}_{k // 20 + 1:02d}.jpg', quality=80)
        print(ds.name, len(thumbs), flush=True)
    (out / 'index.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print('DONE', len(rows))


if __name__ == '__main__':
    main()
