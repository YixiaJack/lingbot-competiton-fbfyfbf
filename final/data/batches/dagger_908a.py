"""DAgger recordings in b60f1e2e 908aa6e5 (task 2: 439-446, 454): human takeover segments by dagger_seg.segments (jerk + gripper-unit, validated on fb04
333-339 and on the pure-policy 105a2d09 172-181), check strips for a by-eye pass, prompts rewritten to the official template mix.
Writes out/dagger_908a.json and out/chk_dagger_908a_N.jpg (policy, START, mid, END-45, END)."""
import json, sys, random
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
import dagger_seg as DS, yolocue_bridge as YC
V1 = Path.home() / 'Downloads/b60f1e2e_908aa6e5/v1'; CAM = 'observation.images.camera_top'
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
lab = {}
for l in open(V1 / 'meta/labels.jsonl'):
    r = json.loads(l); L = (r.get('labels') or [{}])[0]; lab[r['episode_index']] = (L.get('task_instruction') or '', L.get('result'))
cols = ['episode_index', 'frame_index', 'action.effector.position', 'action.arm.position', 'observation.state.arm.position', 'observation.state.ee.position']
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=cols); ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
C, A, S, E = (np.array(t.column(c).to_pylist()) for c in cols[2:])
def frames(e, idx):
    m = meta[e]; p = V1 / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"; t0 = m[f'videos/{CAM}/from_timestamp']
    want = sorted(set(idx)); ts = [t0 + i / 30 for i in want]; got = {}; k = 0
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in ic.decode(s):
            while k < len(want) and float(fr.pts * s.time_base) + 1e-3 >= ts[k]: got[want[k]] = fr.to_image(); k += 1
            if k == len(want): break
    return got
pol = json.load(open(Path.home() / '.cache/field_dev/watch/repo/prompt_policy.json'))['tasks']
out = {}; cells = []; F = ImageFont.load_default(size=14)
for e in [439, 440, 441, 442, 443, 444, 445, 446, 454]:
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; segs, rm, gp = DS.segments(C[m], A[m], S[m])
    if not segs: continue
    for si, (st, en) in enumerate(segs):
        ee = E[m]; v_end = float(np.abs(np.diff(ee[en - 15:en, [0, 1, 2, 7, 8, 9]], axis=0)).max())
        c = C[m][st:en]; out[f'{e}_{si}'] = {'ep': e, 'start': st, 'end': en, 'T': len(m), 'instruction': lab[e][0], 'result': lab[e][1], 'human_s': round((en - st) / 30, 1),
                                             'policy_s_before': round(st / 30, 1), 'ee_step_at_end': round(v_end, 4), 'grip_range': [round(float(c.min()), 2), round(float(c.max()), 2)]}
        idx = [max(st - 60, 0), st, (st + en) // 2, max(en - 45, st), en - 1]; fr = frames(e, idx)
        cell = Image.new('RGB', (5 * 384, 310), 'white')
        for k, i in enumerate(idx): cell.paste(fr[i].resize((384, 288)), (k * 384, 22))
        ImageDraw.Draw(cell).text((4, 3), f"ep{e} ({lab[e][1]}) {lab[e][0][:70]} | policy f{idx[0]}, START f{st}, mid, end-45, END f{en - 1}/{len(m)} | human {out[f'{e}_{si}']['human_s']}s", font=F, fill='black'); cells.append(cell)
        print(f'{e}_{si}', json.dumps(out[f'{e}_{si}'], ensure_ascii=False))
for k in range(0, len(cells), 5):
    cc = cells[k:k + 5]; sh = Image.new('RGB', (1920, 310 * len(cc)), 'white')
    for i, c_ in enumerate(cc): sh.paste(c_, (0, 310 * i))
    sh.save(f'out/chk_dagger_908a_{k // 5 + 1}.jpg', quality=80)
json.dump(out, open('out/dagger_908a.json', 'w'), indent=1, ensure_ascii=False); print(len(out), 'segments')
