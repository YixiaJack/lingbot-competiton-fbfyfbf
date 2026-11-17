"""DAgger recordings in b60f1e2e fb049f09 (eps 333-339): the platform recorded whole inference sessions (policy ~10 s, human takeover,
maybe policy again after resume). Only the human takeover segment may be trained on (docs/08 section 4).
  policy regime  = gripper command outside the teleop range (< -0.02 or > 1.5 on either arm) -- the policy writes gripper in other units
  onset          = last policy-regime frame before the long teleop stretch
  settle         = first frame after onset where |action.arm - state.arm| < 0.15 for 15 frames (master/follower aligned)
  start          = settle + 15 frames (drop 0.5 s of transition)
  end            = first policy-regime frame after start - 15 (post-resume), else the episode end
Writes out/dagger_fb04.json and a check strip per episode out/chk_dagger_fb04.jpg (policy, onset, start, mid, end)."""
import json, sys
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
V1 = Path.home() / 'Downloads/b60f1e2e_fb049f09/v1'; CAM = 'observation.images.camera_top'
meta = {m['episode_index']: m for m in pq.read_table(sorted((V1 / 'meta/episodes').glob('*/*.parquet'))[0]).to_pylist()}
lab = {}
for l in open(V1 / 'meta/labels.jsonl'):
    r = json.loads(l); L = (r.get('labels') or [{}])[0]; lab[r['episode_index']] = (L.get('task_instruction') or '', L.get('result'))
cols = ['episode_index', 'frame_index', 'action.effector.position', 'action.arm.position', 'observation.state.arm.position', 'observation.state.ee.position']
t = pq.read_table(sorted((V1 / 'data').glob('*/*.parquet'))[0], columns=cols)
ep, fi = t.column('episode_index').to_numpy(), t.column('frame_index').to_numpy()
C = np.array(t.column(cols[2]).to_pylist()); A = np.array(t.column(cols[3]).to_pylist()); S = np.array(t.column(cols[4]).to_pylist()); E = np.array(t.column(cols[5]).to_pylist())
def frames(e, idx):
    m = meta[e]; p = V1 / 'videos' / CAM / f"chunk-{m[f'videos/{CAM}/chunk_index']:03d}" / f"file-{m[f'videos/{CAM}/file_index']:03d}.mp4"; t0 = m[f'videos/{CAM}/from_timestamp']
    want = sorted(set(idx)); ts = [t0 + i / 30 for i in want]; got = {}; k = 0
    with av.open(str(p)) as ic:
        s = ic.streams.video[0]; ic.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in ic.decode(s):
            while k < len(want) and float(fr.pts * s.time_base) + 1e-3 >= ts[k]: got[want[k]] = fr.to_image(); k += 1
            if k == len(want): break
    return got
out = {}; cells = []; F = ImageFont.load_default(size=14)
for e in sorted(x for x in meta if x >= 325):
    m = np.where(ep == e)[0]; m = m[np.argsort(fi[m])]; c, a, s, ee = C[m], A[m], S[m], E[m]; T = len(m)
    pol = (c < -0.02).any(1) | (c > 1.5).any(1)
    if not pol.any(): continue
    onset = int(np.where(pol[: int(T * 0.8)])[0].max())
    gap = np.abs(a - s).max(1); ok = gap < 0.15; settle = None
    for j in range(onset + 1, T - 15):
        if ok[j:j + 15].all(): settle = j; break
    start = (settle or onset + 30) + 15
    later = np.where(pol[start:])[0]; end = start + int(later.min()) - 15 if len(later) else T
    v_end = float(np.abs(np.diff(ee[max(end - 15, start):end, [0, 1, 2, 7, 8, 9]], axis=0)).max()) if end - start > 20 else None
    out[e] = {'instruction': lab[e][0], 'result': lab[e][1], 'T': T, 'policy_frames': int(pol.sum()), 'onset': onset, 'settle': settle, 'start': start, 'end': end,
              'secs': {'policy': round(onset / 30, 1), 'human': round((end - start) / 30, 1)}, 'ee_step_at_end': v_end}
    print(e, json.dumps(out[e], ensure_ascii=False))
    idx = [max(onset - 60, 0), onset, start, (start + end) // 2, max(end - 45, start), end - 1]; fr = frames(e, idx)
    cell = Image.new('RGB', (6 * 320, 262), 'white')
    for k, i in enumerate(idx): cell.paste(fr[i].resize((320, 240)), (k * 320, 22))
    ImageDraw.Draw(cell).text((4, 3), f"ep{e} ({lab[e][1]}) {lab[e][0][:60]} | policy f{idx[0]}, onset f{onset}, START f{start}, mid, end-45, END f{end - 1} of {T}", font=F, fill='black'); cells.append(cell)
sh = Image.new('RGB', (1920, 262 * len(cells)), 'white')
for i, c_ in enumerate(cells): sh.paste(c_, (0, 262 * i))
sh.save('out/chk_dagger_fb04.jpg', quality=82); json.dump({str(k): v for k, v in out.items()}, open('out/dagger_fb04.json', 'w'), indent=1, ensure_ascii=False)
