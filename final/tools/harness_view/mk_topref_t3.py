"""Task-3 references for the board's /topref?t=3 page. Run on the Mac with ~/lingbot_v7env/bin/python:
    ~/lingbot_v7env/bin/python mk_topref_t3.py 20261115 204000 [N=10]      (date, from HHMMSS, number of sessions)
Task 3 has no yolocue image, so the frame comes from the platform recorder on the robot host: the first N robotharness
sessions from that time (5080 logs, read-only) give the session start time; agx_frames.py (robot host, read-only, nice 19,
cores 18-23) returns the head-camera JPEG at that moment from /mnt/ssd/bags/record (no box was ever drawn on it).
YOLO knows none of the task-3 objects, so the object ghost is a colour mask: everything that is neither the teal table
nor the green backdrop (dustpan, brush, basket, paper balls; also an arm if it is in view).
Writes topref/{ref,obj,edge,line}_<id>.{jpg,png}, merges task_no 3 entries into topref/refs.json, check_t3_<date>_<from>.jpg."""
import base64, json, os, subprocess, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'topref')
LOGS = 'workspace/lingbot/inference/logs'


def sessions(date, t0, n):
    ls = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'lbreal', f'ls {LOGS}'], capture_output=True, text=True, timeout=60)
    names = sorted(x for x in ls.stdout.split() if x.startswith(f'robotharness_{date}_') and x.endswith('.jsonl') and x.split('_')[2] >= t0)[:n]
    out = []
    for nm in names:
        txt = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lbreal', f'head -c 20000 {LOGS}/{nm}'], capture_output=True, text=True, timeout=60).stdout
        recs = []
        for l in txt.splitlines():
            try:
                recs.append(json.loads(l))
            except ValueError:
                pass
        st = next((r for r in recs if r.get('event') == 'start'), None)
        if not st:
            continue
        obs = ''
        for r in recs:
            if r.get('event') == 'llm':
                try:
                    obs = json.loads(r.get('reply') or '{}').get('observation', '')
                except ValueError:
                    obs = ''
                break
        out.append({'stem': nm[len('robotharness_'):-len('.jsonl')], 'start': float(st['wall_time']), 'task': st.get('task'),
                    'session': st.get('session_id'), 'observation': obs})
    return out


def frames(req):
    arg = base64.b64encode(json.dumps([[s['stem'], s['start']] for s in req]).encode()).decode()
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'agx',
                        f'source /opt/ros/humble/setup.bash >/dev/null 2>&1; nice -n 19 ionice -c3 taskset -c 18-23 python3 - {arg}'],
                       stdin=open(os.path.join(HERE, 'agx_frames.py')), capture_output=True, text=True, timeout=600)
    got = {}
    for l in r.stdout.splitlines():
        q = l.split('\t')
        if len(q) == 3:
            got[q[0]] = None if q[1] == 'MISSING' else (float(q[1]), base64.b64decode(q[2]))
    if r.returncode != 0:
        print('agx stderr:', r.stderr[-400:])
    return got


def object_mask(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    bg = (h >= 35) & (h <= 105) & (s >= 45) & (v >= 40)          # teal table (h ~85-100) and green backdrop (h ~40-75)
    fg = (~bg).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    keep = np.zeros_like(fg)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= 60:
            keep[lab == k] = 255
    ff = keep.copy(); mk = np.zeros((keep.shape[0] + 2, keep.shape[1] + 2), np.uint8); cv2.floodFill(ff, mk, (0, 0), 255)
    return keep | cv2.bitwise_not(ff)


def label_parts(img, m):
    """name each colour-mask component by colour and shape: basket (white), dustpan (largest yellow), brush (long yellow),
    paper_ball (small red / yellow); anything else 'object'"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); n, lab, st, cen = cv2.connectedComponentsWithStats(m); parts = []
    for k in range(1, n):
        a = int(st[k, cv2.CC_STAT_AREA])
        if a < 150:
            continue
        sel = lab == k; hh, ss, vv = (float(np.median(hsv[..., i][sel])) for i in range(3))
        cs, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), (w, h), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea)); asp = max(w, h) / max(min(w, h), 1)
        if ss < 110 and vv > 120 and a > 5000:
            name = 'basket'                                  # light beige mesh, S ~75
        elif a < 4000 and asp < 1.8:
            name = 'paper_ball'                              # 1.6k-2.9k px, round (red hue wraps around 0/180: size only)
        elif a >= 12000:
            name = 'dustpan'                                 # ~20k px
        elif asp >= 1.8:
            name = 'brush'                                   # 6.5k-8k px, long
        else:
            name = 'object'
        parts.append({'name': name, 'cx': round(float(cen[k][0]), 1), 'cy': round(float(cen[k][1]), 1), 'area': a, 'ghost': True,
                      'box': [int(st[k, 0]), int(st[k, 1]), int(st[k, 0] + st[k, 2]), int(st[k, 1] + st[k, 3])]})
    return parts


def main():
    date, t0 = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.makedirs(OUT, exist_ok=True)
    ss = sessions(date, t0, n); print(len(ss), 'task-3 sessions from', t0)
    got = frames(ss)
    jf = os.path.join(OUT, 'refs.json')
    refs = {r['id']: r for r in (json.load(open(jf)) if os.path.exists(jf) else [])}
    sheet = []
    for s in ss:
        g = got.get(s['stem'])
        if not g:
            print('MISSING frame', s['stem']); continue
        ts, jpg = g; img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR); H, W = img.shape[:2]
        rid = s['stem']; m = object_mask(img); parts = label_parts(img, m); soft = cv2.GaussianBlur(m, (0, 0), 1.0)
        edge = np.zeros((H, W, 4), np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(edge, cs, -1, (0, 0, 0, 200), 4); cv2.drawContours(edge, cs, -1, (0, 230, 255, 255), 2)
        g2 = cv2.dilate(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.4), 30, 90), np.ones((2, 2), np.uint8))
        line = np.zeros((H, W, 4), np.uint8); line[g2 > 0] = (255, 80, 255, 255)
        with open(os.path.join(OUT, f'ref_{rid}.jpg'), 'wb') as f:
            f.write(jpg)                                              # the recorded frame as is
        cv2.imwrite(os.path.join(OUT, f'obj_{rid}.png'), np.dstack([img * (soft[..., None] > 0), soft]))
        cv2.imwrite(os.path.join(OUT, f'edge_{rid}.png'), edge)
        cv2.imwrite(os.path.join(OUT, f'line_{rid}.png'), line)
        tm = f'{rid[9:11]}:{rid[11:13]}:{rid[13:15]}'
        refs[rid] = {'id': rid, 'time': tm, 'date': rid[:8], 'session': s['session'], 'task': s['task'], 'task_no': 3,
                     'target': None, 'dest': None, 'objects': parts, 'observation': s['observation'],
                     'frame_offset_s': round(ts - s['start'], 3), 'source': 'robot host recorder /mnt/ssd/bags/record (head camera)'}
        print(rid, tm, f'frame +{ts - s["start"]:.2f}s', '|', ', '.join(p['name'] for p in parts))
        bgc = np.full((H, W, 3), 60, np.uint8); bgc[(np.indices((H, W)).sum(0) // 16) % 2 == 0] = 90
        a = soft[..., None] / 255.0; ghost = (img * a + bgc * (1 - a)).astype(np.uint8)
        e = edge[..., 3:] / 255.0; ghost = (ghost * (1 - e) + edge[..., :3] * e).astype(np.uint8)
        la = line[..., 3:] / 255.0; lines = (img * 0.45 * (1 - la) + line[..., :3] * la).astype(np.uint8)
        row = np.hstack([img, ghost, lines])
        cv2.putText(row, f'{tm} +{ts - s["start"]:.2f}s', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(row, f'{tm} +{ts - s["start"]:.2f}s', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        sheet.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA))
    out = sorted(refs.values(), key=lambda r: (r['date'], r['time']))
    with open(jf + '.part', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(jf + '.part', jf)
    if sheet:
        cv2.imwrite(os.path.join(OUT, f'check_t3_{date}_{t0}.jpg'), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(sum(1 for r in out if r.get('task_no') == 3), 'task-3 references in', jf)


if __name__ == '__main__':
    main()
