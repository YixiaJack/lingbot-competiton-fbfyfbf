"""Task-2 box cues (same cue as task 1): white box = target cosmetic, orange box = destination organizer level.

Instruction parsing uses the official task-2 templates and object phrases (field_training/prompt_policy.json); the
detector has one class per organizer level (organizer_top / _middle / _bottom), the destination box is the most confident
box of that class (>= conf). Target: best box of its class, conf >= 0.5 and 0.15 ahead of a second box of that class.
boxgrasp: grasp frame = first close of the gripper COMMAND after it opened (t1_boxes.grasp_frame); a demo gets a cue only
if it has one acting arm, the measured gripper holds something after that close and the arm lifts > 3 cm.
Checked 11/14 on the official 216: 213 boxable, target right 128/133 vs frame-difference truth, level boxes right on
every flagged case looked at; grasp rule holds on 215/216.
Usage: t2_boxes.py --data <v1> --dets dets_first.json --policy prompt_policy.json --out out.json [--instructions rewritten.json]
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_boxes as TB


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--dets', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--instructions', type=Path, help='rewritten instructions {episode: {instruction}} (our own data)')
    p.add_argument('--conf', type=float, default=0.5)
    a = p.parse_args()
    pol = json.loads(a.policy.read_text())['tasks']['2']
    OBJ2 = {v: k for k, v in pol['object_phrases'].items()}
    pats = [re.compile(re.escape(t['text']).replace(r'\{object\}', '(?P<o>.+?)').replace(r'\{destination\}', '(?P<d>top|middle|bottom)'), re.I)
            for t in pol['templates']]

    def parse2(s):
        for pt in pats:
            m = pt.fullmatch(s.strip())
            if m and m['o'].lower() in OBJ2:
                return OBJ2[m['o'].lower()], 'organizer_' + m['d'].lower()
        return None

    dets = json.loads(a.dets.read_text())
    ins = json.loads(a.instructions.read_text())['episodes'] if a.instructions else None
    lab = {}
    for line in open(a.data / 'meta/labels.jsonl'):
        d = json.loads(line); L = d['labels'][0]
        cf = L.get('custom_fields') or []
        lab[d['episode_index']] = (next((f['value'] for f in cf if f['name'] == 'id'), f"ep{d['episode_index']:04d}"), L['task_instruction'])
    if ins is not None:
        eps = sorted(int(k) for k, v in ins.items() if v['task'] == 2)
        text = {e: ins[str(e)]['instruction'] for e in eps}
    else:
        eps = sorted(e for e, (i, _) in lab.items() if i.startswith('T2_'))
        text = {e: lab[e][1] for e in eps}
    t = pq.read_table(sorted((a.data / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'action.effector.position',
                      'observation.state.effector.position', 'observation.state.ee.position'])
    ep = t.column('episode_index').to_numpy(); fi = t.column('frame_index').to_numpy()
    C = np.array(t.column('action.effector.position').to_pylist()); G = np.array(t.column('observation.state.effector.position').to_pylist())
    E = np.array(t.column('observation.state.ee.position').to_pylist())
    out, why = {}, {}
    for e in eps:
        pr = parse2(text[e])
        if pr is None:
            why['object has no detector class / not parsed'] = why.get('object has no detector class / not parsed', 0) + 1; continue
        m = ep == e; o = np.argsort(fi[m]); c, g, ee = C[m][o], G[m][o], E[m][o]
        arms = [x for x in (0, 1) if (c[:, x] > TB.CMD_OPEN).any()]; gf = TB.grasp_frame(c)
        if gf is None or len(arms) != 1:
            why['grasp rule does not fit'] = why.get('grasp rule does not fit', 0) + 1; continue
        x = arms[0]
        hold = float(np.median(g[gf + 15:gf + 30, x]) - np.median(g[:15, x])); lift = float(ee[gf:gf + 60, 7 * x + 2].max() - ee[gf, 7 * x + 2])
        if not (hold > 0.02 and lift > 0.03):
            why['grasp rule does not fit'] = why.get('grasp rule does not fit', 0) + 1; continue
        dd = dets.get(f'first/{e:04d}', [])
        tc = sorted([(r[5], r[1:5]) for r in dd if r[0] == pr[0]], key=lambda z: -z[0])
        lv = [(r[5], r[1:5]) for r in dd if r[0] == pr[1] and r[5] >= a.conf]
        reason = ('target not detected' if not tc or tc[0][0] < a.conf else
                  'target ambiguous' if len(tc) > 1 and tc[0][0] - tc[1][0] < 0.15 else
                  'level not detected' if not lv else 'ok')
        why[reason] = why.get(reason, 0) + 1
        if reason != 'ok':
            continue
        out[str(e)] = {'id': lab[e][0], 'instruction': text[e], 'target': pr[0], 'dest': pr[1],
                       'target_box': [round(v, 1) for v in tc[0][1]], 'dest_box': [round(v, 1) for v in max(lv, key=lambda z: z[0])[1]],
                       'grasp_frame': gf, 'T': int(m.sum())}
    summ = {'t2_episodes': len(eps), 'with_cue': len(out), 'reasons': why, 'mode': 'boxgrasp', 'line_px': 5}
    a.out.write_text(json.dumps({'summary': summ, 'episodes': out}, indent=1, ensure_ascii=False))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
