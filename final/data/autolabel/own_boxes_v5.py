"""Box cues for our own task-1/2 demos with YOLO v5/v6 (labels from our own scenes; classes for our own objects too; --tag).

Same cue and rules as t1_boxes.py / t2_boxes.py, two differences for our data:
- the instruction parsed is the REWRITTEN one (official phrasing, the prompt the plain dataset trains with), so the box
  and plain copies of a demo share one prompt;
- object -> class covers v5's classes for our own objects (starfruit, green_chili, glitter_compact, brush_small).
Target: best box of its class, conf >= 0.5 and 0.15 ahead of a second box of that class. Destination: task 1 the plate
of the named colour (t1_boxes.select), task 2 the most confident box of the named organizer level (conf >= 0.5).
boxgrasp: grasp frame = first close of the gripper COMMAND after it opened; a demo gets a cue only if it has one acting
arm, the measured gripper holds something 0.5-1 s after that close and the arm lifts > 3 cm (task 1: private_prep's
grasp_rule_ok; task 2: computed here as in t2_boxes.py).
Usage: own_boxes_v5.py --data V1 --prep private_prep.json --instructions instr.json --dets dets.json --policy prompt_policy.json --out DIR
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_boxes as TB

OWN_T1 = {'starfruit': 'starfruit', 'green chili pepper': 'green_chili'}
OWN_T2 = {'black round glitter compact': 'glitter_compact', 'small black and gold makeup brush': 'brush_small'}


def main():
    p = argparse.ArgumentParser()
    for k in ('data', 'prep', 'instructions', 'dets', 'policy', 'out'):
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--tag', default='v5', help='detector version, used in the output file names (own_t1_boxes_<tag>.json)')
    a = p.parse_args()
    pol = json.loads(a.policy.read_text())['tasks']
    P = json.loads(a.prep.read_text())['episodes']; I = json.loads(a.instructions.read_text())['episodes']
    D = json.loads(a.dets.read_text())
    cls = {1: {v.lower(): k for k, v in pol['1']['object_phrases'].items()} | OWN_T1,
           2: {v.lower(): k for k, v in pol['2']['object_phrases'].items()} | OWN_T2}
    pats = {t: [re.compile(re.escape(x['text']).replace(r'\{object\}', '(?P<o>.+?)').replace(r'\{destination\}', '(?P<d>[a-z]+)'), re.I)
                for x in pol[str(t)]['templates']] for t in (1, 2)}

    def parse(s, t):
        for pt in pats[t]:
            m = pt.fullmatch(s.strip())
            if m and m['o'].lower() in cls[t]:
                return cls[t][m['o'].lower()], ('plate_' if t == 1 else 'organizer_') + m['d'].lower()
        return None

    t = pq.read_table(sorted((a.data / 'data').glob('*/*.parquet'))[0], columns=['episode_index', 'frame_index', 'action.effector.position',
                      'observation.state.effector.position', 'observation.state.ee.position'])
    ep = t.column('episode_index').to_numpy(); fi = t.column('frame_index').to_numpy()
    C = np.array(t.column('action.effector.position').to_pylist()); G = np.array(t.column('observation.state.effector.position').to_pylist())
    E = np.array(t.column('observation.state.ee.position').to_pylist())
    out = {1: {}, 2: {}}; why = {1: Counter(), 2: Counter()}
    for k in sorted(I, key=int):
        task = I[k]['task']
        if task not in (1, 2): continue
        instr = I[k]['instruction']; pr = parse(instr, task)
        if pr is None:
            why[task]['object has no YOLO class / not parsed'] += 1; continue
        e = int(k); m = ep == e; o = np.argsort(fi[m]); c, g, ee = C[m][o], G[m][o], E[m][o]
        if task == 1:
            ok, gf = bool(P[k].get('grasp_rule_ok')), P[k].get('grasp_frame')
        else:
            arms = [x for x in (0, 1) if (c[:, x] > TB.CMD_OPEN).any()]; gf = TB.grasp_frame(c); ok = False
            if gf is not None and len(arms) == 1:
                x = arms[0]
                hold = float(np.median(g[gf + 15:gf + 30, x]) - np.median(g[:15, x])); lift = float(ee[gf:gf + 60, 7 * x + 2].max() - ee[gf, 7 * x + 2])
                ok = hold > 0.02 and lift > 0.03
        if not ok or gf is None:
            why[task]['grasp rule does not fit'] += 1; continue
        dets = D.get(f'first/{e:04d}', [])
        if task == 1:
            tb, db, reason = TB.select(dets, *pr, conf_target=a.conf, conf_dest=a.conf)
        else:
            tc = sorted([(r[5], r[1:5]) for r in dets if r[0] == pr[0]], key=lambda z: -z[0])
            lv = [(r[5], r[1:5]) for r in dets if r[0] == pr[1] and r[5] >= a.conf]
            reason = ('target not detected' if not tc or tc[0][0] < a.conf else
                      'target ambiguous' if len(tc) > 1 and tc[0][0] - tc[1][0] < 0.15 else 'level not detected' if not lv else 'ok')
            tb = [round(v, 1) for v in tc[0][1]] if reason == 'ok' else None
            db = [round(v, 1) for v in max(lv, key=lambda z: z[0])[1]] if reason == 'ok' else None
        why[task][reason] += 1
        if reason == 'ok':
            out[task][k] = {'id': f'own_ep{e:04d}', 'instruction': instr, 'target': pr[0], 'dest': pr[1], 'target_box': tb, 'dest_box': db,
                            'grasp_frame': int(gf), 'T': int(m.sum())}
    a.out.mkdir(parents=True, exist_ok=True)
    for task in (1, 2):
        n = sum(1 for k in I if I[k]['task'] == task)
        summ = {'episodes': n, 'with_cue': len(out[task]), 'reasons': dict(why[task]), 'yolo': a.tag, 'mode': 'boxgrasp', 'line_px': 5}
        (a.out / f'own_t{task}_boxes_{a.tag}.json').write_text(json.dumps({'summary': summ, 'episodes': out[task]}, indent=1, ensure_ascii=False))
        print(f'task {task}:', json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
