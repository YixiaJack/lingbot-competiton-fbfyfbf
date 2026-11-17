"""Rewrite our collected task-1/2 instructions into the official phrasing, with the official template distribution.

Templates and object phrases come from field_training/prompt_policy.json (built from the official 647 demos):
task 1 five plate templates (41/43/46/43/42 of 215), task 2 five organizer templates (48/42/43/40/43 of 216).
Each task's episodes get templates in proportion to those weights (largest remainder, seeded shuffle), so the
distribution matches the official one. The meaning (object, destination) is kept. Object names are mapped to the
official phrase where the official list has the object (aliases below; a bare "apple" is resolved from the frame-0
detector when exactly one apple class is seen); objects the official list does not have keep their own name.
Usage: private_instructions.py --prep private_prep.json --policy prompt_policy.json --dets dets_first_v3.json --out out.json
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path

T1_ALIASES = {'grapes': 'bunch of grapes', 'cherry': 'cherries', 'kiwi fruit': 'kiwi',
              'watermellon': 'watermelon'}                     # 11/14: typo in a collected instruction (priv ep114)
# 11/14: "dark-colored round highlighter compact" (priv ep51) is the two-tone dark compact = the official contour compact
# (checked on frames: the light highlighter stays on the table, the dark two-tone one goes into the organizer)
T2_ALIASES = {'dark-colored round highlighter compact': 'dark-colored round contour compact'}
APPLE_CLASSES = {'red_apple': 'red apple', 'green_apple': 'green apple', 'red_delicious': 'red delicious apple', 'wax_apple': 'wax apple'}
T1_IN = re.compile(r'(?:pick up|put|place|move|could you put) the (?P<o>.+?) (?:and place it )?(?:on|onto) the (?P<d>[a-z]+) plate[.?]?$', re.I)
T2_IN = re.compile(r'(?:put|place|store|organize|take) the (?P<o>.+?) (?:and put it )?(?:in|into|inside) the (?P<d>top|middle|bottom) (?:compartment|section|level)', re.I)


def allocate(n, weights, seed):
    tot = sum(weights); raw = [n * w / tot for w in weights]
    cnt = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - cnt[i], reverse=True)[:n - sum(cnt)]:
        cnt[i] += 1
    seq = [i for i, c in enumerate(cnt) for _ in range(c)]
    random.Random(seed).shuffle(seq)
    return seq, cnt


def main():
    p = argparse.ArgumentParser()
    for k in ('prep', 'policy', 'dets', 'out'):
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args()
    P = json.loads(a.prep.read_text())['episodes']; pol = json.loads(a.policy.read_text())['tasks']
    dets = json.loads(a.dets.read_text())
    t1_official = set(pol['1']['object_phrases'].values())
    out, notes = {}, Counter()
    for task, seed in ((1, 1), (2, 2)):
        tpls = pol[str(task)]['templates']
        eps = sorted((k for k, r in P.items() if r['task'] == task), key=int)
        seq, cnt = allocate(len(eps), [t['weight'] for t in tpls], seed)
        for k, ti in zip(eps, seq):
            src = P[k]['instruction'].strip()
            m = (T1_IN if task == 1 else T2_IN).match(src)
            if not m:
                out[k] = {'task': task, 'original': src, 'instruction': src, 'note': 'not parsed, kept'}; notes['not parsed'] += 1; continue
            o, d = m['o'].lower(), m['d'].lower()
            note = ''
            if task == 1:
                if o in T1_ALIASES:
                    o, note = T1_ALIASES[o], f'object alias -> {T1_ALIASES[m["o"].lower()]}'
                elif o == 'apple':
                    seen = {c for c, *_, s in dets.get(f'first/{int(k):04d}', []) if c in APPLE_CLASSES and s >= 0.5}
                    if len(seen) == 1:
                        o = APPLE_CLASSES[seen.pop()]; note = f'apple resolved from detector -> {o}'
                    else:
                        note = f'apple ambiguous ({sorted(seen)}), kept'
                if o not in t1_official:
                    note = (note + '; ' if note else '') + 'object not in the official list, own name kept'
            else:
                d = d.lower()
                if o in T2_ALIASES:
                    o, note = T2_ALIASES[o], f'object alias -> {T2_ALIASES[o]}'
                if o not in set(pol['2']['object_phrases'].values()):
                    note = 'object not in the official list, own name kept'
            new = tpls[ti]['text'].format(object=o, destination=d)
            out[k] = {'task': task, 'original': src, 'instruction': new, 'template': tpls[ti]['id'], 'note': note}
            if note: notes[note] += 1
        print(f'task {task}: {len(eps)} episodes, templates {dict(zip([t["id"] for t in tpls], cnt))} '
              f'(official {[t["weight"] for t in tpls]})')
    a.out.write_text(json.dumps({'episodes': out, 'notes': dict(notes)}, indent=1, ensure_ascii=False))
    for n, c in notes.most_common(): print(f'  {c} x {n}')


if __name__ == '__main__':
    main()
