import sys, json, collections
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'code'))
import field_build as fb
fb.ALIASES.update({'mandarin': 'tangerine', 'cherries': 'cherry', 'bunch of grapes': 'grapes'})
D = json.load(open(sys.argv[1])); idx = json.load(open('index.json'))
names = sorted({d[0] for v in D.values() for d in v} | set(fb.COSMETIC))
NAMES = ['green_apple','peach','pomegranate','tangerine','watermelon','banana','red_apple','grapes','starfruit','red_delicious','strawberry','persimmon','lemon','orange','mangosteen','pineapple','mango','pear','passion_fruit','dragon_fruit','kiwi','avocado','cherry','wax_apple','papaya','tomato','mushroom','cucumber','pumpkin','bell_pepper','carrot','red_chili','green_chili','eggplant','onion','potato','plate_white','plate_yellow','plate_pink','plate_purple','plate_blue','lotion_bottle','lip_gloss','lipstick','eyeshadow_palette','brush_large','brush_small','highlighter','contour','powder_puff','glitter_compact','organizer_bottom','organizer_middle','organizer_top','dustpan','brush','basket','paper_ball_red','paper_ball_white']
rows = []
for r in idx:
    t = fb.task_of(r['instr']); o, d = fb.object_phrase(r['instr'], t); tgt = fb.match_class(o, NAMES, t)
    dest = f'plate_{d}' if t == 1 else f'organizer_{d}' if t == 2 else 'basket'
    dets = D[f"first/{r['ep']:04d}"]
    def top(c): 
        c = [x for x in dets if x[0] == c]; return max(x[5] for x in c) if c else 0
    rank = None
    if tgt:   # rank of the target among task objects by confidence (is it the detector's best guess anywhere?)
        rank = sum(1 for x in dets if x[5] > top(tgt) and x[0] == tgt) 
    rows.append(dict(ep=r['ep'], t=t, tgt=tgt, dest=dest, tconf=top(tgt) if tgt else None, dconf=top(dest), n25=sum(x[5] >= .25 for x in dets)))
for t in (1, 2, 3):
    R = [x for x in rows if x['t'] == t]
    npar = sum(x['tgt'] is not None for x in R)
    print(f'== task {t}: {len(R)} episodes, target parsed {npar}')
    for th in (0.05, 0.1, 0.25, 0.5):
        tt = sum((x['tconf'] or 0) >= th for x in R) if t != 3 else None
        nd = sum(x['dconf'] >= th for x in R); mb = sum(sum(d[5] >= th for d in D['first/%04d' % x['ep']]) for x in R) / len(R)
        print(f'  conf>={th}: target-class box {tt} | dest-class box {nd} | mean boxes/img {mb:.1f}')
    c = collections.Counter(d[0] for x in R for d in D[f"first/{x['ep']:04d}"] if d[5] >= .25)
    print('  classes predicted @0.25:', c.most_common(25))
json.dump(rows, open(sys.argv[1].replace('.json', '_rows.json'), 'w'))
