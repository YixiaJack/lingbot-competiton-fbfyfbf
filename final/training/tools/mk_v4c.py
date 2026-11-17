"""lists/v4c.txt (11/15): user rule = newest data much higher weight; task ratio and task-3 subtask mix set by us.
Targets: task 1 : 2 : 3 = 28 : 24 : 48 of samples; inside task 3: grasp 18 / sweep 34 / dump 34 / home 14 (the 11/14 eval
failed in sweep-finishing and dump, grasp was 10/10; 'home' is long and easy and took 29% under frame-uniform sampling).
Recency weight per episode: public 1, older own 2-3, new1115 8, cbd7865e 10, 0200 12, 0308 15, 4a2a (11/15 06:18) 18, 9f7c (06:57) 21, fb04 (08:14) 24, fb04 DAgger takeover segments 48, 42e9 (09:39) 27, 42e9 DAgger 54, 3541 (10:17) 30, f64d (11:30) 33, f64d DAgger 66.
Weights -> integer repeats (the loader samples frames uniformly, so a dataset listed n times is n x as likely).
usage: python3 mk_v4c.py [--scale S] [--write]"""
import collections, glob, os, re, sys
import pandas as pd

D = '/data/lingbot/data'
TASK_T = {1: 0.28, 2: 0.24, 3: 0.48}
SUB_T = {'grasp': 0.18, 'sweep': 0.34, 'dump': 0.34, 'home': 0.14}
T12 = [('public_t12', 1), ('priv_t12_fix', 2), ('priv2_t12_fix', 2), ('priv3_t12', 3), ('own_t12_new1115', 10),
       ('public_t1_box_v3fix_named', 1), ('public_t2_box_v3fix_named', 1), ('own_t1_box_v7a_named', 3), ('own_t2_box_v7a_named', 3),
       ('public_t1_box_v3fix_gen', 1), ('own_t1_box_v7a_gen', 3), ('own_t2_box_v7a_gen', 3)]   # public t2 gen left out: task 2 already has the larger box sets
T3 = [('public_t3_sub', 1), ('priv_t3_sub_fix', 2), ('priv2_t3_sub_fix', 3), ('new1115_t3_sub', 8), ('own_t3_sub_new1115', 10)]
# 11/15 02:00 snapshot 53e3655c (29 task-1 eps on the confusion pairs): the latest data -> highest weight; used when staged
for n, r in [('own_t12_new0200', 11), ('own_t1_box_v7a_new0200_named', 11), ('own_t1_box_v7a_new0200_gen', 11)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 11/15 03:08 snapshot 0d3bf9f6 (2 task-1 + 36 task-2 eps, palette level mismatches ep282/294 already dropped): latest
for n, r in [('own_t12_new0308', 12), ('own_t12_box_v7a_new0308_named', 12), ('own_t12_box_v7a_new0308_gen', 12)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 11/15 04:10: box copies (cue checked by eye) of own task-1/2 episodes that had none, split by batch so each takes its
# batch's weight; named + gen like every other box set
for b, r in [('old', 3), ('1115', 10), ('0200', 11), ('0308', 12)]:
    for v in ('named', 'gen'):
        n = f'own_t12_box_extra_{b}_{v}'
        if os.path.exists(f'{D}/{n}/meta/info.json'):
            T12.append((n, r))
# 11/15 06:18 snapshot 4a2ab123: ep302 (task 2, v9 cue) + eps 303-307 (task 3, 4-subtask split): latest -> 18
for n, r in [('own_t12_new4a2a', 13), ('own_t12_box_v9_new4a2a_named', 13), ('own_t12_box_v9_new4a2a_gen', 13)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
if all(os.path.exists(f'{D}/own_t3_sub_new4a2a__{s}/meta/info.json') for s in SUB_T):
    T3.append(('own_t3_sub_new4a2a', 13))
# 11/15 06:57 snapshot 9f7c7f8c: eps 308-324 (17 task-1 eps, ep318 dropped: persimmon left the scene): latest -> 21
for n, r in [('own_t12_new9f7c', 14), ('own_t12_box_v9_new9f7c_named', 14), ('own_t12_box_v9_new9f7c_gen', 14)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 11/15 08:14 snapshot fb049f09: eps 325-332 normal task-1 demos -> 24; eps 333-338 = DAgger recordings of 5080 sessions,
# human takeover segment only (HG-DAgger / Sirius: train on the corrections, weight them above everything) -> 48; 12:10 -> 24 = 1x its batch, total DAgger capped (DAGGER_MAX)
for n, r in [('own_t12_newfb04', 15), ('own_t12_box_v9_newfb04_named', 15), ('own_t12_box_v9_newfb04_gen', 15),
             ('own_t1_dagger_fb04', 23)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 11/15 09:39 snapshot 42e9a0f4 (+5d3bafef): eps 340-373, all task 2. Normal demos 340-342/357-361 -> 27 (above fb04's 24);
# DAgger runs 343-356/362-373 (08:23-09:3x 5080 sessions), human takeover segments only (dagger_seg.py, 26 segs, 536 s)
# -> 54 = 2x the newest demos; 12:10 -> 27 = 1x its batch (DAGGER_MAX cap). Two ladders: demos newer > older, DAgger newer > older, DAgger = 2x its batch's demos.
for n, r in [('own_t12_new42e9', 16), ('own_t12_box_v9_new42e9_named', 16), ('own_t12_box_v9_new42e9_gen', 16),
             ('own_t2_dagger_42e9', 24)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 11/15 10:17 snapshot 3541a365: eps 374-388, no DAgger. Task 2 374-382, task 1 385-388 (plain 13 eps, box 9) -> 30;
# eps 383/384 = task-3 sweeping demos mislabelled with a task-2 prompt, relabelled and split into the 4 subtasks -> 30
for n, r in [('own_t12_new3541', 17), ('own_t12_box_v9_new3541_named', 17), ('own_t12_box_v9_new3541_gen', 17)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
if all(os.path.exists(f'{D}/own_t3_sub_new3541__{s}/meta/info.json') for s in SUB_T):
    T3.append(('own_t3_sub_new3541', 17))
# 11/15 11:30 snapshots 4e7de127 + f64d6d11: eps 389-411, all task 1. Demos -> 33 (above 3541's 30); DAgger recordings of
# the 10:59-11:19 5080 sessions (401-411), human takeover segments only -> 66 = 2x; 12:10 -> 33 = 1x its batch (DAGGER_MAX cap). Box set from v9 or v11 cues.
for n, r in [('own_t12_newf64d', 18), ('own_t12_box_v9_newf64d_named', 18), ('own_t12_box_v9_newf64d_gen', 18),
             ('own_t12_box_v11_newf64d_named', 18), ('own_t12_box_v11_newf64d_gen', 18), ('own_t1_dagger_f64d', 27)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 11/15 14:01 snapshot b6ef6674 (STAGED 14:20): eps 412-438, no task 3. Plain demos 412-422 (task 1 x10, task 2 x1), box set
# from YOLO v12 cues (9 eps; 414/418 pear undetected -> no cue), DAgger = task-2 human takeover segments (15) -> ladder v2
# batch 19: demos 19, DAgger 1.5x = 28.5.
for n, r in [('own_t12_newb6ef', 19), ('own_t12_box_v12_newb6ef_named', 19), ('own_t12_box_v12_newb6ef_gen', 19),
             ('own_t2_dagger_b6ef', 28.5)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 11/15 16:05 snapshot 908aa6e5 (batch 20): eps 439-474, task 2 439-465 (some DAgger), task 3 466-474 -> demos 20, DAgger 30.
# Names follow the b6ef / 3541 pattern; tools/b20_auto.sh aborts if a STAGED set is missing from the list.
for n, r in [('own_t12_new908a', 20), ('own_t12_box_v12_new908a_named', 20), ('own_t12_box_v12_new908a_gen', 20),
             ('own_t2_dagger_908a', 30)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 908a task 3: eps 466-474 (harness sessions), human takeover segments only (dagger_seg ranges), cut into the four
# subtasks by real_split.segment on the whole episode (seg_t3_dagger_908a.py; boundaries checked by eye on 466/467/469/472/
# 474, all 9 end states by the data session). 10500 (v4p) used all 35 pieces as one DAgger set own_t3_dagger_908a x30.
# 16:32 user rule: only the subtask the human took over IN is DAgger (x30, 1.5x batch 20; first piece of each human
# segment, 11 pieces) -> own_t3_dagger_908a_to; the later subtasks the human then did in full are normal demos (x20,
# 24 pieces) -> own_t3_sub_new908a. Labels: labels/t3_dagger_908a_takeover.json, labels/t3_followup_908a.json.
# Neither covers all four subtasks (takeover has no dump, follow-up no grasp); missing subtasks get no lines.
if any(os.path.exists(f'{D}/own_t3_sub_new908a__{s}/meta/info.json') for s in SUB_T):
    T3.append(('own_t3_sub_new908a', 20))
if any(os.path.exists(f'{D}/own_t3_dagger_908a_to__{s}/meta/info.json') for s in SUB_T):
    T3.append(('own_t3_dagger_908a_to', 30))
# 11/15 16:42 snapshot 77235417 (batch 21): eps 475-485, all task 3 (data session: dagger_seg + subtask check). Demos and
# follow-up pieces -> own_t3_sub_new7723 x21; takeover pieces -> own_t3_dagger_7723_to x31.5 (1.5x). tools/b21_auto.sh
# splits both into __{subtask} after STAGED_7723 and aborts if a staged set is missing from the list.
if any(os.path.exists(f'{D}/own_t3_sub_new7723__{s}/meta/info.json') for s in SUB_T):
    T3.append(('own_t3_sub_new7723', 21))
if any(os.path.exists(f'{D}/own_t3_dagger_7723_to__{s}/meta/info.json') for s in SUB_T):
    T3.append(('own_t3_dagger_7723_to', 31.5))
# 11/15 18:40 snapshot 2c3d609e (batch 22, preheated 19:00): eps 486-510, task 3 486-493, task 2 494-510 (some DAgger) ->
# ladder v2: demos 22, DAgger 1.5x = 33. Box set cues from YOLO v13 (what the 5080 runs since 18:29), named + gen as
# usual. Task 3 as 7723: demos + follow-up pieces -> own_t3_sub_new2c3d, takeover pieces -> own_t3_dagger_2c3d_to.
# Only non-empty sets get staged; tools/b22_auto.sh aborts if a staged set is missing from the list.
for n, r in [('own_t12_new2c3d', 22), ('own_t12_box_v13_new2c3d_named', 22), ('own_t12_box_v13_new2c3d_gen', 22),
             ('own_t2_dagger_2c3d', 33)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
if any(os.path.exists(f'{D}/own_t3_sub_new2c3d__{s}/meta/info.json') for s in SUB_T):
    T3.append(('own_t3_sub_new2c3d', 22))
if any(os.path.exists(f'{D}/own_t3_dagger_2c3d_to__{s}/meta/info.json') for s in SUB_T):
    T3.append(('own_t3_dagger_2c3d_to', 33))
# 11/15 21:36 snapshot e94f77bf (batch 23): eps 511-528, all task 2 normal teleop (no policy frames -> no DAgger, no task
# 3) -> ladder v2 demos 23. Box set from YOLO v13 cues (17 eps; 516 has no cue), named + gen as usual. tools/b23_auto.sh.
for n, r in [('own_t12_newe94f', 23), ('own_t12_box_v13_newe94f_named', 23), ('own_t12_box_v13_newe94f_gen', 23),
             ('own_t2_dagger_e94f', 34.5)]:
    if os.path.exists(f'{D}/{n}/meta/info.json'):
        T12.append((n, r))
# 11/15 22:40: batches from 24 on come from tools/batches.txt (one tag per line in collection order, optional 2nd word
# "hold" for tools/auto_multi.sh; '#' comments). Ladder v2: line i (0-based) -> demos 24 + i, DAgger 1.5x. A batch counts
# only once STAGED_<tag>.txt exists (no half-copied sets). Names follow the data session's pattern:
#   own_t12_new<tag>, own_t12_box_v13_new<tag>_{named,gen}, own_t1_dagger_<tag>, own_t2_dagger_<tag>,
#   own_t3_sub_new<tag>__<subtask>, own_t3_dagger_<tag>_to__<subtask>.
BATCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'batches.txt')
NEW_TAGS = [l.split()[0] for l in open(BATCH_FILE) if l.strip() and not l.startswith('#')] if os.path.exists(BATCH_FILE) else []
for i, tag in enumerate(NEW_TAGS):
    r = 24 + i
    if not os.path.exists(f'{D}/STAGED_{tag}.txt'):
        continue
    for n, rr in [(f'own_t12_new{tag}', r), (f'own_t12_box_v13_new{tag}_named', r), (f'own_t12_box_v13_new{tag}_gen', r),
                  (f'own_t1_dagger_{tag}', 1.5 * r), (f'own_t2_dagger_{tag}', 1.5 * r)]:
        if os.path.exists(f'{D}/{n}/meta/info.json'):
            T12.append((n, rr))
    if any(os.path.exists(f'{D}/own_t3_sub_new{tag}__{s}/meta/info.json') for s in SUB_T):
        T3.append((f'own_t3_sub_new{tag}', r))
    if any(os.path.exists(f'{D}/own_t3_dagger_{tag}_to__{s}/meta/info.json') for s in SUB_T):
        T3.append((f'own_t3_dagger_{tag}_to', 1.5 * r))

# 11/15 03:00: mislabelled demos dropped into *_d copies (drop_episodes.py): public_t1_box_v3fix eps 52/96 (box on the
# wrong fruit), own 0200 ep 22 (papaya called mango). Use the clean copy whenever it exists.
CLEAN = {'public_t1_box_v3fix_named': 'public_t1_box_v3fix_d_named', 'public_t1_box_v3fix_gen': 'public_t1_box_v3fix_d_gen',
         'own_t12_new0200': 'own_t12_new0200_d', 'own_t1_box_v7a_new0200_named': 'own_t1_box_v7a_new0200_d_named',
         'own_t1_box_v7a_new0200_gen': 'own_t1_box_v7a_new0200_d_gen'}
# 11/15: ep0193 has the wrong object and destination; exclude raw and both box-prompt copies.
CLEAN.update({'own_t12_new1115': 'own_t12_new1115_drop193', 'own_t2_box_v7a_named': 'own_t2_box_v7a_drop193_named', 'own_t2_box_v7a_gen': 'own_t2_box_v7a_drop193_gen'})
T12 = [(CLEAN[n] if n in CLEAN and os.path.exists(f'{D}/{CLEAN[n]}/meta/info.json') else n, r) for n, r in T12]


def frames_by_task(name):
    eps = pd.concat([pd.read_parquet(f, columns=['length', 'tasks']) for f in glob.glob(f'{D}/{name}/meta/episodes/**/*.parquet', recursive=True)])
    c = collections.Counter()
    for n, ts in zip(eps['length'], eps['tasks']):
        c[1 if 'plate' in ts[0] else 2] += n
    return c


fr12 = {n: frames_by_task(n) for n, _ in T12}
fr3 = {(n, s): sum(pd.read_parquet(f, columns=['length'])['length'].sum() for f in glob.glob(f'{D}/{n}__{s}/meta/episodes/**/*.parquet', recursive=True))
       for n, _ in T3 for s in SUB_T}
# 12:10 user: "dagger权重太大，把之前采的质量好的遥操数据都覆盖了" -> DAgger sets take their batch's demo weight (1x, not 2x)
# and ALL DAgger sets together are capped at DAGGER_MAX of the samples; when new DAgger batches push the total over, every
# DAgger weight is scaled down by one common factor (order newer > older inside DAgger is kept). DAgger is left out of the
# demo recency ladder / caps: the ladder protects demos, the cap protects demos from DAgger.
# 13:00 user: "同一批的演示比dagger权重小，但是总的趋势是新数据权重更大就行" -> gentler demo ladder (+1 per batch from 0200 on:
# 11, 12, ... 18) and DAgger = 1.5x its batch's demos (fb04 23, 42e9 24, f64d 27). The cap stays as a safety net for later
# DAgger batches, but never pushes DAgger below 1.1x its batch's demos (DM_MIN = 1.1 / 1.5).
DAGGER_MAX = float(sys.argv[sys.argv.index('--dagger-max') + 1]) if '--dagger-max' in sys.argv else 0.08
DM_MIN = 1.1 / 1.5
# 16:28: task-3 DAgger has its own cap (13% of the task-3 samples = the share task-1/2 DAgger has inside tasks 1/2), so it neither squeezes the task-1/2 DAgger below
# 1.1x its batch's demos nor floods the task-3 teleop demos. The task-1/2 cap above counts task-1/2 DAgger only.
T3_DAGGER_MAX = float(sys.argv[sys.argv.index('--t3-dagger-max') + 1]) if '--t3-dagger-max' in sys.argv else 0.13
scale = float(sys.argv[sys.argv.index('--scale') + 1]) if '--scale' in sys.argv else 1.0
frames = {n: sum(fr12[n].values()) for n, _ in T12}
frames.update({f'{n}__{s}': fr3[(n, s)] for n, _ in T3 for s in SUB_T})


def IS_D(n):
    return '_dagger_' in n


def solve(dm, d3=1.0, verbose=False):
    # subtask multipliers so that recency-weighted task-3 frames match SUB_T (task-3 DAgger scaled by its cap factor d3)
    w3 = {n: r * (d3 if IS_D(n) else 1.0) for n, r in T3}
    nat = {s: sum(w3[n] * fr3[(n, s)] for n, _ in T3) for s in SUB_T}
    tot_nat = sum(nat.values())
    mult = {s: SUB_T[s] / (nat[s] / tot_nat) for s in SUB_T}
    w = {n: float(r) * (dm if IS_D(n) else 1.0) for n, r in T12}
    w.update({f'{n}__{s}': w3[n] * mult[s] for n, _ in T3 for s in SUB_T})
    # balance task 1 vs task 2: scale UP only the short task's single-task (box) sets, never down. An older set may grow
    # at most to the newest batch's base weight - 1 (user rule: the newest data always weighs the most); clamped sets are
    # fixed and the factor re-solved on the rest.
    # 09:50: DAgger correction sets keep exactly their set weight (never scaled by task balancing), so their share stays
    # what we chose (2x the batch's demos) instead of growing by the balance factor
    one = {n: (0 if '_dagger_' in n else 1 if fr12[n][2] == 0 else 2 if fr12[n][1] == 0 else 0) for n, _ in T12}
    base_w = {n: w[n] for n, _ in T12}
    top = max(r for n, r in T12 if not n.startswith('public') and not IS_D(n))
    cap = top - 1
    # 06:40: an own batch may also never pass the next newer own batch (0200 x1.98 would have reached 17 > 0308's 15)
    own_bases = sorted({r for n, r in T12 if not n.startswith('public') and not IS_D(n)})
    capn = {n: (cap if n.startswith('public') else min([cap] + [b - 1 for b in own_bases if b > base_w[n]])) for n, _ in T12}
    # 13:00: the newest demo batch is not scaled up by balancing either (keeps DAgger = 1.5x its batch's demos); the
    # balance then comes from older / public single-task sets, which is what the user wants (older clean teleop counts more)
    for n, _ in T12:
        if not n.startswith('public') and not IS_D(n) and base_w[n] == top:
            capn[n] = base_w[n]
    r = TASK_T[1] / TASK_T[2]
    t1 = sum(w[n] * fr12[n][1] for n, _ in T12); t2 = sum(w[n] * fr12[n][2] for n, _ in T12)
    k = 1 if t1 / t2 < r else 2
    clamped, f = set(), 1.0
    for _ in range(20):
        S = [n for n, _ in T12 if one[n] == k and n not in clamped]
        X = sum(base_w[n] * fr12[n][k] for n in S)
        O1 = sum((capn[n] if n in clamped else base_w[n]) * fr12[n][1] for n, _ in T12 if n not in S)
        O2 = sum((capn[n] if n in clamped else base_w[n]) * fr12[n][2] for n, _ in T12 if n not in S)
        f = max(1.0, (r * O2 - O1) / X if k == 1 else (O1 / r - O2) / X)
        new_clamp = {n for n in S if base_w[n] * f > capn[n]}
        if not new_clamp:
            break
        clamped |= new_clamp
    for n, _ in T12:
        w[n] = capn[n] if n in clamped else base_w[n] * (f if (one[n] == k) else 1.0)
    if verbose:
        print(f'task-{k}-only sets x{f:.2f}' + (f'; capped below the next newer batch: {sorted((n, capn[n]) for n in clamped)}' if clamped else ''))
    # scale task 3 against tasks 1/2 so task 3 gets its target share
    f12 = sum(w[n] * sum(fr12[n].values()) for n, _ in T12)
    f3 = sum(w[f'{n}__{s}'] * fr3[(n, s)] for n, _ in T3 for s in SUB_T)
    k3 = TASK_T[3] / (1 - TASK_T[3]) * f12 / f3
    for n, _ in T3:
        for s in SUB_T:
            w[f'{n}__{s}'] *= k3
    wf = {n: v * scale for n, v in w.items() if frames[n] > 0}   # no lines for absent subtasks
    return {n: max(1, round(v)) for n, v in wf.items()}, k, f, clamped, wf


def shares(rep):
    tot = sum(rep[n] * frames[n] for n in rep)
    t3 = sum(rep[n] * frames[n] for n in rep if '__' in n)
    d12 = sum(rep[n] * frames[n] for n in rep if IS_D(n) and '__' not in n) / tot
    d3 = sum(rep[n] * frames[n] for n in rep if IS_D(n) and '__' in n) / t3 if t3 else 0.0
    return d12, d3


dm, d3 = 1.0, 1.0
for _ in range(20):
    rep, k, f, clamped, _ = solve(dm, d3)
    dshare, d3share = shares(rep)
    ok12 = dshare <= DAGGER_MAX + 0.002 or dm <= DM_MIN
    ok3 = d3share <= T3_DAGGER_MAX + 0.002
    if ok12 and ok3:
        break
    if not ok12:
        dm = max(DM_MIN, dm * DAGGER_MAX / dshare)
    if not ok3:
        d3 = d3 * T3_DAGGER_MAX / d3share
rep, k, f, clamped, wf = solve(dm, d3, verbose=True)

# 11/15 16:55 user (via data session), from v4q on: raise the combined share of our own batches from 1115 on (1115 0200 0308
# 4a2a 9f7c fb04 42e9 3541 f64d b6ef 908a 7723 ...) by 3-5 pts with their internal ratios kept, taking it from the rest
# (public, priv*, old box sets) inside each task so T1/T2/T3 stay where solve() put them. Applied after the DAgger caps:
# one common factor g on the group; task-3 others x h3; task-1/2 others: mixed sets x h12, task-1-only x a, task-2-only x b.
GROUP_BOOST = float(sys.argv[sys.argv.index('--group-boost') + 1]) if '--group-boost' in sys.argv else 0.04
GROUP_RE = re.compile('(' + '|'.join(['1115', '0200', '0308', '4a2a', '9f7c', 'fb04', '42e9', '3541', 'f64d', 'b6ef', '908a',
                                          '7723', '2c3d', 'e94f'] + NEW_TAGS) + ')')


def in_group(n):
    return bool(GROUP_RE.search(n)) and not n.startswith(('public', 'priv'))


def task_w(n, r):
    return (0, 0, r * frames[n]) if '__' in n else (r * fr12[n][1], r * fr12[n][2], 0)


def boost(extra):
    """Group factor g so the own batches gain `extra` of the samples (before rounding); others shrink inside each task."""
    tw = {n: task_w(n, wf[n]) for n in wf}
    T = [sum(v[i] for v in tw.values()) for i in range(3)]
    G = [sum(tw[n][i] for n in wf if in_group(n)) for i in range(3)]
    tot = sum(T); s0 = sum(G) / tot
    g = (s0 + extra) * tot / sum(G)
    O3 = [n for n in wf if not in_group(n) and '__' in n]
    O12 = [n for n in wf if not in_group(n) and '__' not in n]
    mixed = [n for n in O12 if fr12[n][1] and fr12[n][2]]
    only1 = [n for n in O12 if fr12[n][1] and not fr12[n][2]]
    only2 = [n for n in O12 if fr12[n][2] and not fr12[n][1]]
    h3 = (T[2] - g * G[2]) / sum(tw[n][2] for n in O3)
    h12 = (T[0] + T[1] - g * (G[0] + G[1])) / sum(tw[n][0] + tw[n][1] for n in O12)
    a = (T[0] - g * G[0] - h12 * sum(tw[n][0] for n in mixed)) / sum(tw[n][0] for n in only1)
    b = (T[1] - g * G[1] - h12 * sum(tw[n][1] for n in mixed)) / sum(tw[n][1] for n in only2)
    # 22:20 (e94f = a task-2-only batch inside the group): the task-2-only others (public / old own task-2 box sets) would
    # need x0.19 and the whole boost was skipped. Hold the single-task factor that falls too low at FMIN and take the rest
    # from the mixed task-1/2 others, then re-solve the other single-task factor, so T1 and T2 stay exact.
    FMIN = 0.2
    if min(a, b) <= FMIN and min(h3, h12) > FMIN and mixed:
        m1, m2 = sum(tw[n][0] for n in mixed), sum(tw[n][1] for n in mixed)
        if b <= FMIN:
            b = FMIN
            h12 = (T[1] - g * G[1] - b * sum(tw[n][1] for n in only2)) / m2
            a = (T[0] - g * G[0] - h12 * m1) / sum(tw[n][0] for n in only1)
        else:
            a = FMIN
            h12 = (T[0] - g * G[0] - a * sum(tw[n][0] for n in only1)) / m1
            b = (T[1] - g * G[1] - h12 * m2) / sum(tw[n][1] for n in only2)
    if min(h3, h12, a, b) < FMIN - 1e-9:
        return None, s0, None, (g, h3, h12, a, b)
    fac = {n: g if in_group(n) else h3 if n in O3 else a if n in only1 else b if n in only2 else h12 for n in wf}
    out = {n: max(1, round(wf[n] * fac[n])) for n in wf}
    # rounding must not create ties in the recency ladder (task-1/2 own sets, demos and DAgger separately): bump the newer
    base0 = dict(T12)
    for _ in range(5):
        bumped = False
        for n, a_ in base0.items():
            for m, b_ in base0.items():
                if (not n.startswith('public') and not m.startswith('public') and IS_D(n) == IS_D(m) and a_ < b_
                        and out[n] >= out[m]):
                    out[m] = out[n] + 1; bumped = True
        if not bumped:
            break
    tw2 = {n: task_w(n, out[n]) for n in out}
    s1 = sum(sum(tw2[n]) for n in out if in_group(n)) / sum(sum(v) for v in tw2.values())
    return out, s0, s1, (g, h3, h12, a, b)


if GROUP_BOOST > 0:
    extra = GROUP_BOOST
    for _ in range(8):   # x1 floors and the order bumps eat part of the gain: raise the internal target until it lands
        out, s0, s1, fx = boost(extra)
        if out is None or s1 >= s0 + GROUP_BOOST - 0.002:
            break
        extra += s0 + GROUP_BOOST - s1
    partial = ''
    if out is None:
        # 11/16 02:10 (batches 26-28 add task-1/2 data, little non-group task-1/2 data is left to take from: h12 < 0): boost as
        # far as the x0.2 floor allows instead of skipping it - bisect the largest feasible internal target below the failed one
        lo, hi, best = 0.0, extra, None
        for _ in range(16):
            mid = (lo + hi) / 2
            o_, s0_, s1_, f_ = boost(mid)
            if o_ is None:
                hi = mid
            else:
                lo, best = mid, (o_, s0_, s1_, f_)
        if best is not None and best[2] > best[1] + 0.005:
            out, s0, s1, fx = best
            partial = ', PARTIAL: full target infeasible with others >= x0.2, largest feasible taken'
    if out is None:
        print('WARNING group boost skipped: others would shrink too far (g h3 h12 a b = ' + ' '.join(f'{x:.2f}' for x in fx) + ')')
    else:
        rep = out
        # the x1 floors keep task-1/2 others from shrinking, so task 3 drops below TASK_T[3]: scale the task-3 sets with
        # repeats >= 10 (rounding stays fine there) back up to the target
        t12w = sum(task_w(n, rep[n])[0] + task_w(n, rep[n])[1] for n in rep)
        F = sum(rep[n] * frames[n] for n in rep if '__' in n and rep[n] < 10)
        V = sum(rep[n] * frames[n] for n in rep if '__' in n and rep[n] >= 10)
        c3 = (TASK_T[3] / (1 - TASK_T[3]) * t12w - F) / V
        if 0.8 < c3 < 1.25:
            rep = {n: round(r * c3) if '__' in n and r >= 10 else r for n, r in rep.items()}
        tw2 = {n: task_w(n, rep[n]) for n in rep}
        s1 = sum(sum(tw2[n]) for n in rep if in_group(n)) / sum(sum(v) for v in tw2.values())
        print(f'group boost: own batches 1115+ {100 * s0:.1f}% -> {100 * s1:.1f}% (target +{100 * GROUP_BOOST:.0f} pts; group x{fx[0]:.3f}, '
              f'others: task-3 x{fx[1]:.2f}, task-1/2 mixed x{fx[2]:.2f}, task-1-only x{fx[3]:.2f}, task-2-only x{fx[4]:.2f}; x1 sets stay at 1; '
              f'task-3 sets with x>=10 re-scaled x{c3:.3f} to hold task 3 at {100 * TASK_T[3]:.0f}%{partial})')

task_of = {n: None for n, _ in T12}
tot = sum(rep[n] * frames[n] for n in rep)
t1 = sum(rep[n] * fr12[n][1] for n, _ in T12) / tot
t2 = sum(rep[n] * fr12[n][2] for n, _ in T12) / tot
t3 = 1 - t1 - t2
sub = {s: sum(rep.get(f'{n}__{s}', 0) * fr3[(n, s)] for n, _ in T3) for s in SUB_T}
st = sum(sub.values())
print(f'scale {scale}: lines {sum(rep.values())} | T1 {100 * t1:.1f}% T2 {100 * t2:.1f}% T3 {100 * t3:.1f}% | in T3: ' +
      ' '.join(f'{s} {100 * v / st:.1f}%' for s, v in sub.items()))
for n in rep:
    print(f'  {n:34s} x{rep[n]:<3d} {frames[n]:7d} frames {100 * rep[n] * frames[n] / tot:5.1f}%')
# recency check (demos only; DAgger has its own cap): the newest demo batch keeps a higher repeat than every older own set
base = dict(T12)
top = max(r for n, r in base.items() if not n.startswith('public') and not IS_D(n))
newest_rep = min(rep[n] for n, r in base.items() if r == top and not IS_D(n))
bad = [n for n, r in base.items() if r < top and not n.startswith('public') and not IS_D(n) and rep[n] >= newest_rep]
print(f'  recency: newest demo batch (base {top}) x{newest_rep}; ' + ('OK, above every older own task-1/2 demo set' if not bad else f'WARNING older sets at/above it: {bad}'))
# pairwise inside each kind: demos newer > older, DAgger newer > older
order = [(n, m) for n, a in base.items() for m, b in base.items() if not n.startswith('public') and not m.startswith('public')
         and IS_D(n) == IS_D(m) and a < b and rep[n] >= rep[m]]
print('  recency pairwise: ' + ('OK, each own batch below every newer one (demos and DAgger separately)' if not order else f'WARNING older >= newer: {order[:6]}'))
dsets = {n: rep[n] for n in rep if IS_D(n) and '__' not in n}
for dn in dsets:   # same-batch check: DAgger above its batch's plain demos
    tag = dn.split('_dagger_')[1]
    demo = [m for m in rep if not IS_D(m) and m.endswith('new' + tag)]
    if demo:
        print(f'  {dn}: x{rep[dn]} vs same-batch demos {demo[0]} x{rep[demo[0]]} -> {rep[dn] / rep[demo[0]]:.2f}x')
print(f'  DAgger: {100 * sum(rep[n] * frames[n] for n in dsets) / tot:.1f}% of samples (cap {100 * DAGGER_MAX:.0f}%, common factor {dm:.2f}): ' +
      ', '.join(f'{n} x{r}' for n, r in dsets.items()))
d3sets = {n: rep[n] for n in rep if IS_D(n) and '__' in n}
if d3sets:
    t3f = sum(rep[n] * frames[n] for n in rep if '__' in n)
    print(f'  task-3 DAgger: {100 * sum(rep[n] * frames[n] for n in d3sets) / t3f:.1f}% of task-3 samples, '
          f'{100 * sum(rep[n] * frames[n] for n in d3sets) / tot:.1f}% of all (cap {100 * T3_DAGGER_MAX:.0f}% of task 3, factor {d3:.2f}): ' +
          ', '.join(f'{n} x{r}' for n, r in d3sets.items()))
newest = sum(rep[n] * frames[n] for n in rep if 'new1115' in n and not n.startswith('new1115')) / tot
print(f'  newest snapshot (own_*_new1115) {100 * newest:.1f}% | new1115_t3_sub {100 * sum(rep[n] * frames[n] for n in rep if n.startswith("new1115")) / tot:.1f}%')
if '--write' in sys.argv:
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else '~/lingbot_train/lists/v4c.txt'
    out = os.path.expanduser(out)
    with open(out + '.tmp', 'w') as f:   # write then rename: an armed switch never reads a half-written list
        for n, r in rep.items():
            f.write(f'agilex_cobot_magic {D}/{n}\n' * r)
    os.replace(out + '.tmp', out)
    print('wrote', out)
