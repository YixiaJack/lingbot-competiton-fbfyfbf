import os
import json
from pathlib import Path
import statistics
import random

ROOT=Path(os.path.expandvars('$SCRATCH/lingbot/final_yolo_20261111'))
groups={}
for p in (ROOT/'results').glob('*/*/result.json'):
    r=json.loads(p.read_text())
    if r.get('metric_version')!=2: continue
    key=(p.parent.parent.name,r['split'])
    groups.setdefault(key,[]).append(r)
out=[]
for (name,split),rows in sorted(groups.items()):
    n=len(rows);k=sum(r['first_approach_correct'] for r in rows);s=sum(r['success'] for r in rows)
    # Wilson interval: small preliminary batches must not appear conclusive.
    z=1.96;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    delta=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    warm=[x for r in rows for x in r['latency_ms'][1:]]
    out.append({'condition':name,'split':split,'n':n,'first_correct':k,'first_rate':k/n,
                'first_95ci':[max(0,center-delta),min(1,center+delta)],'successes':s,'success_rate':s/n,
                'no_approach':sum(r['first_approach'] is None for r in rows),
                'cue_fallbacks':sum(r['mode']!='text' and not r['cue_applied'] for r in rows),
                'warm_latency_median_ms':statistics.median(warm) if warm else None})
comparisons=[]
for mode in ('box','coords'):
    reference={r['seed']:r for r in groups.get(('text_text_oracle','heldout_combo'),[])}
    for cue,source in [(mode,'oracle'),('text','oracle'),(mode,'yolo'),(mode,'jitter'),(mode,'wrong10')]:
        condition=f'{mode}_{cue}_{source}'
        treatment={r['seed']:r for r in groups.get((condition,'heldout_combo'),[])}
        shared=sorted(set(reference)&set(treatment))
        comparable=[];mismatches=[]
        for seed in shared:
            a,b=reference[seed],treatment[seed]
            if a.get('scene_fingerprint') and a.get('scene_fingerprint')==b.get('scene_fingerprint'):
                comparable.append((a,b))
            else: mismatches.append(seed)
        if comparable:
            deltas=[int(b['first_approach_correct'])-int(a['first_approach_correct']) for a,b in comparable]
            rng=random.Random(42)
            boot=sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(3000))
            comparisons.append({'condition':condition,'reference':'text_text_oracle','split':'heldout_combo',
                'paired_n':len(deltas),'first_delta_pp':100*statistics.mean(deltas),
                'bootstrap_95ci_pp':[100*boot[75],100*boot[2924]],
                'success_delta_pp':100*statistics.mean(int(b['success'])-int(a['success']) for a,b in comparable),
                'scene_mismatch_seeds':mismatches,'all_30_paired':len(deltas)==30})
        elif shared:
            comparisons.append({'condition':condition,'paired_n':0,'scene_mismatch_seeds':mismatches})
report={'groups':out,'paired_comparisons':comparisons,'interpretation':'Only completed rollout records; do not infer equivalence or >15pp benefit from incomplete runs.',
        'artifact_scope':'pre-event simulation experiment, not final-competition weights'}
p=ROOT/'summary.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(p)
print(json.dumps(report,indent=2))
