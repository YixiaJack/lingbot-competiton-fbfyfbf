"""Training configs, group configs, eval lists and phases for the cue A/B (submits nothing).
Arms: s22_* start from the prelim 22k+8k RoboTwin model (the team's choice at 2:18: the untrained base cannot do these
sim tasks, so the comparison would read zero everywhere); base_* start from the official 6B base (what the final uses),
fine-tuned with the official real-robot learning rate. Cue mix: box/coords runs train on 50% plain + 50% cued samples.
"""
import os
import json
from pathlib import Path
import yaml

R = Path(os.path.expandvars('$SCRATCH/lingbot/final_real_20261112')); BASE = R.parent
KIT = BASE / 'tmp/gpu_queue_kit'; WORK = BASE / 'LingBotRoboTwin'
S22 = BASE / 'runs/strong_scratch22k_cont8k/checkpoints/global_step_8000/hf_ckpt'
OFFICIAL = BASE / 'ckpts/lingbot-vla-v2-6b'
# 11/12 (decision): only the prelim 22k+8k RoboTwin model, plain text vs box, as the team decided at 2:18 (the official base
# has never been trained in this simulator and could read near zero in both arms, so the comparison would show
# nothing). The official 6B is still checked for the plug-in (integ_6b.py), just not trained here.
# 11/14: third arm, box with the target box dropped once the gripper closes (cues_real.GraspLatch); same data, steps, lr.
RUNS = {  # name: (init, variant, steps, lr, lr_min)
    's22_text': (S22, 'text', 2000, 1e-5, 5e-6), 's22_box': (S22, 'box', 2000, 1e-5, 5e-6),
    's22_boxgrasp': (S22, 'boxgrasp', 2000, 1e-5, 5e-6),
}
for d in ('configs', 'lists', 'runs', 'logs'):
    (R / d).mkdir(exist_ok=True)
for name, (init, variant, steps, lr, lr_min) in RUNS.items():
    lst = R / 'lists' / f'train_{name}.txt'
    lst.write_text(f'robotwin_clean {R}/lerobot/text\n' + (f'robotwin_clean {R}/lerobot/{variant}\n' if variant != 'text' else ''))
    cfg = yaml.safe_load((WORK / 'configs/robotwin_clean.yaml').read_text())
    cfg['model']['model_path'] = str(init)
    cfg['data'].update(train_path=str(lst), image_augment=False, num_workers=4)
    cfg['train'].update(output_dir=str(R / 'runs' / name), max_steps=steps, save_steps=500, lr=lr, lr_min=lr_min,
                        lr_warmup_ratio=.03, tokenizer_max_length=128, global_batch_size=128, enable_resume=True)
    (R / 'configs' / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (R / 'configs' / f'group_{name}.sh').write_text(
        f"source {KIT}/config.example.sh\nLOG_DIR={R}/logs\nGROUP_NAME=lb_real_{name}_1112\nGROUP_DIR={R}/runs/{name}\n"
        f"TRAIN_CMD='bash {R}/code/train_real.sh {name} {steps}'\n")
(R / 'lists/eval_items.txt').write_text(''.join(f'{i}\n' for i in range(60)))
(R / 'lists/variants.txt').write_text('text\nbox\nboxgrasp\n')
(R / 'lists/collect.txt').write_text(''.join(f'{i}\n' for i in range(260)))
json.dump({k: [str(v[0]), *v[1:]] for k, v in RUNS.items()}, open(R / 'ab_runs.json', 'w'), indent=1)
print('ok')
