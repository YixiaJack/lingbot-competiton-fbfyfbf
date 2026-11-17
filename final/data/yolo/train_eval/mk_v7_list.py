"""Train/val lists + dataset yaml for a v7 run. Per epoch: corrected team frames x4, new whole frames x6, copy-paste
images x2, a seeded half of the official images. Val = corrected team val frames + official val.
usage: python mk_v7_list.py NAME PASTE_DIR NEW_DIR   -> ~/locany/yolo/ds_NAME.yaml, ds_NAME_train.txt, ds_NAME_val.txt"""
import random, sys
from pathlib import Path
Y = Path.home() / 'locany/yolo'; name, paste, new = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
g = lambda d: sorted(str(p) for p in Path(d).glob('*.jpg'))
team, pst, nw = g(Y / 'ds_v7_team/images/train'), g(paste / 'images/train'), g(new / 'images/train')
off = [p for p in g(Y / 'ds_v6/images/train') if int(Path(p).stem.split('_')[0][2:]) < 1000 and '_r' not in Path(p).stem]
random.Random(42).shuffle(off); off = off[:3500]
lines = team * 4 + nw * 6 + pst * 2 + off
(Y / f'ds_{name}_train.txt').write_text('\n'.join(lines) + '\n')
val = g(Y / 'ds_v7_team/images/val') + [p for p in g(Y / 'ds_v6/images/val') if int(Path(p).stem.split('_')[0][2:]) < 1000]
(Y / f'ds_{name}_val.txt').write_text('\n'.join(val) + '\n')
names = (Y / 'ds_v6/dataset.yaml').read_text().split('names:', 1)[1]
(Y / f'ds_{name}.yaml').write_text(f'path: {Y}\ntrain: ds_{name}_train.txt\nval: ds_{name}_val.txt\nnames:{names}')
print({'team': len(team), 'new': len(nw), 'paste': len(pst), 'official': len(off), 'per_epoch': len(lines), 'val': len(val)})
