"""v10 train list = the v9 list with the official images replaced by ds_v10_off (by-eye-corrected labels; plate-sized
object boxes dropped, palette pans deleted, checked unlabelled objects added) + own eps 189-194 (ds_v10_own) x6 +
task-3 tables with no labels (ds_v10_neg: dustpan / brush / paper balls taught as background) x2."""
import sys
from pathlib import Path
Y = Path.home() / 'locany/yolo'
name, kneg = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ('v10', 2)   # e.g. v10n 0 / v10h 1
base = [l for l in (Y / 'ds_v9_train.txt').read_text().split() if '/ds_v6/' not in l]
g = lambda d: sorted(str(p) for p in (Y / d / 'images/train').glob('*.jpg'))
off, own, neg = g('ds_v10_off'), g('ds_v10_own'), g('ds_v10_neg')
(Y / f'ds_{name}_train.txt').write_text('\n'.join(base + off + own * 6 + neg * kneg) + '\n')
names = (Y / 'ds_v6/dataset.yaml').read_text().split('names:', 1)[1]
(Y / f'ds_{name}.yaml').write_text(f'path: {Y}\ntrain: ds_{name}_train.txt\nval: ds_v7a_val.txt\nnames:{names}')
print({'base_non_official': len(base), 'official': len(off), 'own': len(own), 'neg': len(neg), 'kneg': kneg, 'per_epoch': len(base) + len(off) + 6 * len(own) + kneg * len(neg)})
