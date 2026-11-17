"""v9 train list = the v8 list + the hand-checked newest frames (ds_v9_new: 0308 batch, 26 episodes x up to 5 frames) x6."""
from pathlib import Path
Y = Path.home() / 'locany/yolo'
base = (Y / 'ds_v8_train.txt').read_text().split()
new = sorted(str(p) for p in (Y / 'ds_v9_new/images/train').glob('*.jpg'))
(Y / 'ds_v9_train.txt').write_text('\n'.join(base + new * 6) + '\n')
names = (Y / 'ds_v6/dataset.yaml').read_text().split('names:', 1)[1]
(Y / 'ds_v9.yaml').write_text(f'path: {Y}\ntrain: ds_v9_train.txt\nval: ds_v7a_val.txt\nnames:{names}')
print({'base': len(base), 'new': len(new), 'per_epoch': len(base) + 6 * len(new)})
