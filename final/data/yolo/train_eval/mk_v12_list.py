"""v12 train list = BASE list + hand-checked frames of own eps 302-388 (ds_v12_new) x6, frames of episodes with confusable
objects (tomato, wax apple, lip gloss/lotion, carrot/red chili, brushes, puff...) another x6.
usage: python mk_v11_list.py NAME BASE_LIST     e.g. v12 ds_v11_train.txt"""
import json, re, sys
from pathlib import Path
Y = Path.home() / 'locany/yolo'; name, basef = sys.argv[1], sys.argv[2]
base = (Y / basef).read_text().split()
new = sorted(str(p) for p in (Y / 'ds_v12_new/images/train').glob('*.jpg')); hard = set(json.load(open(Y / 'ds_v12_new/hard.json'))['hard_eps'])
hardf = [p for p in new if int(re.search(r'n12_ep(\d+)_', p).group(1)) in hard]
(Y / f'ds_{name}_train.txt').write_text('\n'.join(base + new * 6 + hardf * 6) + '\n')
names = (Y / 'ds_v6/dataset.yaml').read_text().split('names:', 1)[1]
(Y / f'ds_{name}.yaml').write_text(f'path: {Y}\ntrain: ds_{name}_train.txt\nval: ds_v7a_val.txt\nnames:{names}')
print({'base': len(base), 'new': len(new), 'hard_frames': len(hardf), 'per_epoch': len(base) + 6 * len(new) + 6 * len(hardf)})
