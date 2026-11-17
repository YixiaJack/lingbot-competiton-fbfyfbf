"""HPC replica of the dev machine's training config: machine paths replaced, key=value overrides (value DELETE drops
the key).   python mkcfg_hpc.py SRC_YAML OUT_YAML LIST_TXT [key=value ...]"""
import os
import sys, yaml
R = os.path.expandvars('$SCRATCH/lingbot/devrep_20261114')
M_DEV = '/mnt/forge/models/system/00000000-0000-0000-0000-000000000002'
DEV_HOME = os.environ.get('DEV_HOME', '<dev-home>')   # home directory on the dev machine
W_DEV = DEV_HOME + '/lingbot_train'
src, out, lst = sys.argv[1:4]
c = yaml.safe_load(open(src).read().replace(M_DEV, f'{R}/model').replace(W_DEV + '/rc', f'{R}/aux/rc'))
c['data']['train_path'] = lst
for kv in sys.argv[4:]:
    k, v = kv.split('=', 1); d = c; ks = k.split('.')
    for x in ks[:-1]: d = d.setdefault(x, {})
    if v == 'DELETE': d.pop(ks[-1], None)
    else: d[ks[-1]] = yaml.safe_load(v)
txt = yaml.safe_dump(c, sort_keys=False)
assert '/mnt/forge' not in txt and DEV_HOME not in txt, 'dev path left in config'
open(out, 'w').write(txt); print('wrote', out)
