"""Write a training config from the official real_robot.yaml: machine paths filled in, overrides from argv (key=value)."""
import sys, yaml, os
W = os.path.expanduser('~/lingbot_train'); M = '/mnt/forge/models/system/00000000-0000-0000-0000-000000000002'
c = yaml.safe_load(open('/opt/lingbot-vla-v2/configs/vla/real_robot/real_robot.yaml'))
c['model']['model_path'] = M
c['model']['tokenizer_path'] = f'{M}/assets/Qwen3-VL-4B-Instruct'
c['data']['robot_config_root'] = f'{W}/rc'
c['data']['norm_stats_file'] = f'{W}/rc/agilex_cobot_magic.json'
ap = c['train']['align_params']
ap['depth']['moge_path'] = f'{M}/assets/teachers/moge2-vitb-normal.pt'
ap['depth']['morgbd_path'] = f'{M}/assets/teachers/depth/model.pt'
ap['video']['ckpt_path'] = f'{M}/assets/teachers/dino_video/teacher_step_10000.pth'
ap['video']['config_path'] = f'{M}/assets/teachers/dino_video/config.yaml'
out = sys.argv[1]
for kv in sys.argv[2:]:
    k, v = kv.split('=', 1); d = c; ks = k.split('.')
    for x in ks[:-1]: d = d[x]
    d[ks[-1]] = yaml.safe_load(v)
yaml.safe_dump(c, open(out, 'w'), sort_keys=False)
print('wrote', out)
