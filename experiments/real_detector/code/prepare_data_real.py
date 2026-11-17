"""Matched LeRobot datasets (text / box / coords) from the same demonstrations, same format as the prelim conversion
(src/convert_robotwin_to_lerobot.py): state = 14-d RoboTwin joint_action vector, action = next frame's state, fps 50.
The box variants change only the head camera, the coords variant only the instruction. boxgrasp drops the target box
from the frame on which the gripper has closed (cues_real.GraspLatch on the recorded states, the same rule eval_real.py
applies online). Validated through LingBot's own dataset reader.
Usage: prepare_data_real.py --root R --variant text|box|boxgrasp|coords
"""
import os
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from cues_real import apply_cue, grasp_flags
from catalog import N1, N2

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
MOTORS = [f'joint_{i}' for i in range(14)]


def load(d):  # every array decompressed once: indexing an NpzFile (z['cam_high'][j]) decompresses the whole array on
    with np.load(d / 'episode.npz') as z:  # every access, ~0.4 s per camera per frame, which made a 170-frame episode take 70-100 s
        return {k: z[k] for k in z.files}


def validate(output, sources, variant):
    sys.path.insert(0, os.path.expandvars('$SCRATCH/lingbot/dev/lingbot-vla-v2-dino'))
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset as LBDataset
    ds = LBDataset(str(output), delta_timestamps={'action': [i / 50 for i in range(50)]}, video_backend='torchcodec')
    lengths = [len(np.load(d / 'episode.npz')['state']) - 1 for d in sources]
    assert len(ds) == sum(lengths) and ds.num_episodes == len(sources)
    checked = []
    for ep in sorted({0, N1 - 1, len(sources) - 1} | ({N1} if N2 else set())):
        src = load(sources[ep]); meta = json.loads((sources[ep] / 'complete.json').read_text())
        grasped = grasp_flags(src['state'])
        if variant == 'boxgrasp':   # the episode must actually close a gripper, or the target box never goes away
            assert grasped[lengths[ep] - 1], ep
        for frame in (0, lengths[ep] - 1):
            item = ds[sum(lengths[:ep]) + frame]
            np.testing.assert_allclose(item['observation.state'].numpy(), src['state'][frame], atol=1e-6)
            assert tuple(item['action'].shape) == (50, 14)
            np.testing.assert_allclose(item['action'][0].numpy(), src['state'][frame + 1], atol=1e-6)
            img, prompt = apply_cue(src['cam_high'][frame], meta['instruction'], meta['target_box'], meta['dest_box'], variant,
                                    grasped=grasped[frame])
            assert item['task'] == prompt, (item['task'], prompt)
            err = {}
            for cam in CAMERAS:
                px = item[f'observation.images.{cam}'].permute(1, 2, 0).numpy()
                tgt = img if cam == 'cam_high' else src[cam][frame]
                assert px.shape == tgt.shape and px.dtype == np.uint8
                err[cam] = float(np.abs(px / 255. - tgt / 255.).mean()); assert err[cam] < .03, (ep, frame, cam, err[cam])
            if frame == lengths[ep] - 1:
                assert item['action_is_pad'][1:].all()
            checked.append({'episode': ep, 'frame': frame, 'image_mae': err})
    (output / 'validation.json').write_text(json.dumps({'episodes': ds.num_episodes, 'frames': len(ds), 'checked': checked}, indent=1))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', choices=['text', 'box', 'boxgrasp', 'coords'], required=True)
    a = p.parse_args()
    sources = [a.root / 'episodes' / f'{i:04d}' for i in range(N1 + N2)]
    assert all((d / 'complete.json').exists() for d in sources), 'demonstrations incomplete'
    output = a.root / 'lerobot' / a.variant
    if (output / 'READY').exists():
        return
    if output.exists():  # interrupted build of this experiment only: keep it aside, never overwrite a complete dataset
        output.rename(output.with_name(output.name + f'.interrupted-{time.time_ns()}'))
    first = np.load(sources[0] / 'episode.npz'); h, w = first['cam_high'].shape[1:3]
    features = {'observation.state': {'dtype': 'float32', 'shape': (14,), 'names': [MOTORS]},
                'action': {'dtype': 'float32', 'shape': (14,), 'names': [MOTORS]},
                **{f'observation.images.{c}': {'dtype': 'video', 'shape': (h, w, 3), 'names': ['height', 'width', 'channels']}
                   for c in CAMERAS}}
    ds = LeRobotDataset.create(repo_id=f'real_cue/{a.variant}', root=output, fps=50, robot_type='aloha', features=features,
                               use_videos=True, image_writer_processes=2, image_writer_threads=2)
    for d in sources:
        meta = json.loads((d / 'complete.json').read_text()); data = load(d)
        grasped = grasp_flags(data['state'])
        for j in range(len(data['state']) - 1):
            rgb, instr = apply_cue(data['cam_high'][j], meta['instruction'], meta['target_box'], meta['dest_box'], a.variant,
                                   grasped=grasped[j])
            ds.add_frame({'observation.state': data['state'][j], 'action': data['state'][j + 1], 'task': instr,
                          **{f'observation.images.{c}': rgb if c == 'cam_high' else data[c][j] for c in CAMERAS}})
        ds.save_episode()
    ds.finalize()
    validate(output, sources, a.variant)
    (output / 'READY').write_text('complete\n')


if __name__ == '__main__':
    main()
