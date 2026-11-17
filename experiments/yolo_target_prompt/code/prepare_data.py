"""Build matched LeRobot variants from the same 120 successful demonstrations."""
import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from cues import apply_cue

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
MOTORS = [f'joint_{i}' for i in range(14)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', choices=['text', 'box', 'coords'], required=True)
    p.add_argument('--episodes', type=int, default=120)
    a = p.parse_args()
    root = a.root
    sources = [root / 'episodes' / f'{i:04d}' for i in range(a.episodes)]
    assert all((d / 'complete.json').exists() for d in sources), 'Demonstrations incomplete'
    output = root / 'lerobot' / a.variant
    if (output / 'READY').exists():
        if not (output / 'validation.json').exists():
            from validate_data import validate
            validate(output, sources, a.variant)
        return
    if output.exists():
        # Only quarantine this experiment's interrupted build, never overwrite a complete dataset.
        import time
        output.rename(output.with_name(output.name + f'.interrupted-{time.time_ns()}'))
    first = np.load(sources[0] / 'episode.npz')
    h, w = first['cam_high'].shape[1:3]
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (14,), 'names': [MOTORS]},
        'action': {'dtype': 'float32', 'shape': (14,), 'names': [MOTORS]},
        **{f'observation.images.{cam}': {'dtype': 'video', 'shape': (h,w,3),
                                        'names': ['height','width','channels']} for cam in CAMERAS},
    }
    ds = LeRobotDataset.create(repo_id=f'target_cue/{a.variant}', root=output, fps=50,
                               robot_type='aloha', features=features, use_videos=True,
                               image_writer_processes=2, image_writer_threads=2)
    manifest = []
    for d in sources:
        meta = json.loads((d / 'complete.json').read_text())
        data = np.load(d / 'episode.npz')
        target = meta['boxes'][meta['target']]
        dest = meta['boxes'][3+meta['destination']]
        for j in range(len(data['state'])-1):
            rgb, instruction = apply_cue(data['cam_high'][j], meta['instruction'], target, dest, a.variant)
            frame = {'observation.state': data['state'][j], 'action': data['state'][j+1], 'task': instruction,
                     **{f'observation.images.{cam}': rgb if cam == 'cam_high' else data[cam][j] for cam in CAMERAS}}
            ds.add_frame(frame)
        ds.save_episode()
        manifest.append({'id': d.name, 'seed': meta['seed'], 'target': meta['target'], 'destination': meta['destination']})
        print(f'{a.variant}: {len(manifest)}/{a.episodes} episodes', flush=True)
    ds.finalize()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    from validate_data import validate
    validate(output, sources, a.variant)
    (output / 'READY').write_text('complete\n')


if __name__ == '__main__':
    main()
