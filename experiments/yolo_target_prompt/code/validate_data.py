"""Read representative episode boundaries through the actual LingBot dataset reader."""
import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.expandvars('$SCRATCH/lingbot/dev/lingbot-vla-v2-dino'))
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
from cues import apply_cue


def validate(output, sources, variant):
    ds = LeRobotDataset(str(output), delta_timestamps={'action': [i / 50 for i in range(50)]},
                        video_backend='torchcodec')
    lengths = [len(np.load(d / 'episode.npz')['state']) - 1 for d in sources]
    assert len(ds) == sum(lengths)
    assert ds.num_episodes == len(sources)
    checked = []
    for ep in sorted({0, len(sources) // 2, len(sources) - 1}):
        source = np.load(sources[ep] / 'episode.npz')
        meta = json.loads((sources[ep] / 'complete.json').read_text())
        for frame in (0, lengths[ep] - 1):
            item = ds[sum(lengths[:ep]) + frame]
            np.testing.assert_allclose(item['observation.state'].numpy(), source['state'][frame], atol=1e-6)
            assert tuple(item['action'].shape) == (50, 14)
            np.testing.assert_allclose(item['action'][0].numpy(), source['state'][frame + 1], atol=1e-6)
            expected, prompt = apply_cue(source['cam_high'][frame], meta['instruction'],
                                        meta['boxes'][meta['target']], meta['boxes'][3 + meta['destination']], variant)
            assert item['task'] == prompt
            errors = {}
            for cam in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
                pixels = item[f'observation.images.{cam}'].permute(1, 2, 0).numpy()
                target = expected if cam == 'cam_high' else source[cam][frame]
                assert pixels.shape == target.shape
                assert pixels.dtype == np.uint8, pixels.dtype
                error = float(np.abs(pixels / 255. - target / 255.).mean())
                assert error < .03, (ep, frame, cam, error)
                errors[cam] = error
            if frame == lengths[ep] - 1:
                assert item['action_is_pad'][1:].all()
            checked.append({'episode': ep, 'frame': frame, 'image_mae': errors})
    report = {'episodes': ds.num_episodes, 'frames': len(ds), 'reader': 'LingBot LeRobotDataset', 'checked': checked}
    (output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--episodes', type=int, required=True)
    a = p.parse_args()
    validate(a.root / 'lerobot' / a.variant,
             [a.root / 'episodes' / f'{i:04d}' for i in range(a.episodes)], a.variant)
