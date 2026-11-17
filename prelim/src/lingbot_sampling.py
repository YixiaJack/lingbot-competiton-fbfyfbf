"""Task-balanced, motion-weighted frame sampling for LingBot's MultiVLADataset.

Upstream concatenates the 50 clean datasets and samples frames uniformly, so a task's share of training equals its
share of frames (0.70% for click_bell .. 5.68% for put_bottles_dustbin). The patched MultiVLADataset.getdata calls
remap_index() when LINGBOT_TASK_SAMPLING is set:
  LINGBOT_TASK_SAMPLING=uniform  every task is drawn with probability 1/50 (task = sampler index mod 50)
  LINGBOT_TASK_SAMPLING=temp:T   temperature sampling: task i drawn with probability frames_i^(1/T) / sum_j frames_j^(1/T)
                                 (T=1: upstream's frame-proportional mix, T->inf: uniform; T=2 lifts short tasks without
                                 halving the long ones the way uniform does; arXiv 2510.19373)
  LINGBOT_FRAME_WEIGHTS=1        inside the task, a frame is drawn proportionally to meta/frame_weights.npy
                                 (src/make_frame_weights.py: frames where the arm does not move are down-weighted)
Both weights are pure functions of the clean demonstrations (rule 2.1: reweighting by data statistics).
"""

import os

import numpy as np
import torch

_CDF = {}
_TASK_CDF = {}


def _frame_cdf(multi, task):
    key = (id(multi), task)
    if key not in _CDF:
        sub = multi._datasets[task].dataset
        n = sub.num_frames
        w = np.ones(n, dtype=np.float64)
        if os.environ.get("LINGBOT_FRAME_WEIGHTS") == "1":
            w = np.load(sub.root / "meta" / "frame_weights.npy").astype(np.float64)
            assert len(w) == n, f"{sub.root}: {len(w)} weights for {n} frames"
        c = np.cumsum(w)
        _CDF[key] = c / c[-1]
    return _CDF[key]


def _task_cdf(multi, temperature):
    key = (id(multi), temperature)
    if key not in _TASK_CDF:
        frames = np.array([d.dataset.num_frames for d in multi._datasets], dtype=np.float64)
        w = frames ** (1.0 / temperature)
        c = np.cumsum(w)
        _TASK_CDF[key] = c / c[-1]
    return _TASK_CDF[key]


def remap_index(multi, idx):
    """Map the sampler's index to a (task, frame) drawn by the configured weights; returns a global frame index."""
    mode = os.environ.get("LINGBOT_TASK_SAMPLING", "")
    if mode == "uniform":
        task = idx % len(multi._datasets)
    elif mode.startswith("temp:"):
        cdf = _task_cdf(multi, float(mode.split(":", 1)[1]))
        task = min(int(np.searchsorted(cdf, float(torch.rand(())), side="right")), len(cdf) - 1)
    else:  # keep upstream's frame-proportional task choice, only reweight frames inside the task
        task = int(np.searchsorted(multi.dataset_start_index, idx, side="right") - 1)
    cdf = _frame_cdf(multi, task)
    local = min(int(np.searchsorted(cdf, float(torch.rand(())), side="right")), len(cdf) - 1)
    return multi.dataset_start_index[task] + local
