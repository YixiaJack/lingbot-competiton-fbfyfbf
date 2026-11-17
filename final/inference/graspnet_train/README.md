# graspnet: per-arm "is a tool handle held" from the wrist camera (task 3)

Built 2026-11-16 ~03:00 at the user's request, for the 5080 task-3 harness (patch ~/LingBotRoboTwin/robotharness_patches/patch_graspnet.py).
- Model `graspnet.torchscript` (md5 b2685b4d…, 205k params, 5-layer CNN). Input: one wrist frame resized to 128x96 RGB, float in 0..1. Output: logit; sigmoid = P(held).
- Data: `extract.py`, labels from the gripper trace in 0-1 units:
  - held = after an opening, steady in 0.17–0.40;
  - not held = ≥0.5, or ≤0.08 before the first opening, or ≤0.08 after a release (only when it held something before that last opening);
  - everything else is ambiguous and not used.
  - Train: official task-3 demos (216, stride 10, 33k crops) + 105a2d09 rollouts 93–171. Test: 105a2d09 rollouts 172–190 (held out).
- `train.py`: 10 epochs on MPS, about 2 min. Held-out accuracy 99.8% (3211 crops: 5 false "held", 1 miss).
  - Also catches holds the readings cannot show: ep173 brush at 0.07, ep174 dustpan rim grasp at 0.05.
  - Outputs ~1.0 on black or noise frames, so the harness ignores blank frames (mean < 12 or std < 8).
- 5080 CPU latency: 1.7 ms median for both arms (first call 41 ms).
