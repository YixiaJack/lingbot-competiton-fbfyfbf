# Third-party software / 第三方软件

This repository contains our own code and documentation (Apache-2.0, see [LICENSE](LICENSE)). It does **not** vendor the upstream projects below; where we changed them, we ship patch scripts or diffs that apply to the stated upstream version.

本仓库只包含我们自己写的代码和文档（Apache-2.0）。下面的上游项目没有整体拷进来；改动过的部分以补丁脚本或 diff 的形式给出。

| Project | Used for | License | In this repo |
|---|---|---|---|
| [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) (Robbyant) | Base model and training framework (prelim: commit `be969b8`) | Apache-2.0 | Patch scripts `final/training/patches/`, `prelim/patches/`; diffs `final/inference/upstream_patches/` |
| Qwen3-VL (inside LingBot-VLA 2.0) | Vision-language backbone | See the Qwen license | — |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) | Prelim simulation benchmark and pre-final simulated scenes | See its repository | Eval client patches under `prelim/` |
| Competition inference service (provided by the organizers) | Runs the policy on the robot | Not ours, not included | Our modules hook into it; `final/inference/upstream_patches/` contains only our diffs |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Training the object detectors | AGPL-3.0 | Training scripts only; no weights are distributed |
| LocateAnything-3B | Open-vocabulary auto-labelling of the first YOLO dataset | See its model card | Labelling scripts only |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Prelim feature-fusion experiment | Apache-2.0 | Patch script under `prelim/dino/` |
| DeepSeek API | LLM that judges task-3 progress in the harness | Commercial API | Client code only; no API key in this repo |

No datasets, checkpoints, robot recordings or API keys are included.

仓库里不包含任何数据集、模型权重、机器人录像或 API key。
