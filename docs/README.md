# 文档索引

文档用中文写。英文概览见仓库首页 [README.md](../README.md)。

## 总览

- [competition.md](competition.md)：比赛简介（只含公开信息）
- [timeline.md](timeline.md)：从初赛到决赛结束的时间线
- [references.md](references.md)：参考资料

## 初赛：RoboTwin 2.0 仿真

- [prelim/README.md](prelim/README.md)：赛题、方案、训练路线、数据增强、评测协议、结果和经验

## 决赛前

- [pre_final/plan.md](pre_final/plan.md)：决赛前的技术方案（合规边界、模型代码事实、显存预算、画框方案、训练方案）
- [pre_final/decisions.md](pre_final/decisions.md)：决赛前的决策记录（匿名整理自队内讨论）
- [pre_final/experiments.md](pre_final/experiments.md)：决赛前的仿真实验总览

## 决赛：48 小时真机赛

- [final/README.md](final/README.md)：系统总览和关键设计
- [final/data.md](final/data.md)：数据
- [final/training.md](final/training.md)：训练
- [final/inference.md](final/inference.md)：推理端
- [final/yolo.md](final/yolo.md)：YOLO 目标检测
- [final/task3_harness.md](final/task3_harness.md)：任务三的子任务调度
- [final/tooling.md](final/tooling.md)：工具与看板
- [final/results.md](final/results.md)：结果
- [final/postmortem.md](final/postmortem.md)：复盘

## 代码旁的说明

不少模块在代码目录里还有自己的 README，写的是更细的实现和测试记录，例如 [final/inference/rtc/](../final/inference/rtc/README.md)、[final/inference/yolocue/](../final/inference/yolocue/README.md)、[final/inference/dagger/](../final/inference/dagger/README.md)、[final/training/submissions/](../final/training/submissions/README_third_15200.md)，以及 [experiments/](../experiments/) 下各个实验的 README。这些是比赛中边做边写的记录，保留了当时的时间点和数字。
