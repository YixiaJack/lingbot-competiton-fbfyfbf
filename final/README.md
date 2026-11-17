# final/：决赛代码

决赛 48 小时里实际用过的代码快照。各部分的设计和经过写在 [docs/final/](../docs/final/README.md)，这里只说明目录。

| 目录 | 内容 | 文档 |
|---|---|---|
| [training/](training/) | 训练框架补丁 p00–p10（含测试）、加权列表生成 `mk_v4c.py`、自动接入新批次 `auto_multi.sh`、存档发布 `ckpt_guard.py`、提交打包、两次提交时写的训练说明 | [training.md](../docs/final/training.md) |
| [inference/](inference/) | 第三轮推理机上实际运行的模块：`rtc/` 异步 + RTC，`yolocue/` YOLO 框提示，`robotharness/` 任务三子任务调度（含抓取检测网络和逐步改动的补丁），`actsmooth/` 动作平滑，`dagger/` 接管记录与交还，`upstream_patches/` 对官方代码的 diff，`graspnet_train/` 抓取检测网络的训练 | [inference.md](../docs/final/inference.md)、[task3_harness.md](../docs/final/task3_harness.md) |
| [data/](data/) | 逐批数据处理脚本和人工核对结论（`batches/`）、第一天的自动标注和带框数据集生成（`autolabel/`）、YOLO 各版本的标注流程（`yolo/`） | [data.md](../docs/final/data.md)、[yolo.md](../docs/final/yolo.md) |
| [tools/](tools/) | 现场看板和训练看板网站（`web_board/`）、训练进度采集（`train_board/`）、harness 实时视图（`harness_view/`）、堡垒机后开发机的操作工具（`devmachine/`）、赛前准备的检测器标注工具包（`label_kit/`） | [tooling.md](../docs/final/tooling.md) |

说明：

- 代码运行在主办方的训练平台和推理服务里，这两样不公开，所以多数脚本不能直接运行，作为参考实现发布。带离线测试的部分见首页"能跑起来吗"一节。
- 脚本里的主机、用户名和本地路径已经换成占位符（如 `<inference-host>`、`<dev-hostname>`）。
- 模块目录里的 README 是比赛中边做边写的记录，保留了当时的时间点和测试数字。
- 不包含任何数据集、模型权重（抓取检测网络的 0.8 MB TorchScript 除外）和 API key。
