# 初赛代码（LingBot-VLA 2.0 × RoboTwin 2.0，只用 clean 数据）

初赛的完整代码快照：数据转换、训练补丁、Slurm 抢占队列上的训练和评测脚本、DINOv2 融合实验、AMD 评测移植。
方案、数字和教训写在 [docs/prelim/](../docs/prelim/README.md)，本页只说明代码怎么组织。

上游版本：[lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2) @ `be969b8`，
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) @ `13c3c47`（LingBot 文档指定的版本）。我们的改动都以补丁形式提供，不包含上游代码。

| 路径 | 内容 | 文档 |
|---|---|---|
| `submission/` | **最终提交包的代码**（22k+8k 模型，官方复测 clean 82.22 / randomized 57.26），自带 `prepare_data.sh`、两段式 `train.sh`、`eval.sh` 和复现说明。想复现，从这里开始 | [README §7](../docs/prelim/README.md#7-最终提交和官方成绩) |
| `submission_v3uni5k/` | 22k+8k+5k 备选包（没有提交）与提交包不同的几个文件：三段式 `train.sh`、第 3 段的训练器补丁和增强配置 | [README §7](../docs/prelim/README.md#7-最终提交和官方成绩) |
| `train.sh` / `eval.sh` | 单机版：两段训练出 22k+8k 模型；评测 50 个任务 × 一个 setting。`train_main40k.sh` 是最早的 main 40k 单段训练 | [README §3](../docs/prelim/README.md#3-训练路线) |
| `src/` | 数据转换、增广（`lingbot_aug.py`）、采样（`lingbot_sampling.py`）、上游补丁（`patch_upstream.py`）、评测客户端补丁、结果汇总、配对比较 | [README §4–5](../docs/prelim/README.md#4-数据增强) |
| `configs/` | 训练配置、clean-only 归一化统计、各版增广参数、各阶段实际生效的 `*_lingbotvla_cli.yaml` | |
| `patches/` | 相对上游 `be969b8` 的 diff：方法补丁、DINO 版、第 3 段训练器补丁 | |
| `scripts/hpc/` | 我的算力上的 Slurm + Apptainer 作业：训练 handoff 组、评测 worker 池、W&B 镜像、HF 上传 | [hpc.md](../docs/prelim/hpc.md) |
| `scripts/*.sh` | `setup_lingbot.sh` 拉取上游并打补丁；`amd_*.sh`、`gpu_cap_watch.sh` 是在笔记本上跑的协调脚本（AMD↔集群传 checkpoint、GPU 上限监视） | [amd_eval.md](../docs/prelim/amd_eval.md) |
| `dino/` | DINOv2 融合：模块、补丁、初始 checkpoint、CPU/GPU 等价性测试；`dino/amd/` 为 AMD 上的续训 | [dino.md](../docs/prelim/dino.md) |
| `eval_amd/` | AMD（ROCm）上按种子列表评测 | [amd_eval.md](../docs/prelim/amd_eval.md) |
| `results/` | 评测汇总：逐任务、逐回合 CSV 和说明；提交包和备选包的 `results.json`（每任务前 10 回合） | [README §6](../docs/prelim/README.md#6-结果) |

## 怎么跑（概要）

```bash
# 1. 上游代码 + 补丁（第 3 段训练另加 TRAINER_PATCH=1）
DEST=/path/to/lingbot-vla-v2 bash submission/src/setup_lingbot.sh
# 2. 下载基座、RoboTwin 资源和 50 个任务的 clean 数据，转成 LeRobot 视频数据集
CKPT_DIR=... DATA_DIR=... ROBOTWIN_DIR=... bash submission/src/prepare_data.sh
# 3. 训练：STAGE=1|2|3|all（全局 batch 128，卡数不同时用梯度累积补齐）
cd submission && LINGBOT_DIR=... CKPT_DIR=... DATA_DIR=... RUNS_DIR=... STAGE=all bash train.sh
# 4. 评测：clean 用 seen 指令，randomized 用 unseen 指令，每回合最多 5 次专家尝试
CKPT=<run>/checkpoints/global_step_N/hf_ckpt LINGBOT_DIR=... ROBOTWIN_DIR=... CKPT_DIR=... bash eval.sh
```

环境：conda `lingbotvla`（Python 3.12，torch 2.8.0+cu128）和 `RoboTwin`（Python 3.10，sapien 3.0.0b1，CuRobo v0.7.8），
Ubuntu 24.04 镜像。评测我们只在 A100 / H100 / H200 / L40S 上跑过（RTX PRO 6000 Blackwell 上 CuRobo 起不来）。

## 脱敏说明

- 配置里的集群路径写成字面量 `$SCRATCH/lingbot/...`，`train.sh` 会改写成本机路径；直接用 `scripts/hpc/` 的 sbatch 脚本时，先把它换成你的路径（例如 `envsubst`）。
- Slurm 账号、AMD 主机地址和端口、Hugging Face 仓库名、队伍名和 team_id 都换成了环境变量或占位符（`<amd-host>`、`<hf-user>`、`<team_id>` 等）。
- 代码注释里保留了当时的日期和排查记录，作业号已删去。

## 未包含

- 模型权重和数据集（checkpoint 每个约 73 GB，其中 `hf_ckpt` 24 GB；转换后的数据集 50 个任务）。
- 评测原始日志（`results/*_说明.md` 提到的 `原始日志/` 没有放进来）、训练日志、W&B 记录。
- 工作仓库 `docs/reproduction_report.md`（早期 main 40k 版本，已被 `submission/` 和 `submission_v3uni5k/` 里的复现说明取代）。
- 没有超过 1 MB 的代码文件，所以没有因大小而跳过的文件。
