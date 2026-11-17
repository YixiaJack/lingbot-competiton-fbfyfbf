# 在 HPC 集群上复现开发机的训练（11/14）

> 状态：**完成。** 换数据重启的路径在集群上验证通过，决赛中每次换数据重启都按这个做法；每步时间的拆分说明数据增强不增加耗时、teacher 约占 14%、优化器约占 22%。

## 要回答的问题

决赛开发机（8 × RTX PRO 5000）一直在训练，8 卡全占。重启前要用的改动先在别处验证，不占开发机：

1. **重启路径**：从一个存档换成新的数据列表接着训，步数、学习率、loss 是否连续？只保留导出权重的“瘦身”存档能不能被官方推理代码正常加载？
2. **怎么加速**：每步时间花在哪？打开图像增强会不会变慢？teacher（深度 / 视频对齐用的教师模型前向）和优化器各占多少？

## 为什么做

- 现场每来一批新数据就要换列表重启。重启方式错了（比如把旧数据的读取进度套到新数据上，或者优化器状态没恢复）不会报错，却会悄悄改变训练。
- 开发机磁盘有限：保留续训状态的存档每个 61 GB，只留导出权重则 12 GB，需要知道能删什么。
- 官方给了几条加速建议，要先量出瓶颈再决定动哪里。

## 设置

| 项 | 开发机 | 集群 | 一致？ |
|---|---|---|---|
| 代码 | 官方 11/10 快照 + 我们的补丁 p00–p06（opt 分支 3adb6b2） | 同一个 git bundle（SHA-256 核对） | 一致；另加 `LB_VIT_FORCE_SDPA=1` 开关，让 H200 上的 ViT 也走 sdpa（开发机 sm_120 只能走 sdpa） |
| 环境 | NGC 镜像，Python 3.12、torch 2.8.0+cu128、transformers 4.57.3、flash-attn 2.8.3、lerobot 0.4.2、torchcodec 0.6.0 | 初赛的容器环境，同版本 | 一致；numpy 1.26.4 / pyarrow 19.0.1 另装后放到 PYTHONPATH 最前（集群自带 2.2.6 / 21.0） |
| 底座 + teacher | 平台的系统模型目录 | 官方 6B 底座 + 从开发机取的 `moge2-vitb-normal.pt` | 文件大小全同，config / index / teacher 配置 SHA-256 相同 |
| 数据 | `public_t12`（官方任务 1、2 的 431 条，225,984 帧）、`public_t3_sub`（官方任务 3 的 216 条切成 864 段，179,719 帧）、`public_t1_box`（带框的任务 1） | 用同一个 `derive_dataset.py`（[sweep_harness](../sweep_harness/README.md)）从官方数据重建前两个 | 条数、帧数完全相同；带框集没建（每步计算量一样） |
| 配置 | 开发机 v2 配置：micro batch 16 × 梯度累积 2 | 同一份，只换路径；2 卡，所以全局 batch 64 | 每卡计算量一样 |
| 编译告警 | `CopySlices` / `aten.nonzero` 各 16 次 | 同样 16 次 | 一致 |

**硬件不同**：开发机是 RTX PRO 5000（sm_120），PCIe、没有 NVLink、8 卡分在两个 CPU 上；集群上没有 sm_120 的卡，用的是 2 × H200（有 NVLink）。所以这里的秒数不能当开发机的速度，只看比例；通信占比在这里测不出来。

## 结果

### 1. 重启路径（R1 → R2 → R3）：全部通过

- **R1**：我们的白名单增强 + 官方颜色增强，数据 A（`public_t12`），训 10 步，第 5、10 步存档，保留续训状态（`FORGE_CKPT_KEEP_DCP=1`）。每个存档 61 GB（导出权重 hf_ckpt 12 + 模型 24 + 优化器 26），和开发机一样。
- **R2**：从 R1 的第 10 步续训，**换成数据 A+B**（`public_t12` + `public_t3_sub`）：`--train.load_checkpoint_path …/global_step_10 --train.resume_dataloader_state false`，新的输出目录，不保留续训状态。
  - 日志显示分布式存档加载成功，步数从 11 接着走到 16，学习率保持 5e-5，loss 和 R1 末尾连续（R1 最后几步 1.09 / 1.47 / 1.06，R2 1.29 / 1.37 / 0.72 / 0.93 / 1.11 / 1.04）。
  - 每轮步数按新数据重算（3531 → 6339），不报错。
  - 存档后自动瘦身：第 16 步只剩 hf_ckpt 12 GB，里面带推理需要的 `lingbotvla_cli.yaml`、`robot_config.yaml`、`norm_states.json`。
- **R3**：用开源的 `deploy/lingbot_vla_v2_policy.py`（`strict=True` 加载 3 个分片）对瘦身后的 hf_ckpt 跑官方 `scripts/open_loop_eval.py`，两条轨迹正常推理（只训了 16 步，MSE 数值没有意义）。
- **`ckpt_prune.sh`**：只给最新的存档保留模型和优化器状态；旧存档在 hf_ckpt 完整、且有更新的完整存档时才删续训部分。R1 上实测：第 5 步 61 GB → 12 GB，第 10 步保留 61 GB。

### 2. 每步时间拆分（2 × H200，第 8–25 步的中位数，同步计时）

| 情况 | s/步 | teacher 占比 |
|---|---|---|
| T0 基准（开发机 v2 的全部设置） | 2.678 | 0.372 s = 13.9% |
| T1 + 官方自带的颜色增强 | 2.641（−1.3%） | 14.2% |
| T2 + 我们的白名单增强 | 2.640（−1.4%） | 14.2% |

- **增强不增加时间**（两种都在误差内）；数据加载不是瓶颈（这里每卡 12 个 CPU，开发机每卡 32 个）。
- **teacher 约 14%**：官方建议的“把 teacher 目标离线缓存”最多省这么多，还要 200–600 GB 磁盘。开发机日志里的 `Depth_Forward_Time 0.257s` 没有同步，偏小。
- **profiler**（T0 第 18–19 步，`T0_profile.txt`）：GPU 忙约 2.27 s/步；`DistributedMuon.step`（优化器）约 0.60 s/步，占墙钟 22%，比 teacher 还大，而且每步只做一次、和 micro batch 数无关；NCCL all-gather + reduce-scatter 约 0.45 s/步（NVLink 上）。
- **T3（去掉 teacher 对齐）跑不起来**：官方底座带对齐头的权重，严格加载会报 `Unexpected key 'model.current_video_align_head…'`。teacher 的时间已经由同步计时直接量到，不需要了。
- 同样每卡 micro 16 × 累积 2：H200 2.68 s/步，开发机 8.7 s/步，约 3.2 倍。

## 结论与状态

完成。给开发机的建议：

1. **磁盘**：保留续训状态（能续训），每次存档后清理旧存档的模型和优化器状态。
2. **换数据重启**：`--train.load_checkpoint_path <存档>` + 新的输出目录 + `--train.resume_dataloader_state false`（默认 true，会把旧数据的读取进度套到新数据上）；不要用同一个输出目录靠自动续训。
3. **归一化统计**：续训时换成新数据重算的统计，动作尺度会突变；要么沿用旧统计，要么接受这个跳变。
4. **先量再改**：在开发机上跑 20 步带 profiler（`LB_TIME_BREAKDOWN=1 LB_PROFILE_STEPS=18,19`，约 5 分钟），看 PCIe 上 NCCL 和优化器的真实占比，再决定要不要动通信或优化器。

## 在决赛中用上了什么

- **换数据重启的做法**：决赛中每次换列表重启都恢复模型、优化器、学习率调度和步数，但不恢复 dataloader 位置（`--train.resume_dataloader_state false`），见 [docs/final/training.md](../../docs/final/training.md)。
- **磁盘保护**：`scripts/ckpt_guard.py` 从 11/14 17:28 起在开发机上每 10 分钟跑一次：只在存档所在磁盘剩余低于 150 GB 时，从最旧的开始删续训部分；永远不动每个运行最新的存档、没有完整 hf_ckpt 的存档、还在写的运行和 hf_ckpt 本身。它后来扩展成决赛版（把新存档同步到云存储），在 [final/training/tools/](../../final/training/tools/)。

## 文件说明

| 文件 | 内容 |
|---|---|
| `scripts/mkcfg_hpc.py` | 把开发机的训练配置换成集群路径，支持 `key=value` 覆盖（`DELETE` 删键）；开发机家目录用环境变量 `DEV_HOME` |
| `scripts/run_case.sh` | 在容器里起一次训练，记录用到的 `LB_* / LINGBOT_* / FORGE_*` 环境变量 |
| `scripts/job.sbatch` | T0–T3（时间拆分）和 R1–R2（重启路径）的作业；同时排几个备选，第一个开始的取消其余 |
| `scripts/r3.sbatch` | R3：用开源推理代码对瘦身存档做开环评测 |
| `scripts/summarize.py` | 从日志里汇总每步时间、teacher 占比和重启证据 |
| `scripts/ckpt_prune.sh` | 只给最新存档保留续训状态 |
| `scripts/ckpt_guard.py` | 开发机上按磁盘阈值触发的存档清理 |
| `scripts/devfetch.py` | 经堡垒机的交互式终端从开发机取文件（base64 + SHA-256 校验，约 2.5 MB/s） |
| `scripts/tq.sh` | 向已登录的 tmux 会话发命令并取回输出 |
| `T0_profile.txt` | T0 第 18–19 步的 PyTorch profiler 表 |

代码里的路径是集群上的 `$SCRATCH/lingbot/...`。
