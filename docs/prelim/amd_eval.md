# 把评测和续训搬到 AMD（ROCm）机器上

[← 返回初赛总览](README.md) · 代码：[`prelim/eval_amd/`](../../prelim/eval_amd/)、[`prelim/dino/amd/`](../../prelim/dino/amd/)、
[`src/rebuild_videos.py`](../../prelim/src/rebuild_videos.py)、[`scripts/hpc/pull_amd_ckpt.sh`](../../prelim/scripts/hpc/pull_amd_ckpt.sh)

赛事提供了 AMD 算力（一台 8×W7900D 的容器实例，下文称 `<amd-host>`）。我们想用它多评几个 checkpoint、顺便多训一条线。
最后它确实跑完了一次续训和一次评测，但这台机器上的数字和 NVIDIA 上的数字不能直接比，原因主要在仿真环境。

**结论**：AMD 镜像里的 RoboTwin 版本、运动规划器、渲染器都和官方指定的不一样，同一个模型在两边的成绩差很多。
要在 AMD 上得到可比的数字，至少要（1）换回官方要求的 SAPIEN 版本，（2）用 NVIDIA 那次评测的种子列表，不让 AMD 上的专家挑种子。
即便如此，我们也只把 AMD 上的结果当作参考。

## 1. 机器

| 项 | 内容 |
|---|---|
| GPU | 8 × AMD Radeon PRO W7900D（48 GB），ROCm |
| 网络 | **没有任何外网出口**：Hugging Face、hf-mirror、ModelScope 都连不上，所有文件只能从外面推进去 |
| 存储 | 容器自己的盘（`/root`）在实例重建时会清空：有一次实例被重建成新容器，前一天的续训、checkpoint 和传进去的文件全部丢失。持久的只有 `/workspace`（NFS，100 GB）和 `/models` |
| 镜像自带 | `/opt/robotwin-env`（Python 环境）、`/RoboTwin`（见下表）、主办方改过的 lingbot-vla-v2（上游 `951475a` + ROCm / LeRobot 0.6 适配） |

## 2. 仿真环境和官方要求不一致

| 项目 | 官方要求（LingBot 文档） | AMD 镜像 |
|---|---|---|
| RoboTwin commit | `13c3c47` | `266f3aa`，比官方晚 40 个 commit，另有 18 处本地改动，包括 `envs/_base_task.py`、`envs/robot/planner.py`、`envs/robot/robot.py` |
| 运动规划 | CuRobo | MPLib 0.2.1（没有 CuRobo） |
| SAPIEN | 3.0.0b1 | 3.0.3 |
| 推理设置（主办方启动脚本） | FP32，执行长度 50 | BF16，执行长度 25 |

我们最初在 AMD 上的对比（两边回合数不同，AMD 侧的评测设置也还没完全对齐）：同一个权重、同一份 clean-only 归一化统计，在 47 个共同的 clean 任务上，NVIDIA 85.5%（每任务 10 回合），
AMD 67.2%（每任务 7 回合）。大部分任务两边接近（31 个任务差距在 15 个百分点以内），但约 10 个任务在 AMD 上几乎全败，例如：

| 任务 | NVIDIA | AMD |
|---|---|---|
| press_stapler | 10/10 | 0/7 |
| stack_blocks_two | 9/10 | 0/7 |
| stack_blocks_three | 8/10 | 0/7 |
| place_a2b_right | 9/10 | 1/7 |
| place_dual_shoes | 9/10 | 2/7 |
| place_phone_stand | 9/10 | 2/7 |

另一个模型在这几个任务上也一样失败，所以不像是模型的问题。AMD 上约 42% 的 clean 场景专家 5 次都失败，日志里大量
`IK Failed! Cannot find valid solution.`，而在 NVIDIA 环境下 clean 场景的专家几乎都能成功。精度差异一般不会让一个任务从 10/10
变成 0/7，我们判断主要是版本和本地改动造成的。我们把这些差异整理成问题，想请主办方提供基于 `13c3c47` 的环境、说明本地改动、确认官方复测用的环境（是否得到答复没有记录）。

## 3. 我们做的适配

### 3.1 渲染器：换回 SAPIEN 3.0.0b1

- 3.0.0b1 在这台机器上默认根本起不来：创建任何渲染器都报 `invalid PCI string`，即使显式指定设备也一样；3.0.3 可以。
- 能用的组合：**每个进程只让它看到一张卡**，`DRI_PRIME=pci-0000_XX_00_0`（总线号来自 `rocm-smi --showbus`）+
  `MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1`。不设 `DRI_PRIME` 时，不管 `HIP_VISIBLE_DEVICES` 怎么设，所有客户端都在 0 号卡上渲染。
- 共享环境里保留 3.0.3（别人也在用），3.0.0b1 装在单独目录，评测进程通过 `PYTHONPATH` 加载。
- 坑：用 pip 来回切换版本会删掉 `site-packages/sapien.libs`，之后只能从 wheel（PyPI 上 `manylinux_2_28` 的那个）重装恢复。
- OIDN 去噪器只支持 CUDA，所以 AMD 上的光线追踪渲染没有去噪（官方环境有）。这是我们没法消除的差异。

### 3.2 其他环境改动

- `mplib` 的 `planner.py` 有一行和官方 RoboTwin 要求的写法不同（多了 `or collide`），改回官方形式，原文件留了备份。
- 相机近裁剪面：镜像里 `envs/camera/camera.py` 是 0.1，我们的评测要求 0.01，启动时检查（[`run_seedlist_eval.py`](../../prelim/eval_amd/run_seedlist_eval.py)）。
- 推理服务按官方设置跑：FP32、执行长度 50；ROCm 上关掉 `torch.compile`（主办方自己的运行也是关的）。
- 一个误会：主办方客户端里的 `accept_expert_info_on_failure=true`（专家失败也照样评测，不换种子）符合答疑第 13 条，等价于 0 次重试，不是 bug。

### 3.3 按种子列表评测

AMD 上的专家用 MPLib 规划，挑出的种子和 NVIDIA 上不同，而且 42% 的 clean 场景专家都过不了，用它挑种子会让两边比的不是同一批场景。
所以我们改成**直接评测 NVIDIA 上 22k+8k 那次评测实际用过的种子**（[`seeds_q100_s22k8k_8k_20ep.json`](../../prelim/eval_amd/seeds_q100_s22k8k_8k_20ep.json)，每任务 20 个），完全不跑专家规划：

- 生成指令需要的场景信息（哪个物体、用哪只手；50 个任务里都只依赖场景）来自 `play_once`，但事先把 `plan_success` 设为 False，
  每个动作立刻返回，什么都不规划、不执行；失败时才真的跑一次专家取信息。
- NVIDIA 那次里“场景无法搭建”的回合在这里也记为失败。
- 同样有断点续跑，日志格式与 NVIDIA 客户端一致，[`episode_table.py`](../../prelim/src/episode_table.py) 和
  [`compare_paired.py`](../../prelim/src/compare_paired.py) 两边通用。
- 实现：[`patch_xpolicylab_seedlist.py`](../../prelim/eval_amd/patch_xpolicylab_seedlist.py)（打在主办方客户端的副本上）和
  [`run_seedlist_eval.py`](../../prelim/eval_amd/run_seedlist_eval.py)（8 张卡并行调度，每卡一个推理服务 + 客户端；推理服务约 13.4 GB，客户端渲染约 2.7 GB）。

## 4. 往机器里搬东西

| 路径 | 实测速度 |
|---|---|
| 笔记本 → AMD | 约 0.3–0.4 MB/s |
| 集群 → AMD（直连） | 约 86 KB/s |
| 集群 → AMD（经笔记本转发 ssh agent 的 ControlMaster） | 一次测到约 37 MB/s，实例重建后再测约 0.3 MB/s |
| AMD → 集群 | 单流约 1.9 MB/s，4 个并行流约 2.9 MB/s；长连接会慢慢降到约 100 KB/s，所以每个 rsync 流每 10 分钟重启一次（`--append-verify` 续传） |
| AMD → 笔记本 | 约 0.5 MB/s |

一个 `hf_ckpt` 有 24 GB，按这个速度要传好几个小时，所以：

- **数据集不传视频，现场重建**。转换好的 LeRobot 数据集里视频约 2.25 GB，上传太慢；而 AMD 镜像里有官方 clean_50 演示的另一种打包。
  [`src/rebuild_videos.py`](../../prelim/src/rebuild_videos.py) 只传 `meta/` 和 `data/`，在 AMD 上按 lerobot 0.4.2 的编码参数从原始 JPEG 重新编码视频。
  用同一版 PyAV 时与原文件逐字节相同（在集群上验证过）；编码器版本不同时只有编码舍入差异，脚本用 `--verify` 量 PSNR、帧数和每帧时间戳。
- **checkpoint 由集群直接从 AMD 拉**：笔记本只做协调（[`scripts/amd_ckpt_to_cluster.sh`](../../prelim/scripts/amd_ckpt_to_cluster.sh)），
  集群登录节点上的 [`pull_amd_ckpt.sh`](../../prelim/scripts/hpc/pull_amd_ckpt.sh) 并行拉每个文件、逐个对 SHA-256，
  通过后才写 `.ready`，集群上的评测池看到它才开始评测这个 checkpoint。
- **持久化**：续训时另起一个进程，每 5 分钟把最新的完整 checkpoint 镜像到持久卷（[`dino/amd/mirror_ckpt.sh`](../../prelim/dino/amd/mirror_ckpt.sh)）。
  卷只有 100 GB，一个完整 checkpoint 约 75 GB，所以先删旧的、写到 `.partial` 再改名；空间不够时只存 `hf_ckpt` + `extra_state`。

## 5. 在 AMD 上续训

从 22k+8k 出发续训（[`dino/amd/`](../../prelim/dino/amd/)，不带 DINO）：

| 项 | 内容 |
|---|---|
| 代码 | 主办方镜像里的 lingbot-vla-v2 + 我们的方法补丁 + [`dino/patch_amd.py`](../../prelim/dino/patch_amd.py)：主办方版本用 LeRobot 0.6 读数据，`__getitem__` 提前返回，我们追加在函数末尾的 seen 指令替换永远执行不到，所以在提前返回的路径上再加一遍 |
| 配方 | `aug_strong_v3.json`，温度采样 temp:2，静止帧降权，seen 指令替换，状态噪声 0.01，视觉塔 lr ×0.1；lr 5e-6 → 2e-6，3% warmup；全局 128 = 每卡 16 × 8 卡 |
| 深度损失 | 降到 1e-8：AMD 上的深度教师给出的损失约 4，我的算力上约 0.5，判断是这台机器上的数值问题，干脆关掉 |
| 步数 | 0 → 3,000（日程 3,000）；之后以 seed 43 从第 3,000 步的权重接着训向 4,000，在 3,815 步时停掉。评测用第 3,000 步 |
| 已知问题 | 和我的算力一样有“各 rank 样本相同”的 bug（[hpc.md 5.4](hpc.md#54-所有-rank-拿到相同的样本)）：第 3,000 步的 8 个 `extra_state` 里基础种子完全相同，每步实际只有 16 个不同样本 × 8 份 |

评测结果：

| 在哪评 | 回合 × 任务 | clean | randomized |
|---|---|---|---|
| AMD（按 22k+8k 的种子列表） | 7 × 49 | 286/343 = 83.4% | 204/343 = 59.5% |
| 我的算力（标准协议） | 7 × 34（clean，记录时未跑完）/ 7 × 50（randomized） | 194/238 = 81.5% | 215/350 = 61.4% |
| 对照：22k+8k（我的算力） | 7 × 50 | 296/350 = 84.6% | 228/350 = 65.1% |

AMD 上没有 22k+8k 自己的对照（AMD 上那份 22k+8k 的拷贝在评测前被删了），所以 AMD 那一行没法和任何东西直接比。
在我的算力上，+3k 前 7 回合的 randomized 比 22k+8k 低（215 对 228），没有带来提升。

## 6. 教训

1. **先核对仿真环境的版本**，再在新平台上评测。版本不同的仿真器给出的是另一套任务。
2. 能控制变量的地方都控制：固定种子列表、相同推理精度和执行长度、相同归一化统计（脚本启动时校验 SHA-256）。
3. 没有外网、容器会被重建的机器上，重要的东西要第一时间放进持久卷，并且假设任何时候都可能丢。
4. 跨平台比较一定要在同一平台上放一个对照模型。我们丢了 AMD 上的对照，AMD 评测的结果就只能当参考。
