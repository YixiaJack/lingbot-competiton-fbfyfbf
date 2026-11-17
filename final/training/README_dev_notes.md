# 开发机上的真机训练（11/14）

开发机 8× RTX PRO 5000 72GB（Blackwell，sm_120）。工作目录 `~/lingbot_train`；`/opt/lingbot-vla-v2` 只读，代码用副本：`code`（master）和 worktree `code_opt`（opt 分支，下面的优化全部用环境变量开关）。连机器和传文件见 `tools/devrun_tmux.sh`、`tools/devput.sh`（堡垒机菜单只能交互，用 tmux 会话驱动）。

## 现在在跑：v2（13:52 起）

| 项 | 内容 |
|---|---|
| 数据 | `public_t12`：官方任务 1、2 共 431 条整段示范，原指令<br>`public_t3_sub`：官方任务 3 共 216 条切成 864 个子 episode，每段一条固定指令（[sweep_harness](../../experiments/sweep_harness/README.md) 的 4 段）<br>`public_t1_box`：任务 1 中 157 条，头顶相机画 boxgrasp 框（下节） |
| 我们采的数据 | 还没出现在开发机（`/mnt/forge/datasets/private` 为空），这版没有 |
| 超参 | 官方 `real_robot.yaml`：Muon、lr 5e-5 恒定、global batch 256；显存只有 71 GB，micro batch 16 × 梯度累积 2 |
| 代码 | `code_opt` 3adb6b2，`LB_VIT_BATCHED_SDPA=1 LB_GRID_CACHE=1 LB_FSDP_VISION=1` |
| 速度 | 约 8.7 s/步；每 500 步存一次（保留 DCP，可续训） |
| 产物 | `runs/v2/`：train.log、checkpoints、code_commit.txt、env.txt、norm_stats.json、v2.txt |

v1（同样的数据但没有带框集）只跑到 97 步就停了，换成 v2 从头训。

## 任务 1 的 YOLO 框（和 real_detector 实验同一种提示）

- 检测器：[autolabel](../data/autolabel) 的 YOLO v1（LocateAnything 第一轮标注训练的 yolo11s，推理机 `~/locany/yolo/runs/v1/weights/best.pt`），用它对每条示范第 0 帧的检测 `dets_first.json`。
- 选框（`autolabel/t1_boxes.py`）：规则解析指令（5 个模板）→ 目标类 + 盘子颜色；目标置信度 ≥ 0.5 且领先同类第二框 0.15，盘子取该颜色里离目标最近的。缺任何一个框就不给提示。215 条任务 1 里 157 条有框（52 条目标没检出，6 条盘子没检出）；能用差分真值核对的 144 条里白框 143 条框对。
- 画法（`autolabel/derive_box_dataset.py`）：白框 = 目标，橙框 = 目的地，线宽 5 像素（640×480 缩到模型输入后仍可见；仿真时 320×240 用 2 像素）。**boxgrasp**：夹爪合上后去掉白框，橙框保留。抓取时刻用遥操作的夹爪**指令**判断（先超过 0.5，之后第一次低于 0.2）；夹爪读数不能用：大物体（如南瓜）合上时读数只从 0.90 降到 0.83。
- 只重新编码头顶相机（crf 17，16 个文件），腕部相机直接引用原视频。和不带框的同一批示范一起训，就是之前的 50% 带框 / 50% 纯文字混训；任务 1 在训练里的比重因此变大。
- **部署时要做同样的事**：推理端对第 0 帧跑 YOLO、按同样规则选框和画框，按 VLA 输出的夹爪指令在线判断何时去掉白框；没把握时不画框。

## 跑起来前修掉的问题

1. LeRobot 初始化时逐个 stat 视频文件，开发机 `/mnt/forge` 是 ossfs 挂载，8 个进程一起卡死：视频拷到本地盘再软链。
2. 官方 `agilex_cobot_magic.yaml` 动作 `subtract_state: true` 但没写 `relative_type`，代码默认按四元数处理，每个样本断言失败：在我们的机器人配置里加 `relative_type: joint`（随权重一起打包）。
3. 镜像里 flash-attn 2.8.3 没有 sm_120 kernel：ViT 注意力在 sm_120 上改走 sdpa（patches/p01）。
4. micro batch 32 显存不够：16 × 累积 2，`gradient_accumulation_steps` 要显式写。

## 四项官方优化建议的实测（20 步，同配置同种子）

| 配置 | s/步 | 前 14 步 loss 和官方最大差 |
|---|---|---|
| 官方代码（+ sdpa 回退） | 9.826 | — |
| + ViT 等长图片批量 SDPA + 视觉位置元数据缓存（建议 2，p00/p02） | 9.614 | 0.003 |
| + ViT blocks / deepstack merger / ViT 从根分组拆出（建议 4，p03） | **8.674** | 0.003 |
| + 显式 FSDP 预取 + 编译的 RMSNorm/SwiGLU（建议 4 后半、建议 3，p04/p05） | 8.678 | 0.003 |

- 建议 2 的元数据缓存在 CPU 上和原版输出、梯度逐位相同（patches/test_vit_cpu.py）。
- 预取和 RMSNorm/SwiGLU 编译不加速（整个模型已 torch.compile），没开。Q/K/V 准备的 Triton 融合没移植，理由同上。
- 建议 1（teacher 目标离线缓存）没做：深度 + 视频目标全量约 620 GB，本地盘剩 386 GB；只缓存深度约 207 GB，teacher 占比还没量。

## 训练时 RTC（p10，11/14 夜，v4 从总第 3000 步起）

起因：11/14 评测时推理是同步的，每执行 25 步（0.83 秒）就要停约 0.47 秒等推理，机器人有 34–36% 的时间在原地不动。改成异步 + RTC 要训练和推理两边一起改。这里先改训练端，推理端等训出来的权重够用了再上。

- 方法：Black 等 2025.12 的训练时 RTC（arXiv 2512.05964）。每个样本随机给前 d 个真值动作作为干净前缀，时间设为 0。本代码 x_t = t·noise + (1−t)·actions，t=0 是干净动作，和论文约定相反。loss 只算前缀之后的 token。d 以 30% 概率取 0（保住同步、无前缀路径），其余在 1..24 里均匀取。推理时每段请求在剩 18 步时发出，前缀长度就是 18。
- 改动（[patches/p10_rtc.py](patches/p10_rtc.py)，只改 modeling_lingbot_vla.py 和 modeling_lingbot_vla_v2.py）：
  - `embed_suffix` 支持每个 token 一个流时间（[B, T]）；
  - `AdaRMSNorm` / `FixAdaRMSNorm` 接受逐 token 的条件 [B, 1+T, D]，状态 token 用噪声 token 的时间；
  - 训练前向加前缀和 loss 掩码；
  - `sample_actions(prefix_actions=, prefix_len=)` 每个欧拉步把前缀写回去。
  - 不加参数，优化器状态和权重两个方向都兼容；默认关闭，`LB_RTC_MAX_DELAY > 0` 才打开。
- 测试（[patches/test_p10.py](patches/test_p10.py)，CPU、无权重）：
  - 两种 AdaRMSNorm 在逐 token 条件都相同时与原版逐位一致；
  - embed_suffix 时间全相同时与原版一致，有前缀时只改变前缀 token；
  - 前缀长度的分布符合设定，掩码后的 loss 均值等于前缀之后部分的均值；
  - 训练前向在 train 模式下有前缀、eval 模式下走原路径；
  - `sample_actions` 前缀为 0 时与原循环逐位一致，有前缀时前缀原样保留。
- 切换（[tools/v4r_switch.sh](tools/v4r_switch.sh)）：
  - 等 global_step_1000 的 DCP 和 hf_ckpt 都写完、日志出现第 1001 步，然后停掉 v4；
  - 用 code_v4r = code_v4s + p10 完整续训（模型、优化器、学习率、数据加载、步数），配置、列表、环境变量和上次启动一样，另加 `LB_RTC_MAX_DELAY=24 LB_RTC_P0=0.3`；
  - 新进程跑不到第 1002 步就退回 code_v4s 不带 RTC 续训。
