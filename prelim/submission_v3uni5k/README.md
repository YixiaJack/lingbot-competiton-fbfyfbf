> **备选包，最终没有提交。** 这是 22k+8k 之后再训 5k 步（三段式）的版本，代码和实际提交的 [`../submission/`](../submission/README.md) 只差三段式 `train.sh`、`eval.sh` 的默认 checkpoint、`setup_lingbot.sh` 的训练器补丁开关、`configs/aug_strong_v3.json` 和 `src/patches/trainer_stage3.diff`；其余文件（`src/`、`configs/` 的其他部分）见 `../submission/`。

# LingBot-VLA 2.0 × RoboTwin 2.0（仅 clean 数据）复现代码

## 本次提交

| 项 | 内容 |
|---|---|
| 提交模型 | `v3uni_noon / global_step_5000`（`03_模型权重/checkpoint/v3uni_noon/checkpoints/global_step_5000/hf_ckpt`） |
| 评测结果 | 每个任务每个 setting 测 **10 次**：demo_clean 407/500 = **81.4%**，demo_randomized 335/500 = **67.0%** |
| 训练数据 | 只用 RoboTwin 2.0 Aloha-AgileX 50 个任务的 `aloha-agilex_clean_50`（每任务 50 条），**未使用任何 randomized 数据** |
| 归一化统计 | 只在上述 clean 数据上重新计算（`configs/norm_stats/robotwin_clean.json`），未使用官方基于 clean + randomized 计算的 `assets/norm_stats/robotwin.json` |
| 代码基础 | [lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2) @ `be969b8` + 本目录的补丁；[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) @ `13c3c47`（LingBot 文档指定版本） |

## 目录结构

```text
README.md
train.sh                        一键训练：STAGE=1|2|3|all，三段的超参数都内置（见「提交模型的训练过程」）
eval.sh                         一键评测：加载提交的 checkpoint，跑 demo_clean + demo_randomized，输出 results.json
configs/
  robotwin_clean.yaml           训练配置（含模型配置），由官方 configs/vla/robotwin/robotwin.yaml 改动数据路径与 batch 得到
  robot_configs/robotwin_clean.yaml   robot config，与官方 robotwin.yaml 相同，只把 norm_stats 指向 clean-only 统计
  norm_stats/robotwin_clean.json      clean-only 归一化统计（SHA-256 92b37257234b37c8642894997847da496c6e6bb550c2968eced4439c8b6bed31）
  robotwin_clean.txt            50 个 LeRobot 数据集列表（data.data_name: multi）
  tasks50.json                  50 个任务名（顺序与官方结果模板一致）
  aug_whitelist.json            第 1 段的图像增广配置
  aug_strong.json               第 2 段的图像增广配置
  aug_strong_v3.json            第 3 段的图像增广配置
  nvidia_icd.json               容器内无 Vulkan ICD 时给 SAPIEN 渲染用（export VK_ICD_FILENAMES=$PWD/configs/nvidia_icd.json）
src/
  setup_lingbot.sh              拉取官方 lingbot-vla-v2 @ be969b8 并打上本目录的补丁
  patch_upstream.py             方法补丁（环境变量开关），结果等于 patches/method.diff
  patches/method.diff           方法补丁的 diff（44 行新增，5 个文件）
  patches/trainer_stage3.diff   第 3 段训练实际使用的训练器补丁（2 个文件）
  lingbot_aug.py                训练期白名单图像增广
  lingbot_sampling.py           任务均匀采样 + 帧权重采样
  prepare_data.sh               下载模型、RoboTwin、clean 数据，并转换为 LeRobot 数据集
  convert_robotwin_to_lerobot.py  RoboTwin 原始 hdf5 → LeRobot v3.0（视频）
  make_seen_pool.py             每条轨迹的 seen 指令池（meta/seen_instructions.json）
  make_frame_weights.py         逐帧采样权重（meta/frame_weights.npy）
  eval_one_task.sh              单任务评测：启动推理服务 + RoboTwin 评测客户端（与官方启动脚本参数一致）
  patch_eval_client.py          评测客户端补丁（见「对官方代码的改动」）
  stage_ckpt.py                 评测前把 checkpoint 训练配置里的路径本地化，并校验归一化统计的 SHA-256
  make_results_first_n.py       汇总每个 (setting, 任务) 的前 N 个 episode → results.json（官方模板格式）
```

## 环境与依赖安装

| conda 环境 | 用途 | 内容 | 安装 |
|---|---|---|---|
| `lingbotvla` | 训练、推理服务、数据转换 | Python 3.12，torch 2.8.0+cu128，flash-attn 2.8.3，lerobot 0.4.2 | 在 LingBot 代码目录里运行官方 `tools/create_train_env.sh`（未改动） |
| `RoboTwin` | 仿真评测客户端 | Python 3.10，torch 2.4.1，sapien 3.0.0b1，mplib 0.2.1，pytorch3d，curobo v0.7.8 | 在 RoboTwin 目录里按官方 `script/_install.sh` 安装 |

我们在 Ubuntu 24.04（CUDA 12.8.1 + cuDNN 9.8）上运行：SAPIEN 渲染需要宿主机 NVIDIA GL/Vulkan 库，它要求 glibc ≥ 2.36，
在 22.04 上会报 `Render Error`；24.04 自带的 FFmpeg 6 也能让 torchcodec 直接解码训练视频。
另需 `git`、`unzip` 和 Hugging Face CLI（`pip install "huggingface_hub>=0.34"`）。

## 复现步骤

```bash
cd 02_代码材料
export CKPT_DIR=/path/to/ckpts DATA_DIR=/path/to/data ROBOTWIN_DIR=/path/to/RoboTwin RUNS_DIR=/path/to/runs
export LINGBOT_DIR=/path/to/lingbot-vla-v2 LINGBOT_DIR_STAGE3=/path/to/lingbot-vla-v2-stage3

# 1. 代码：官方 lingbot-vla-v2 @ be969b8 + 补丁（第 3 段训练多一份训练器补丁）
DEST=$LINGBOT_DIR bash src/setup_lingbot.sh
DEST=$LINGBOT_DIR_STAGE3 TRAINER_PATCH=1 bash src/setup_lingbot.sh
# 然后分别按官方说明创建 conda 环境 lingbotvla（LingBot）和 RoboTwin（仿真器）

# 2. 模型、RoboTwin 资源、clean 数据，并转换为 LeRobot 数据集
PYTHON=$(conda run -n lingbotvla which python) bash src/prepare_data.sh

# 3. 训练（三段，共 35,000 步）；也可以 STAGE=1 / 2 / 3 分开跑
conda activate lingbotvla && STAGE=all bash train.sh

# 4. 评测：默认加载本提交包里的 checkpoint；EVAL_TEST_NUM=10 得到与 01_评测结果/results.json 相同口径的结果
EVAL_TEST_NUM=10 bash eval.sh        # 去掉 EVAL_TEST_NUM 即官方的每任务 100 次
```

- **数据转换**（`src/convert_robotwin_to_lerobot.py`）：与官方 pi0 流程（`process_data_pi0.sh` + `generate.sh`）一致：
  state = 第 j 帧关节 `[左臂6, 左夹爪, 右臂6, 右夹爪]`，action = 第 j+1 帧，最后一帧丢弃；每条轨迹随机取一条 `seen` 指令；fps = 50。
  两处不同：图像保持原始 320×240（评测时客户端发送的就是这个分辨率）；以视频格式存储（图片嵌在 parquet 里时每条样本要解码约
  150 张 PNG，实测 3.5 s/样本，改成视频后 0.03 s/样本）。结果：50 个数据集，2,500 条轨迹，549,787 帧。
- **归一化统计**：已随代码提供。计算方式为官方 `scripts/compute_norm_stats.py ./configs/vla/norm_compute/post_data.yaml
  --data.robot_name robotwin_clean --data.train_path configs/robotwin_clean.txt --data.robot_config_root configs/robot_configs`，
  只读 clean 数据。`eval.sh` 会校验它的 SHA-256，不一致就拒绝评测，避免误用官方的 clean + randomized 统计。
- **路径**：配置文件里记录的是我们训练集群上的绝对路径（`$SCRATCH/lingbot/...`）。`train.sh` 在输出目录的 `_cfg/`
  下生成改写后的副本；`eval.sh` 通过 `src/stage_ckpt.py` 在 `$OUT/_staged/` 下生成 checkpoint 配置的本地化副本（权重只软链接）。

## 提交模型的训练过程

提交的 checkpoint 不是从基座一次训到底，而是**三段接力，共 35,000 步**。三段都只用 clean 数据和 clean-only 归一化统计，
全局 batch 都是 128，都打开图像增广（`--data.image_augment true`，只用规则 2.1.1 白名单内的像素级变换）。

| 段 | 我们的 run 名 | 步数 | 起点 | 学习率（余弦） | 训练期设置 |
|---|---|---|---|---|---|
| 1 | `strong_scratch24k` | 22,000（日程按 24,000） | `robbyant/lingbot-vla-v2-6b` 基座 | Muon，1e-4 → 5e-5，无 warmup，MoE 专家 lr 按上游 `use_moe_expert_lr` 放大 | `aug_whitelist.json`；seen 指令池替换；状态噪声 0.01；视觉塔 lr × 0.1；seed 42 |
| 2 | `strong_scratch22k_cont8k` | 8,000 | 第 1 段 `global_step_22000` | 2e-5 → 5e-6，warmup 3% | `aug_strong.json`；另加任务均匀采样 + 低动作帧降权；seed 42 |
| 3 | `s22k8k_v3uni_15k(_b200)` | 5,000（日程按 15,000） | 第 2 段 `global_step_8000` | 5e-6 → 2e-6，`lr_decay_ratio` 0.75，warmup 3% | `aug_strong_v3.json`；同第 2 段；训练器补丁（per-rank 随机流） |

- 第 1 段计划 24,000 步，在 22.5k 步被抢占，于是从已存档的 `global_step_22000` 开始第 2 段。
- 第 3 段计划 15,000 步，提交的是其中的 `global_step_5000`。它在可抢占队列上由多个作业接力完成：
  1–4,500 步由主线（seed 42，2×A100 → 2×H100）训练；4,501–5,000 步由 B200 分支（2×B200）接手主线的
  `global_step_4500`，以 seed 43 续训得到。`STAGE=3 bash train.sh` 按同样的方式两步完成。
- 不同卡型之间全局 batch 保持 128（每卡 micro batch 与梯度累积随卡型调整），训练过程中多次被抢占后从 checkpoint 续训，
  因此逐比特复现不可行；配方、超参数与随机种子安排与实际训练一致。实际生效的完整配置见
  `03_模型权重/checkpoint/v3uni_noon/lingbotvla_cli.yaml`（第 3 段训练程序自动写出的）。

## 评测

- `eval.sh`：每张卡同时评一个任务，每个任务启动一个 LingBot-VLA 推理服务（FP32，`--use_length 50`，`--use_compile True`）
  和一个 RoboTwin 评测客户端（`--seed 0 --policy_name ACT`），参数与官方 `experiment/robotwin/start_robotwin_infer_and_eval.sh` 相同。
  两个 setting 都跑完后用 `src/make_results_first_n.py` 生成官方模板格式的 `results.json`。
- 客户端以 `seed 0` 启动时，episode 种子从 100000 起依次使用，场景只由种子决定。所以 `EVAL_TEST_NUM=10` 的结果与
  100 次评测的前 10 次完全相同；`01_评测结果/results.json` 就是每个任务前 10 个 episode 的成功次数。
- 指令类型按规则 A.5：`demo_clean` 用 **seen** 指令，`demo_randomized` 用 **unseen** 指令（官方启动配置
  `policy/ACT/deploy_policy.yml` 两个 setting 都写的是 unseen，所以 `src/eval_one_task.sh` 用 `--instruction_type` 显式传入）。

## 对官方代码的改动

### LingBot-VLA 2.0

官方仓库本身不改，`src/setup_lingbot.sh` 在其副本上打补丁：

1. **方法补丁**（`src/patch_upstream.py`，结果等于 `src/patches/method.diff`，44 行新增）：每一项都由环境变量开关，
   不设置时与官方代码行为完全一致；**推理路径不受影响**（评测用的就是打了这份补丁的代码）。

   | 环境变量 | 修改的文件 | 作用 |
   |---|---|---|
   | `LINGBOT_AUG_CONFIG=<json>` | `lingbotvla/data/vla_data/transform.py` | 用 `src/lingbot_aug.py` 的白名单像素级增广替换官方的颜色增广（仍需 `--data.image_augment true`）；配置里的 `p_fourier`、`p_wrist_drop` 分别对应傅里叶幅度扰动和腕部相机随机丢弃 |
   | `LINGBOT_SEEN_POOL=1` | `lingbotvla/data/vla_data/base_dataset.py` | 每次读取时在该轨迹自己的 seen 指令池内替换指令（规则 2.1） |
   | `LINGBOT_STATE_NOISE=<std>` | `lingbotvla/data/vla_data/utils.py` | 训练时给归一化后的本体状态加高斯噪声 |
   | `LINGBOT_TASK_SAMPLING=uniform` / `LINGBOT_FRAME_WEIGHTS=1` | `lingbotvla/data/vla_data/multi_vla_dataset.py` | 50 个任务等概率采样 / 按 `meta/frame_weights.npy` 给低动作帧降权（只依赖演示数据本身的统计） |
   | `LINGBOT_VISION_LR_SCALE=<x>` | `lingbotvla/optim/optimizer.py` | 视觉编码器参数组的学习率乘以 x |

2. **训练器补丁**（`src/patches/trainer_stage3.diff`，只用于第 3 段训练；修改 `tasks/vla/train_lingbotvla.py`、
   `lingbotvla/checkpoint/checkpointer.py`）：
   - **每个 rank 独立的随机流**：官方 `set_seed` 让所有 rank 的数据加载、增广和 flow-matching 噪声完全相同，一步里实际只有
     128 / 卡数 个不同样本；补丁按 (seed, dp rank, step, epoch) 为每个 rank 生成独立随机流，同一 checkpoint 重跑结果相同。
   - 抢占队列下的健壮性：续训时沿用配置中的学习率；在不同卡数之间续训（找不到本 rank 的 extra_state 时用已保存的、
     卡数变化时不恢复 dataloader 状态）；各 rank 对是否成功加载同一 checkpoint 达成一致；按时间额外存续训点（`LINGBOT_TIME_SAVE_MIN`，
     不含 hf_ckpt；`train.sh` 单机运行时关闭）；只保留必要的旧 checkpoint（带 `keep` 文件的不删）。这些不改变训练目标与优化器。
   - 第 1、2 段用的是只打了方法补丁的代码，没有这份训练器补丁。

### RoboTwin 评测代码

官方的 `experiment/robotwin/eval_policy_client_lingbotvla.py` 本身不改。`src/eval_one_task.sh` 把它复制到 `RoboTwin/script/` 时，
用 `src/patch_eval_client.py` 做了三处修改：

1. `test_num = 100` 改为读取 `EVAL_TEST_NUM`（默认 100）。**本次提交设为 10**，所以 `results.json` 中每个任务的 `attempts` 都是 10。
2. **断点续跑**（`EVAL_STATE_FILE`）：每跑完一个 episode，把下一个种子、成功数、已测次数和 episode 编号写入状态文件；
   进程重启时从记录的位置继续。RoboTwin 用种子设置 `np.random` 和 `torch` 的随机状态，场景和指令只由种子决定，
   所以续跑后评测到的种子与不中断时完全相同，成功判定逻辑没有任何改动。
3. **专家尝试次数上限**（主办方答疑第 13 条）：官方客户端在专家失败时会无限换种子；这里每个 episode 最多 5 次专家尝试
   （`EVAL_EXPERT_MAX_ATTEMPTS`，默认 5），5 次都失败时**仍在最后一个场景上评测模型并计入测试次数**；场景无法搭建
   （专家运行报错，例如 `UnStableError`）或无法生成指令时，该 episode 记为失败。这一改动只会让成绩更保守。

## 辅助监督（冻结教师模型）披露（规则 2.5 / 3.4）

训练沿用官方 LingBot-VLA 2.0 配方自带的深度、视频两路对齐辅助损失（`configs/robotwin_clean.yaml` 中的 `align_params`）。
教师模型全程冻结，只接收 clean 演示图像，输出只作为辅助训练目标，推理时不加载；动作主标签全部来自 clean demonstration。

| 教师 | 权重来源 | 接入位置 | 辅助损失权重 |
|---|---|---|---|
| LingBot-Depth（MoRGBD），配合 MoGe-2 ViT-B normal 编码器 | `robbyant/lingbot-vla-v2-6b` 中的 `depth/model.pt`；`Ruicheng/moge-2-vitb-normal` 中的 `model.pt` | `align_params.depth`：当前帧、未来帧的深度对齐查询 token | `depth_loss_weight` 0.004，`future_depth_loss_weight` 0.004 |
| DINO-Video | `robbyant/lingbot-vla-v2-6b` 中的 `dino_video/teacher_step_10000.pth` 和 `config.yaml` | `align_params.video`：当前帧、未来帧的视频特征对齐 | `future_video_loss_weight` 0.004 |
