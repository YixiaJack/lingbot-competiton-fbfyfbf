# 复现与调优说明

## 1. 基础信息

- 团队名称：（略）
- 使用模型版本：LingBot-VLA 2.0，基座 `robbyant/lingbot-vla-v2-6b`；代码为官方 [lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2) @ `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`
- 代码版本 / Commit：上游 `be969b8` + `02_代码材料/src/` 中的补丁（`src/setup_lingbot.sh` 自动拉取并打补丁）；
  仿真评测 RoboTwin @ `13c3c47ff4312dd62484bcd51be034af55c062d1`
- 提交 checkpoint 名称：`v3uni_noon / global_step_5000`
  - 路径：`03_模型权重/checkpoint/v3uni_noon/checkpoints/global_step_5000/hf_ckpt`（文件校验见同目录 `SHA256SUMS.txt`）
  - 推理服务从 `hf_ckpt/../../../lingbotvla_cli.yaml` 读取训练配置，请保持该目录层级

## 2. 训练复现

- 使用数据集：RoboTwin 2.0 Aloha-AgileX 50 个任务的 `aloha-agilex_clean_50`，每个任务 50 条，共 2,500 条轨迹、549,787 帧
- 是否使用 randomized 数据训练：**否**。归一化统计也只在 clean 数据上重新计算（`02_代码材料/configs/norm_stats/robotwin_clean.json`），
  未使用官方基于 clean + randomized 计算的 `assets/norm_stats/robotwin.json`
- 一键训练脚本：`02_代码材料/train.sh`（`STAGE=all bash train.sh`，三段的设置均已内置）
- 训练配置文件：`02_代码材料/configs/robotwin_clean.yaml`；本 checkpoint 实际生效的完整配置为
  `03_模型权重/checkpoint/v3uni_noon/lingbotvla_cli.yaml`
- robot config：`02_代码材料/configs/robot_configs/robotwin_clean.yaml`

提交的 checkpoint 由**三段接力训练**得到，共 35,000 步，全局 batch 均为 128：

| 段 | 步数 | 起点 | 优化器 / 学习率（余弦） | 训练期设置 |
|---|---|---|---|---|
| 1 | 22,000（日程按 24,000） | `robbyant/lingbot-vla-v2-6b` | Muon，1e-4 → 5e-5，无 warmup | 白名单增广 `aug_whitelist.json`；seen 指令池替换；状态噪声 0.01；视觉塔 lr × 0.1；seed 42 |
| 2 | 8,000 | 第 1 段 `global_step_22000` | 2e-5 → 5e-6，warmup 3% | 增广 `aug_strong.json`；另加任务均匀采样与低动作帧降权；seed 42 |
| 3 | 5,000（日程按 15,000） | 第 2 段 `global_step_8000` | 5e-6 → 2e-6，`lr_decay_ratio` 0.75，warmup 3% | 增广 `aug_strong_v3.json`；同第 2 段；训练器补丁 |

- 第 1 段计划 24,000 步，在约 22.5k 步时被抢占，因此从已存档的 `global_step_22000` 开始第 2 段。
- 第 3 段的 1–4,500 步由主线以 seed 42 训练（2×A100，后为 2×H100），4,501–5,000 步由另一个作业（2×B200）
  接手主线的 `global_step_4500`、以 seed 43 续训得到本次提交的 `global_step_5000`。`train.sh` 的 `STAGE=3` 按同样的方式分两步完成。
- 训练在我的算力（可抢占队列）上进行，过程中多次被抢占并从 checkpoint 续训，不同卡型之间以每卡 micro batch 和
  梯度累积保持全局 batch 不变。因此无法逐比特复现，但配方、超参数与随机种子安排与实际训练一致。

## 3. 评测复现

- 一键评测脚本：`02_代码材料/eval.sh`（两个 setting 都跑，最后生成 `results.json`）
- checkpoint 加载方式：`eval.sh` 默认加载本提交包中的 `03_模型权重/checkpoint/v3uni_noon/checkpoints/global_step_5000/hf_ckpt`
  （可用 `CKPT=` 指定）。它先调用 `src/stage_ckpt.py`，把 checkpoint 训练配置中的集群路径改写为本机路径（写到 `$OUT/_staged/`，
  权重只软链接），并校验 clean-only 归一化统计的 SHA-256，不一致则拒绝评测
- clean setting 评测命令：`SETTINGS=demo_clean EVAL_TEST_NUM=10 LINGBOT_DIR=... ROBOTWIN_DIR=... CKPT_DIR=... bash eval.sh`
- randomized setting 评测命令：`SETTINGS=demo_randomized EVAL_TEST_NUM=10 LINGBOT_DIR=... ROBOTWIN_DIR=... CKPT_DIR=... bash eval.sh`
  （不指定 `SETTINGS` 时两个都跑；不指定 `EVAL_TEST_NUM` 时为官方的每任务 100 次）
- 评测环境：RoboTwin @ `13c3c47`（conda 环境 `RoboTwin`：Python 3.10，torch 2.4.1，sapien 3.0.0b1）；
  推理服务（conda 环境 `lingbotvla`：Python 3.12，torch 2.8.0+cu128）FP32、`--use_length 50`；Ubuntu 24.04，CUDA 12.8
- 评测口径：**每个任务每个 setting 测 10 次**（客户端 `seed 0`，episode 种子从 100000 起依次使用），
  `demo_clean` 用 seen 指令、`demo_randomized` 用 unseen 指令（规则 A.5）。结果：

  | setting | 成功次数 / 测试次数 | 成功率 |
  |---|---|---|
  | demo_clean | 407 / 500 | 81.4% |
  | demo_randomized | 335 / 500 | 67.0% |

## 4. 代码改动说明

- 是否修改 LingBot-VLA 2.0 官方代码：**是，以补丁形式，官方仓库本身未改**。由 `src/setup_lingbot.sh` 打在官方代码副本上：
  - 方法补丁 `src/patch_upstream.py`（= `src/patches/method.diff`）：修改 `lingbotvla/data/vla_data/transform.py`、`base_dataset.py`、
    `utils.py`、`multi_vla_dataset.py` 与 `lingbotvla/optim/optimizer.py`，实现训练期白名单增广、seen 指令池替换、状态噪声、
    任务均匀采样与帧权重、视觉塔学习率倍率。每一项都由环境变量开关，不设置时与官方行为一致；推理路径不受影响。三段训练与评测都使用。
  - 训练器补丁 `src/patches/trainer_stage3.diff`：修改 `tasks/vla/train_lingbotvla.py`、`lingbotvla/checkpoint/checkpointer.py`，
    只用于第 3 段训练。内容为每个 rank 独立的随机流（官方实现中各 rank 的数据加载、增广和 flow-matching 噪声完全相同），
    以及抢占队列下的续训健壮性（续训沿用配置中的学习率、不同卡数间续训、各 rank 对加载结果达成一致、按时间存续训点、清理旧 checkpoint）。
    这些不改变训练目标、模型结构与优化器。
- 是否修改 RoboTwin 官方评测代码：**是，仅评测客户端，且以补丁副本形式**。官方 `experiment/robotwin/eval_policy_client_lingbotvla.py`
  本身不改，`src/eval_one_task.sh` 复制它时由 `src/patch_eval_client.py` 修改三处：
  1. `test_num` 改为读取 `EVAL_TEST_NUM`（本次为 10）；
  2. 断点续跑（`EVAL_STATE_FILE`）：作业被打断后从下一个种子继续，评测到的种子序列与不中断时完全相同；
  3. 专家尝试次数上限（主办方答疑第 13 条）：每个 episode 最多 5 次专家尝试，都失败时仍在最后一个场景上评测并计入测试次数，
     场景无法搭建时记为失败。只会让成绩更保守。

  另外 `src/eval_one_task.sh` 用 `--instruction_type` 按规则 A.5 显式传入指令类型（官方启动配置对两个 setting 都写的是 unseen）。
- 修改内容与对应文件位置的详细说明见 `02_代码材料/README.md`「对官方代码的改动」。

## 5. 调优说明

- 训练参数调整：从基座重新训练（不使用官方 robotwin 后训练权重），三段递减学习率接力：1e-4 → 5e-5（Muon，22k 步）、
  2e-5 → 5e-6（8k 步）、5e-6 → 2e-6（5k 步）；全局 batch 128；后两段 warmup 3%。
- 数据处理方式：只用 clean 数据，并在 clean 数据上重算归一化统计；数据以视频格式存储、图像保持 320×240；
  每条轨迹在其 seen 指令池内随机替换指令；第 2、3 段对 50 个任务等概率采样，并对几乎不动的帧（动作与状态差 < 1e-3）降权到 0.3。
- 模型结构改动：无，模型结构与官方 LingBot-VLA 2.0 一致（沿用官方配方自带的深度、视频两路冻结教师对齐辅助损失）。
- 推理策略：FP32 推理，单 checkpoint，不做集成、不按任务路由；图像分辨率与训练一致。
- 主要提升点：clean-only 归一化统计；规则白名单内的强图像增广（颜色、噪声、模糊、几何、遮挡、傅里叶幅度扰动、腕部相机随机丢弃等）
  与本体状态噪声，主要用于提升 randomized setting 下的泛化；分段递减学习率续训。

### 已知不足

- 每个任务只测了 10 次，少于官方要求的 100 次。
- 第 3 段计划 15,000 步，提交的是第 5,000 步的 checkpoint。
