# 22k+8k 评测结果（q50x7_s22k8k_8k）

- 模型：`strong_scratch22k_cont8k/checkpoints/global_step_8000/hf_ckpt`（由 lingbot-vla-v2-6b 用 clean 数据训练 22k 步，再用强增广 + Fourier + 腕部相机丢弃 + 任务均匀采样续训 8k 步）
- 环境：我的算力，NVIDIA GPU（A100 / H100 / L40S），RoboTwin `13c3c47`（官方指定版本）+ CuRobo，FP32，执行长度 50，clean 归一化统计
- **新口径（符合主办方答疑）**：clean 用 seen 指令，randomized 用 unseen 指令（每个结果文件里都核对过）；专家每个场景最多试 5 次，5 次都失败时照样评测模型并计入次数（答疑第 13 条）
- 每个任务 7 次，seed 0（种子从 100000 起）

## 合计

| setting | 成功 / 次数 | 成功率 |
|---|---|---|
| clean | 296 / 350（50 个任务） | 84.6% |
| randomized | 228 / 350（50 个任务） | 65.1% |
| 合计 | 524 / 700 | 74.9% |

## 专家相关统计

- 因为专家失败跳过的种子：clean 163 个，randomized 190 个（每个回合最多跳 4 个）。
- 专家 5 次都失败、照样评测并计入次数的回合：clean 7 个（其中模型成功 3 个），randomized 14 个（其中成功 1 个）。

## 文件

| 文件 | 内容 |
|---|---|
| `汇总.csv` | 每个任务 clean / randomized 的成功次数和成功率 |
| `每回合明细.csv` | 700 个回合，每回合一行：种子、成功或失败、用了多少步、步数上限、这一回合前跳过了几个种子、是否属于"专家 5 次都失败后照样评测"的回合 |
| `原始日志/<setting>/eval_logs/<任务>.log` | 评测客户端的完整输出 |
| `原始日志/<setting>/inference_logs/<任务>.log` | 推理服务日志 |
| `原始日志/<setting>/eval_results/<任务>/_result.txt` | RoboTwin 写出的结果文件（含指令类型） |
| `原始日志/<setting>/state/<任务>.json` | 断点续跑用的进度记录（评测期间有工人被抢占，续跑后种子序列与不中断时相同） |
| `原始日志/<setting>/<任务>.result` | "成功次数 总次数" |

逐回合明细已和每个任务的成功次数逐一核对，全部一致。没有录视频。
