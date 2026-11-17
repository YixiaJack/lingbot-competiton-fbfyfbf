# 22k 评测结果（q50x10_scratch22k）

- 模型：`strong_scratch24k/checkpoints/global_step_22000/hf_ckpt`（从 lingbot-vla-v2-6b 用 clean 数据训练 22k 步，白名单增广）
- 环境：我的算力，NVIDIA GPU，RoboTwin `13c3c47`（官方指定版本）+ CuRobo，FP32，执行长度 50，clean 归一化统计
- 每个任务 10 次，seed 0（种子从 100000 起）

## 合计

| setting | 成功 / 次数 | 成功率 |
|---|---|---|
| clean | 411 / 480（48 个任务） | 85.6% |
| randomized | 293 / 490（49 个任务） | 59.8% |

clean 缺 2 个任务、randomized 缺 1 个任务：当时的评测作业被撤掉了，没有跑完。

## 注意：这是旧口径，偏乐观，不能作为最终成绩

1. clean 用的是 **unseen** 指令（规则要求 clean 用 seen）。
2. 专家失败时会**无限换种子**（答疑第 13 条要求每个场景最多 5 次）。所以这里只统计了专家能完成的场景，
   clean 一共跳过了 283 个种子，randomized 跳过了 456 个。

## 文件

| 文件 | 内容 |
|---|---|
| `汇总.csv` | 每个任务 clean / randomized 的成功次数和成功率 |
| `每回合明细.csv` | 每个回合一行：种子、成功或失败、用了多少步、步数上限、这一回合前因为专家失败跳过了几个种子 |
| `原始日志/<setting>/eval_logs/<任务>.log` | 评测客户端的完整输出。开头可能有几段 CUDA 报错，那是作业最初被分到 Blackwell 显卡时的失败记录（CuRobo 不支持），之后在 A100/H100 上重跑成功 |
| `原始日志/<setting>/inference_logs/<任务>.log` | 推理服务日志 |
| `原始日志/<setting>/eval_results/<任务>/_result.txt` | RoboTwin 写出的结果文件 |
| `原始日志/<setting>/state/<任务>.json` | 断点续跑用的进度记录 |
| `原始日志/<setting>/<任务>.result` | "成功次数 总次数" |

没有录视频（评测时 `eval_video_log=False`）。失败的回合都是跑满步数上限仍未完成，没有中途报错。
