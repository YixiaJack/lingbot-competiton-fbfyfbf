# 推理端异步 + RTC（5080，11/14 夜）

训练端是 [p10 训练时 RTC](../../training/patches/p10_rtc.py)，v4 从总第 3000 步起使用。官方推理服务本身已经支持异步：后台准备下一段、跳过推理期间已经走掉的步数、把正在执行那段剩下的动作和前缀长度交给 `policy.preprocess`。之前是 LingBot 适配器和配置校验把 RTC 挡住了。

## 改动（[rtc_infer_patch.py](rtc_infer_patch.py)，可重复运行，改前备份到 `~/rtc/backup`）
1. **模型**：打 p10，`sample_actions(prefix_actions, prefix_len)`。前缀为 0 时走原路径。推理机上补丁后的模型文件和训练用的 code_v4r 逐字节相同。
2. **适配器**（`code/deploy/policy_adapter/policy_adapter.py`）：
   - `validate_rtc`：配置不对就拒绝 START，要求 `refill_threshold ≤ 24` 且 `use_length ≥ 2×refill_threshold`；
   - 判断权重是否 RTC 训练过：hf_ckpt 里有 `rtc_trained.json`，或者路径写在 `~/rtc/allow.txt` 里。没有标记不会拒绝，照样异步运行，只是不用前缀，日志会给出警告；
   - `preprocess` 把上一段剩下的前 d 步绝对动作挂在 `__rtc_prefix__` 上。
3. **原生推理**（`code/lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py`）：
   - 前缀要换算到模型的动作空间（归一化、相对当前关节、补齐到 55 维）。官方 `FeatureTransform.unapply` 每一维都是仿射的，用一次 56 行的探测就能求逆；
   - 换算结果再经 `unapply` 还原一遍，复现不了前缀就不用前缀；
   - 每段打一行日志 `RTC chunk: d=.. roundtrip=.. prefix_kept=.. junction/median step: arm a/b, effector c/d`。
4. **配置校验**（`code/deploy/utils/arguments.py`）：去掉对 `rtc_enabled` 一刀切的拒绝。
5. **配置文件**：
   - `config/inference.sync.yaml` 是原配置；
   - `config/inference.rtc.yaml` 是 `use_length: 44`，加上 `async_pipeline: {enabled: true, refill_threshold: 20, rtc_enabled: true}`；
   - 在用的 `config/inference.yaml` 目前仍是同步配置。

参数：推理约 0.35–0.47 秒，即 11–14 步（30 Hz）。还剩 20 步时请求下一段，前缀长度 20，不超过训练时的最大值 24。下一段大约在第 14 步时接上，每段新执行约 24 步（0.8 秒），不再停顿。

## 测试（[test_rtc_infer.py](test_rtc_infer.py)，GPU，真实权重 + 录好的帧，11/14 23:45）
| 项目 | 结果 |
|---|---|
| RTC 配置能解析；未标记的第 2000 步权重被识别为没有 RTC 训练 | 是 |
| 不带前缀，同一种子跑两次 | 最大差 6–9e-3 rad，GPU 运算本身不确定 |
| 请求 RTC 但权重未标记 | 不挂前缀；与无前缀调用差 0.8–1.5e-2 rad，和上一行同一量级 |
| 前缀 20 步：还原误差 / 前缀保留 | 1e-8 / 完全相同 |
| 耗时 | 有前缀 355 ms，无前缀 355 ms |
| 接缝（第 2000 步权重，**没有 RTC 训练**） | 手臂 0.10–0.24 rad，接缝后各步中位数 0.03–0.04；夹爪 0.01，和平常一样 |

没训练过的权重会忽略前缀，接缝处就跳一下。这是预期结果，也说明不加 RTC 直接开异步会出问题。拿到 RTC 权重后要重测接缝。

## 切换（[switch.sh](switch.sh)，推理机 `~/rtc/switch.sh`）
- `switch.sh rtc <hf_ckpt> --trained`：标记权重、换成 RTC 配置、重启空闲的服务、重新加载权重；
- `switch.sh sync`：换回同步；
- `--dry-run` 只显示会做什么，不改动。会话还在跑时脚本拒绝执行。
