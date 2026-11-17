# DINOv2 视觉特征融合实验

[← 返回初赛总览](README.md) · 代码：[`prelim/dino/`](../../prelim/dino/)、[`patches/lingbot-vla-v2-dino.diff`](../../prelim/patches/lingbot-vla-v2-dino.diff)

**结论先说**：在 22k+8k 上加一路冻结的 DINOv2 特征、续训 6,000 步后，clean 持平，randomized 在相同种子上反而更差
（139 个配对回合 80 对 90，p = 0.03）。但这次续训的数据多样性因为两个随机流 bug 大幅下降，结论被污染，
不能说明 DINOv2 特征本身没用。我们在截止前一天放弃了这条线。

## 1. 动机

randomized 设置换背景、加杂物、改光照。DINOv2 这类自监督视觉特征通常对这类外观变化更稳。规则允许加入外部公开预训练模块，
只要披露（主办方答疑第 6、7 条），而主干仍需从 lingbot-vla-v2-6b 初始化（第 14 条）。

## 2. 设计

| 项 | 内容 |
|---|---|
| 模块 | DINOv2 with registers, ViT-B/14（[`facebook/dinov2-with-registers-base`](https://huggingface.co/facebook/dinov2-with-registers-base)，Apache-2.0），**全程冻结**，只作特征提取器 |
| 输入 | 与 Qwen3-VL 视觉塔看到的完全相同的图像：从处理器输出的 `pixel_values` 反解出图像（CPU 测试要求误差 < 1.5/255），缩放到 224×224，按 ImageNet 均值方差归一化。训练时的增广因此也作用在这份图像上，数据管线和推理服务都不用改 |
| 接入位置 | `FlowMatchingV2.embed_prefix`：DINOv2 的 patch 特征平均池化到 8×8（与 Qwen merger 输出的每路相机视觉 token 网格一致），经 `LayerNorm` 和 `Linear(768→2560)` 后**加到**对应相机的视觉 token 上。序列长度、注意力掩码、位置编码都不变 |
| 新增可训练参数 | 只有 `norm` 和 `proj`，约 2M。`proj` **零初始化**，所以第 0 步的模型与原模型逐元素相同 |
| 权重存放 | DINOv2 权重存进同一个 `hf_ckpt`（`model.dino_fusion.dino.*`），推理仍只加载一份权重，不涉及集成或任务路由 |
| 学习率 | 主干同续训；融合层 ×10（`LINGBOT_DINO_LR_SCALE`）；视觉塔 ×0.1 |

实现：[`dino/dino_fusion.py`](../../prelim/dino/dino_fusion.py)（模块）、[`dino/patch_dino.py`](../../prelim/dino/patch_dino.py)
（把改动打进**另一份**上游 checkout，不碰正在训练的那份）、[`dino/make_dino_init.py`](../../prelim/dino/make_dino_init.py)
（构建初始 checkpoint：22k+8k 权重 + DINOv2 权重 + 单位 LayerNorm + 零投影）。

## 3. 验证

- **CPU 测试**（[`test_dino_cpu.py`](../../prelim/dino/test_dino_cpu.py)）：从 `pixel_values` 反解出的图像与原图一致（含 bf16 转换后）；由反解图像和原图算出的 DINO 特征余弦相似度 > 0.999；token 网格为 3 路相机 × 64 × 768；零初始化时输出严格为 0；`train()` 后 DINOv2 仍处于 eval 模式且不求梯度。
- **GPU 测试**（[`test_dino_gpu.py`](../../prelim/dino/test_dino_gpu.py)，走真实的推理加载器）：
  1. 原 `hf_ckpt` 和零初始化的 DINO 版，相同输入、相同种子，输出动作完全相同；
  2. 把投影层换成随机值，动作随之改变（分支确实接上了）；
  3. 训练后的 DINO checkpoint 能严格加载（没有多余或缺失的键）并正常推理。
- 第一次 GPU 测试失败在 `Unexpected key model.dino_fusion...`：上游 `LingbotVLAConfig.__init__` 调 `super().__init__()` 时不带
  kwargs，未知字段被**静默丢弃**，模型根本没建融合层。修复是训练脚本把新字段 `setattr` 到模型配置上，推理端自己补上 yaml 里的额外字段。

## 4. 训练

| 项 | 内容 |
|---|---|
| 起点 | 22k+8k（`strong_scratch22k_cont8k/global_step_8000`）+ 零初始化融合层 |
| 批量 | 全局 128（B200 / H100 每卡 16，2×A100 每卡 8、4×L40S 每卡 4，后两者开梯度检查点和全分片） |
| 硬件 | handoff 组（[hpc.md 第 3 节](hpc.md#3-handoff-组多种卡型接力训练同一个-run)）：4×B200 约 1.97 s/步，2×A100 约 15.4 s/步 |
| 评测的 checkpoint | 第 6,000 步 |

配方在训练过程中改过，记录如下（[`configs/dino_cont8k_6k_lingbotvla_cli.yaml`](../../prelim/configs/dino_cont8k_6k_lingbotvla_cli.yaml)
是第一次启动时写出的配置）：

- 开始时：`aug_strong.json`、任务均匀采样，学习率 1e-5 → 5e-6（3% warmup），计划 6,000 步。
- 约第 500 步起重新提交：改用 `aug_strong_v3.json` 和温度采样 `temp:2`，计划步数改为 15,000，学习率 5e-6 → 2e-6。
  但由于续训时学习率被 checkpoint 覆盖的 bug（[hpc.md 5.2](hpc.md#52-续训时学习率被-checkpoint-覆盖)），这次改动后的一段时间仍按 1e-5 训练，直到修复。
- 第 6,000 步时冻结一份副本用于评测。

## 5. 结果

| 比较 | 回合 | DINO 第 6k 步 | 22k+8k | 说明 |
|---|---|---|---|---|
| clean，每任务 7 回合，44 个任务 | 308 | 257 = 83.4% | 同 44 个任务上 2202/2640 = 83.4%（60 回合） | 持平 |
| clean，相同种子配对 | 290 | 242 | 243 | 持平 |
| randomized，相同种子配对（21 个任务） | 139 | 80 | 90 | 只有一方成功的回合 4 对 14，McNemar p = 0.03 |

## 6. 为什么输了：结论被污染

DINO 版和 22k+8k 的差别不只是 DINOv2：它还多了 6,000 步小学习率的 clean 续训，而这段续训的数据多样性塌了：

- **续训重放**：温度采样在 worker 里抽样、不看采样器下标，每次续训都把 epoch 开头的样本再抽一遍。各次启动之间按“续训后第几步”对齐，
  B200 那几次启动的逐步损失相关系数在 0.55–0.97 之间。第 6,000 步所在的这条谱系里，第 1 个 epoch 的 4,295 个抽样位置只出现过约 1,354 个，
  前约 300 个 batch 被训了大约 7 遍。
- **各 rank 相同**：所有 rank 种子相同，温度采样下每张卡拿到同样的样本、增广和噪声。
- 两者叠加，不同样本数大约只有一次正常续训的 1/7。VLA 损失在 6,000 步里一直平在约 0.0145（22k+8k 结束时是 0.0137）。

详见 [hpc.md 5.3–5.4](hpc.md#53-续训时数据重放)。所以 randomized 变差可能来自过拟合重复样本，也可能来自 DINOv2，这次实验分不开。

## 7. 后续与教训

- 放弃 DINO 后，我们用修好随机流的训练器、不带融合层做了同样的小学习率续训（阶段 3），它同样没有超过 22k+8k
  （[README 6.2](README.md#62-按种子配对的比较都以-22k8k-为对照)）。这说明“22k+8k 之后再用小学习率续训”本身就难有收益，
  DINOv2 需要一次干净的对照实验才能下结论，比如从阶段 1 或更早就加入融合层。
- 零初始化 + 逐元素等价测试是加模块时很划算的保险：它保证新模块在第 0 步不改变模型，任何后续差异都来自训练。
- 改上游配置类之前，先确认它会不会吞掉未知字段。
