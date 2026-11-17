# 在可抢占的 Slurm 分区上训练和评测

[← 返回初赛总览](README.md) · 代码：[`prelim/scripts/hpc/`](../../prelim/scripts/hpc/)

初赛只有一周。我的算力（一个共享的 Slurm 集群）当时几乎满载：普通队列里 8×H200 的预计开始时间是 5 天以后。
唯一能马上拿到 GPU 的是**抢占分区**（preemption partitions）：作业几分钟到几小时内就能开始，但节点的所有者一需要，
我们的作业在开始约 30 分钟后就可能被杀掉，而且没有宽限时间（GraceTime 0），然后自动重新排队。

所以每个作业都必须是幂等、可续跑的。这一页记录我们为此做的设计，以及这些设计本身带来的 bug。

## 1. 集群环境

| 事实 | 对我们的影响 |
|---|---|
| 抢占分区：`--comment="preemption=yes;preemption_partitions_only=yes;requeue=true" --requeue` | 默认都用它（[`submit.sh`](../../prelim/scripts/hpc/submit.sh)），每个脚本都要能从中断处继续 |
| 被抢占的时间点：作业开始后约 30 分钟（集群繁忙时），不发 SIGTERM 宽限 | 按步数存档不够，要按时间存档（第 4 节） |
| 只写一种 GPU 型号的作业，比写 `a100\|h100` 这种“或”约束的预计开始时间早得多 | 每种卡型单独提交一个作业，组成 handoff 组（第 3 节） |
| 登录节点内存上限 3 GB | 下载模型、编译环境要放进作业里跑，登录节点上会被 OOM 杀掉 |
| Apptainer 容器 + overlay 里的两个 conda 环境（`lingbotvla` 训练和推理，`RoboTwin` 仿真） | 所有命令经 [`run_in_env.sh`](../../prelim/scripts/hpc/run_in_env.sh) 进入容器；镜像必须是 Ubuntu 24.04：宿主机挂进来的 NVIDIA GL/Vulkan 库要求 glibc ≥ 2.36，22.04 镜像里 SAPIEN 报 `Render Error`；24.04 自带的 FFmpeg 6 也让 torchcodec 能直接解码训练视频 |
| Slurm 命令不在容器里；Slurm 在提交时保存作业脚本，重新排队时用的是保存的那份 | 需要调 `squeue`/`scontrol` 的逻辑写在宿主机一侧的 sbatch 脚本里；改了脚本，已提交的作业不会生效，要重新提交 |
| GPU 型号：B200 能训练（flash-attn 有 sm_100）；RTX PRO 6000 Blackwell（sm_120）上 CuRobo 跑不起来 | 评测只用 A100 / L40S / H100 / H200 |
| 每用户 GPU 上限挂在 QOS 上，而且会动态调整；用户不能自选 QOS | 见第 7 节，一次误判 |
| 长时间 GPU 利用率过低（A100 约 50% 以下超过 2 小时）的作业会被取消 | 评测改成多客户端共享推理服务（见 [README 第 5.3 节](README.md#53-让评测在抢占队列上跑完)） |
| 项目盘配额 5 TB，一个完整 checkpoint 约 73 GB | 训练脚本自动清理旧 checkpoint，要评测的 checkpoint 放一个 `keep` 文件 |

## 2. 全局 batch 不随卡型变化

为了让同一个 run 能在不同卡型之间接力，所有成员都用全局 batch 128，每卡 micro batch 按显存选
（[`train.sbatch`](../../prelim/scripts/hpc/train.sbatch)）：

| 卡型 | 每卡 micro | 累积步数 | 其他 |
|---|---|---|---|
| ≥ 120 GB（B200、H200），`train.sbatch` 默认 | 32 | 1 | |
| 80 GB（H100、A100），`train.sbatch` 默认 | 16 | 2 | |
| 2×B200 | 16 | 4 | 全分片 |
| 2×A100、2×H100、2×H200 | 8 | 8 | 梯度检查点 + 全分片 |
| 4×L40S（46 GB） | 4 | 8 | 梯度检查点 + 全分片 |

同一个组里大卡也常降到 16 或 8（阶段 2 的 B200 用 16 × 2，阶段 3 的 4×H200 用 8 + 全分片）。只有全局 batch 必须一致：
每个 rank 每步读 128 / 卡数 个样本，训练时再切成 micro batch，所以 micro 不同不影响续训。

两个当时踩到的坑：

- **上游不会推算累积步数**。它要求全局 batch 恰好等于 micro × 卡数 × 累积步数，否则直接报错。第一个 H200 作业就死在这里，
  之后 `train.sbatch` 自己算好 `--train.gradient_accumulation_steps` 传进去。
- **`nvidia-smi ... | head -1` 杀死了作业**。脚本开着 `set -o pipefail`，`head` 读完一行就退出，`nvidia-smi` 收到 SIGPIPE，
  整条管道返回非零，作业失败。改成 `nvidia-smi -i 0 --query-gpu=...` 直接只查 0 号卡。

## 3. handoff 组：多种卡型接力训练同一个 run

单一卡型的作业要排很久，同时提交多种卡型又可能两个作业同时训同一个 run。我们的做法：同一个 run 的所有成员用同一个
`--job-name`，在 `train.sbatch` 里协调。

1. **最初版本（双胞胎）**：B200 和 H100/H200 各提交一份，谁先开始就留谁，手动取消另一个。教训：一定要等赢家真的开始出训练步，
   再取消另一个。H200 那个“赢家”就是刚开始就死在上面的 batch 问题上。
2. **HANDOFF=1**：慢卡（4×A100）先把 run 跑起来，H100、B200 成员一开始就取消组里其他作业，等它们退出后从最新 checkpoint 续训。
3. **HANDOFF_RANK=n（带等级）**：每个成员启动时把自己的等级写进 `<run>/.handoff/<jobid>`，然后看正在运行的兄弟作业：
   - 有等级更高的（或等级相同但作业号更小的）在跑，就把自己 `scontrol requeue` 回队列，退出；
   - 否则把等级更低的正在运行的兄弟 requeue（而不是取消），让它们留在队列里当后备，然后接手。
   - 排队中的兄弟从不动。这样快卡被抢占后，总有后备在排队；任何时刻只有一个成员在训练。
4. **不可抢占的成员等级最高**。后来拿到了一些常规（不可抢占）分区的 GPU。规则定为：不可抢占成员（等级 10–15）高于所有可抢占成员（1–5），
   只会被更快的不可抢占成员替换，因为一个不会被收走的 12 小时分配不该换成一个随时会被收走的。

最多的时候一个 run 挂了 15 个成员：4×B200、2×B200、4×H200、2×H200、4×H100、2×H100、4×A100、2×A100、4×L40S，
分别在可抢占和不可抢占分区上（[`submit_plain_cont.sh`](../../prelim/scripts/hpc/submit_plain_cont.sh)）。

### 分支与接管（branch / adopt）

阶段 3 有两条线并行：2×H100 主线，以及 B200 成员组成的分支线。为了不浪费任何一边的进度，每个成员启动时先跑
[`branch_seed.sh`](../../prelim/scripts/hpc/branch_seed.sh)：如果另一条线最新的**完整** checkpoint 步数比自己的新，就用硬链接
（`cp -al`，不占额外空间）把它拿过来，然后从那里续训（权重、优化器、学习率日程、步数都接上）。分支线换用 seed 43，
保证数据抽样和主线不同。提交的 22k+8k+5k 就是这样来的：主线训到 4,500 步，B200 分支接手训到 5,000 步。

“完整”的定义：`model/` 和 `optimizer/` 的 `.metadata` 都在，每个 rank 的 `extra_state_rank_*.pt` 都非空，最近 3 分钟内没有文件被写入。
找不到任何 checkpoint 时脚本返回非零，作业回到队列，绝不从头开始训。

## 4. 按时间存档

阶段 2 在 4×B200 上 1.9 s/步，每 1,000 步存一次就是约 31 分钟，正好和抢占窗口差不多长：运气不好的话，每次都在存档前被杀，
进度全丢。慢卡（2×A100 约 15 s/步）更是一次都存不下来。

做法（[`dino/patch_dino.py`](../../prelim/dino/patch_dino.py) 的 time_save 部分，后来整理进
[`trainer_stage3.diff`](../../prelim/patches/trainer_stage3.diff)）：除了按步数存档，还在进程启动后第 24、48、72……分钟各存一次
只用于续训的 checkpoint（不写 `hf_ckpt`，省时间和空间）。是否该存由 0 号 rank 的时钟决定并广播，保证所有 rank 同时进入集合通信。

第一版是“距上次存档 24 分钟”。结果一次步数存档恰好落在第 13 分钟左右，把下一次按时间存档推到了第 37 分钟，超过了抢占时间，
丢了约 60 步。第二版改成从进程启动起的固定网格，只在 3 分钟内刚存过时跳过。

配套的清理：每次存档后只保留刚存的这个、最新一个带完整 `hf_ckpt` 的（用于评测），以及任何含 `keep` 文件的目录；
每种保存时的卡数各保留最新一个（原因见下节）；只删比刚存的更旧的，从不删更新的。

## 5. 续训一致性：几个真实发生过的 bug

### 5.1 换卡数续训时挂死

2×A100 成员存了第 664 步的 checkpoint，里面只有 rank 0、1 的 `extra_state`。随后 4×B200 成员接手：rank 2、3 找不到自己的文件，
抛 `FileNotFoundError`，然后**各自悄悄**回退到更早的 checkpoint（日志只在 rank 0 打印，所以看不到），而 rank 0、1 已经在
`dist.barrier()` 等着。10 分钟后 NCCL 超时，作业状态是 FAILED（不是被抢占），所以不会自动重新排队，这个成员就从组里消失了。

修复：

- 缺本 rank 文件时读一个已保存 rank 的文件；保存时的卡数和当前不同，就不恢复各 rank 的数据加载器状态（数据顺序重新开始，其余照常续训）。
- **续训协商**：每个候选 checkpoint 都要所有 rank 都加载成功才算数（`all_reduce` 取 MIN），否则所有 rank 一起试下一个更旧的。
- 一个都加载不了时，训练程序不自己从头开始，而是把决定写进 `<run>/.resume_failed/<jobid>`：第一次失败且有别的卡数存的 checkpoint，
  就 requeue 交给其他成员；否则把读不了的 checkpoint 挪到一边，从这个 run 的初始模型重新训。由宿主机一侧的 `train.sbatch` 执行
  （因为容器里没有 Slurm 命令）。

### 5.2 续训时学习率被 checkpoint 覆盖

DINO 实验中途用新的学习率（5e-6）重新提交，但训练实际仍按 1e-5 跑。原因是 PyTorch 的 `LambdaLR.load_state_dict` 会恢复
checkpoint 里保存的 base lr，命令行的新值被覆盖。修复：续训时保留配置里的学习率（`LINGBOT_RESUME_KEEP_LR=1` 可恢复旧行为）。

### 5.3 续训时数据重放

我们加的温度采样（`LINGBOT_TASK_SAMPLING=temp:T`，按帧数的 1/T 次方给任务分配概率）在数据加载 worker 里用 `torch.rand`
抽任务和帧，没有用采样器给的下标。上游用的 `StatefulDataLoader` 续训时会恢复 worker 的基础种子，但 worker 的随机数生成器从
“基础种子 + worker 编号”重新开始。结果是每次续训都把这个 epoch 开头的那一批样本重新抽一遍。

证据：DINO 那条线每次启动后按“续训后第几步”对齐，各次启动的逐步损失相关系数在 0.6–0.92 之间（错开一步就明显下降）；
在第 6,000 步这条谱系里，第 1 个 epoch 的 4,295 个抽样位置只出现过约 1,354 个，前约 300 个 batch 被训了大约 7 遍。

### 5.4 所有 rank 拿到相同的样本

上游的 `set_seed` 在所有 rank 上设的是同一个种子。温度采样不看采样器下标，于是每张卡抽到的样本、增广参数完全相同；
CUDA 随机数也一样，所以 flow matching 的噪声也相同。4×B200 上一步的 128 个样本里实际只有 32 个不同的；
AMD 的 8 卡续训更是 16 个不同样本 × 8 份拷贝。

阶段 2（22k+8k）用的是均匀任务采样，任务由采样器下标决定，各 rank 的任务不同，所以没有这个问题；只有任务内抽帧和增广参数的
随机序列在各 rank 间相同。阶段 2 也是一次跑完、没有续训，不受 5.3 影响。

修复（`#rank_seed`，在 [`dino/patch_dino.py`](../../prelim/dino/patch_dino.py) 和 [`trainer_stage3.diff`](../../prelim/patches/trainer_stage3.diff) 里）：
每次创建数据迭代器前，用（seed、数据并行 rank、全局步数、epoch）为每个 worker 和主进程（CPU 和 CUDA）重新设种子。
用真实的 `StatefulDataLoader` + `StatefulDistributedSampler` 在 CPU 上测：旧代码 rank 0 和 rank 1 完全相同，续训后的 batch 有一大半是重复的；
补丁后各 rank 不同、续训无重复，而且同一 checkpoint 重跑结果一致。阶段 3（22k+8k+5k）用的是修好后的训练器。

## 6. 评测在抢占分区上的工程

（概要见 [README 第 5.3 节](README.md#53-让评测在抢占队列上跑完)，代码 [`eval_pool.sbatch`](../../prelim/scripts/hpc/eval_pool.sbatch)。）

- **分阶段队列（PHASES）**：一个 worker 池可以排好几个阶段，例如“新 checkpoint 的 randomized 前 7 回合 → 它的 clean 前 7 回合 →
  旧 checkpoint 从 60 回合扩到 100 回合”。某个阶段可以指定自己的 checkpoint，并等待 `<ckpt>/.ready` 出现（拷贝并校验完才写），
  这样可以在 checkpoint 还没出来时就把 worker 排进队列，先做别的阶段。扩展回合数只要把阶段的回合数调大，每个任务从状态文件续跑。
- **squeue 太慢**：集群调度器最忙时一次 `squeue` 要 1–2 分钟。最初每检查一个被别人领走的任务就调一次，worker 在任务之间空等 10–30 分钟。
  改成每轮扫描只调一次。
- **领取竞争**：两个 worker 同时接手同一个无主任务时都以为自己拿到了。修复：接手后等 2 秒再读一次 owner；
  `squeue` 失败时不当作“领取者已不在运行”。
- **自己重新排队**：作业时间限制前 5 分钟收到 USR1，自己 `scontrol requeue`，任务从状态文件继续。
- **小请求更容易排上**：一个任务只要约 2 个 CPU 核、不到 33 GB 内存，所以 worker 用 `--cpus-per-task=4 --mem=48G --time=02:00:00` 这样的小请求。

## 7. 一次误判：GPU 上限

截止前一晚，几个 4 卡训练成员一直排不上，原因是 `QOSMaxGRESPerUser`（每用户 GPU 上限）。我们以为是评测 worker 占了名额，
写了一个监视脚本（[`gpu_cap_check.py`](../../prelim/scripts/hpc/gpu_cap_check.py)），在训练成员被这个原因卡住时停掉几个评测 worker，
一共停了 8 个。后来查清：常规分区的作业被分到另一个 QOS，那个 QOS 的每用户上限是动态调整的（同一晚我们看到过 16、0、2 三个不同的值），
评测 worker 在另一个 QOS 里，停掉它们对训练成员没有任何帮助。应该先用
`scontrol show assoc_mgr flags=qos` 看清各 QOS 的实时上限和自己的占用，再动手。
最后的解决办法是改提交 2 卡的成员，它们落在上限之内。

## 8. 其他

- **W&B 镜像**：训练配置里 `use_wandb: false`，另起一个 CPU 小作业把 TensorBoard 标量同步到 W&B，跟着 handoff 组的作业名存活
  （[`wandb_sync.sbatch`](../../prelim/scripts/hpc/wandb_sync.sbatch)、[`src/tb_to_wandb.py`](../../prelim/src/tb_to_wandb.py)）。凭据从集群上的 `~/.netrc` 读取，不在仓库里。
- **上传到 Hugging Face**：按提交包的目录结构整理 checkpoint（权重用硬链接），上传到私有仓库后逐个文件核对大小和 SHA-256
  （[`stage_and_upload.sbatch`](../../prelim/scripts/hpc/stage_and_upload.sbatch)）。我们用的仓库是私有 HF 仓库（未公开），脚本里用 `HF_REPO_PREFIX` 占位。
- **一个 B200 成员挂死的根因在 5.1**；另一个常被误判的日志：续训开始时每个 rank 会打印两条
  `Error detected in torch::autograd::CopySlices ... Traceback` 的 UserWarning，是无害的，别用 grep `Traceback` 判断作业失败。
