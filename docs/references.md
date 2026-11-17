# 参考资料

## 赛事与基座模型

- LingBot-VLA 2.0 官方仓库：<https://github.com/robbyant/lingbot-vla-v2>
  - Cobot Magic 配置：`configs/robot_configs/agilex_cobot_magic.yaml`
  - 真机后训练配置：`configs/vla/real_robot/real_robot.yaml`
  - 训练参数说明（含 4×A6000 示例和显存）：`configs/vla/Training_Config.md`
  - 自定义数据说明：`lingbotvla/data/vla_data/README.md`
  - RoboTwin 评测说明（BF16 / FP32 差异）：`experiment/robotwin/README.md`
- LingBot-VLA 2.0 模型权重：<https://huggingface.co/robbyant/lingbot-vla-v2-6b>
- LingBot-VLA 2.0 论文 *From Foundation to Application: Improving VLA Models in Practice*：<https://arxiv.org/abs/2607.06403>
- LeRobot 版本：<https://huggingface.co/lerobot/lingbot_vla_v2_base>
- 初赛落幕报道（每日经济新闻）：<https://www.nbd.com.cn/articles/2026-10-28/4623646.html>

## 目标提示与语言对齐

- Point-VLA *Point What You Mean: Visually Grounded Instruction Policy*：<https://arxiv.org/abs/2512.18933>
- ECoT *Robotic Control via Embodied Chain-of-Thought Reasoning*：<https://arxiv.org/abs/2407.08693>
- ECoT-Lite：<https://arxiv.org/abs/2505.08243>

## 训练方案与 MoE 路由

- MoE-Adapters *Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters*（CVPR 2024）：<https://arxiv.org/abs/2403.11549>；知乎解读：<https://zhuanlan.zhihu.com/p/1915756018090119546>
- HG-DAgger *Interactive Imitation Learning with Human Experts*：<https://arxiv.org/abs/1810.02890>
- DeepSeek-V3 技术报告（无辅助损失均衡、序列级均衡损失，LingBot 的 MoE 沿用了这两点）：<https://arxiv.org/abs/2412.19437>

## 仿真与数据集（决赛不允许用外部数据，这些只作背景）

- RoboTwin 2.0：<https://arxiv.org/abs/2506.18088>，LeRobot 文档 <https://huggingface.co/docs/lerobot/robotwin>
- InternData-A1：<https://arxiv.org/abs/2511.16651>，<https://huggingface.co/datasets/InternRobotics/InternData-A1>
- RoboMIND：<https://arxiv.org/abs/2412.13877>
- MolmoBot：<https://allenai-web.allen.ai/blog/molmobot-robot-manipulation>
- EmbodiChain（支持 CobotMagic 的仿真数据生成框架）：<https://dexforce.github.io/EmbodiChain/main/tutorial/data_generation.html>

## 硬件

- RTX PRO 5000 Blackwell：<https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000>
- 松灵 Cobot Magic：<https://global.agilex.ai/products/cobot-magic>
