# robotharness 接进 5080 推理服务（11/14 晚）

- **作用**：任务 3 的推理会话里，`harness.Fixed` 每个动作块问一次 DeepSeek-V4.1-Flash（输入三路相机画面和夹爪读数），判断当前在哪个子任务，然后直接改策略适配器的 `_task`。下一个动作块就换成该子任务的固定指令，也就是训练用的那句。整个过程不停会话，也不需要平台介入。任务 1、2 不受影响。
- **开关**：`~/robotharness/mode`，可设 `off`（文件缺失也算 off）、`shadow`（只判断、记日志，不改指令）、`on`。每次开始推理时读一次，换档不用重启服务。
- **key**：`~/.deepseek_env` 里写 `DEEPSEEK_API_KEY=...`（权限 600），从本机钥匙串直接写过去，不经过仓库。
- **代码位置**：
  - `~/robotharness/code` 放 harness.py、real_split.py、prompts.py、skills.py，从 `../code` 原样复制；
  - `deps/openai` 是只用标准库写的替代客户端，因为推理环境里没有 openai 包；
  - `robotharness_bridge.py` 是桥接模块。
- **推理服务补丁**（`rh_server_patch.py`，叠在接管记录补丁之上）：
  - start() 在控制循环启动前，同步拿到第一个判断；
  - 开始后由后台线程持续判断；
  - stop() 时关掉 harness。

  每处都包了 try，harness 出错时策略保留原来的指令。补丁前的备份是 `~/dagger_patch/lingbot_infer_server.py.before_rh_20261114_200610`。
- **日志**：`logs/robotharness_<时间>_<会话>.jsonl`，记录开始、每次 LLM 调用（不含图像）、每次换指令、完成、出错和汇总。
- **测试**：`test_rh.py`，用自采 ep84 的四个时刻，离线跑，不碰正在运行的服务。
- **回滚**：先 `echo off > ~/robotharness/mode`；要彻底去掉，把备份拷回原路径，再按 dagger_trace README 里的方法重启服务（kill -9 MainPID 后由 systemd 拉起）。
