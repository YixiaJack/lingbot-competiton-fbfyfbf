"""Append the normalization-statistics notice to the README of our Hugging Face model repos (README.md only).

  HF_HOME=... python scripts/hpc/hf_readme_normstats.py <hf-user>/repo1 <hf-user>/repo2 ...
"""

import sys

from huggingface_hub import HfApi, hf_hub_download

MARK = "<!-- norm-stats-notice -->"
NOTE = f"""

{MARK}
## 归一化统计量（评测前必读）

- 本模型训练用的归一化统计量是 `02_代码材料/configs/norm_stats/robotwin_clean.json`
  （SHA256 `92b37257234b37c8…`）。它**只用** RoboTwin 2.0 Aloha-AgileX 50 个任务的 clean 数据（每任务 50 条）计算，
  用官方脚本 `scripts/compute_norm_stats.py` 生成，归一化方式为 `bounds_99_woclip`（按 1% / 99% 分位数 q01 / q99 缩放）。
- **它和官方的 `assets/norm_stats/robotwin.json` 不同**：官方文件是用 clean + randomized 数据一起算的。
  部分机械臂关节的 q01 / q99 相差 0.2 到 0.29 rad。例如第 6 维的 q99：官方 1.600，我们 1.471；
  第 7 维的 q99：官方 0.995，我们 1.195。
  比赛规定 randomized 数据不得以任何形式参与训练，所以我们没有使用官方统计量。
- **推理和评测时必须使用这份统计量**：把 `03_模型权重/checkpoint/<run>/lingbotvla_cli.yaml` 中的
  `data.norm_stats_file`（以及 robot config 里的 `norm_stats`，如果用到）改成本文件在你机器上的路径。
  yaml 里其他 `$SCRATCH/...` 路径是训练集群的路径，推理时只需要统计量文件和 Qwen3-VL tokenizer（环境变量 `QWEN3VL_PATH`）有效。
- 如果换成官方 `robotwin.json`，模型输出的动作会被系统性地换算错。实测同一个模型（40k+10k）的成功率，
  从 clean 约 77% / randomized 约 40%，掉到 9% / 13.6%。
"""


def main():
    api = HfApi()
    for repo in sys.argv[1:]:
        path = hf_hub_download(repo, "README.md")
        text = open(path, encoding="utf-8").read()
        if MARK in text:
            print(repo, "already has the notice")
            continue
        api.upload_file(path_or_fileobj=(text.rstrip() + NOTE).encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo, commit_message="README: normalization statistics notice (clean-only, differs from official)")
        print(repo, "README updated")


if __name__ == "__main__":
    main()
