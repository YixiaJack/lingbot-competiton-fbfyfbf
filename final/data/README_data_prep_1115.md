# data_prep_1115：11/14 之后的数据处理代码（第二、三轮提交补充）

`data_prep/` 是第一轮（11/14）时的版本；这里是之后新增和修改的全部数据处理代码，按用途分目录：

| 目录 | 内容 |
|---|---|
| `batches/` | 每批自采数据的处理脚本（`own_<批次>.py` 普通示范：指令改写成官方模板、YOLO 框提示、抓取检查；`dagger_<批次>.py` / `dagger_seg.py` 接管录像只保留人工接管片段；`t3_<批次>.py` 任务三子任务切分、接管的"接手子任务 / 后续子任务"划分；`build_*.py` 生成数据集），以及每批的框提示记录、接管片段和人工核对结论（`*_cues.json`、`dagger_*_final.json`、`t3_*_final.json`） |
| `batches/seg_t3_dagger_908a.py`、`labels/` | 908a 批次任务三接管数据的切分脚本（在本地 Mac 上运行）及其标注文件：`labels_t3_dagger_908a.json`（全部接管片段）、`labels_t3_dagger_908a_takeover.json`（接手的子任务，按接管数据加权）、`labels_t3_followup_908a.json`（之后人完成的子任务，按普通示范加权）；`labels/` 由 `make_submit.sh` 从开发机 `~/lingbot_train/` 复制 |
| `autolabel/` | 带框数据集的生成（`derive_box_dataset.py`）、带框指令变体（`build_cue_prompts.py`：点名 / 不点名两种指令） |
| `sweep_harness/` | 数据集派生工具（`derive_dataset.py`）、任务三子任务切分（`real_split.py`） |
| `yolocue/`、`yolocue/relabel/` | 推理时画框的模块（与推理机上同一份代码）、YOLO 各版本的训练 / 列表 / 评测脚本 |
| `yolo_relabel/`、`yolo13/` | YOLO v8–v13 的人工核对标注流程：候选框、修正表（`fix*.json`，每条都写了原因）、最终标注、复制粘贴增强 |

所有自采数据都来自比赛现场的 Cobot Magic 采集（平台私有数据集 b60f1e2e / cfa641b0 / 105a2d09 的各个快照），处理在比赛期间完成；人工核对的结论都记在对应的 json 里。

第二轮提交的权重（总 11500 步，最后一段列表 v4q）用到的是 908a（批次 20）、7723（批次 21）及更早的批次。

第三轮提交的权重（总 15200 步，最后一段列表 v4za）用到 `batches/` 里的全部批次，包括 11/15 晚到 11/16 凌晨新采的 2c3d、e94f、c89c、e04f、4055、42da、2ffa、ad7e、dade、2218、4393、4f62、7928、0e70（v4t–v4za 各段）。新批次的脚本：

- `classify_<批次>.py`：区分普通示范和接管录像，接管录像只保留人工接管的片段（`dagger_seg.py`）；
- `own_<批次>.py`：任务一、二，指令改写成官方模板（`prompt_policy.json` 是改写用的模板表）、YOLO v13 框提示、抓取检查；
- `t3_<批次>.py`：任务三子任务切分（抓取 / 扫 / 倒 / 归位）；接管录像里人接手的那个子任务按接管数据加权，之后人完成的子任务按普通示范加权；
- `build_<批次>.py`：生成数据集。4f62 按队内要求去掉 ep41 的倒和归位两段；7928、0e70 全部保留（0e70 的 ep44 标为部分成功，人工看过结尾球已在篮内、工具已放下）。

每批的切分边界都由人工对照画面核对，结论写在对应的 `t3_<批次>_final.json` 里。
