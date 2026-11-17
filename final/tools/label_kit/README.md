# 决赛现场 YOLO 标注工具包

流程：抽头顶相机帧 → 预标注 → X-AnyLabeling 人工修正 → 质检 → 切分 → 微调 → 验收。
依赖：`extract_frames.py` 要 pyarrow + (torchcodec 或 ffmpeg)，开发机 lingbotvla 环境里都有；
`prelabel.py` 要 ultralytics；`make_split.py` 和 `qa_gallery.py` 只要 Pillow。

## 0. 先确认数据结构（拿到公开数据第一件事）

```bash
python extract_frames.py --inspect /data/train/ds/<快照id>/v1
```
看 `codebase_version`、`video_path`、相机键名、episodes 的字段。脚本按 LeRobot v3.0 / v2.1 的标准结构写，
这里没有真数据测过，如果字段名对不上，按 `--inspect` 打出来的改 `video_location()` 一处即可。

## 1. 抽帧

```bash
python extract_frames.py --out frames/ /data/train/ds/<公开快照>/v1 /data/train/ds/<自采快照>/v1
```
- 默认每条 episode 抽第 0 帧；想多要几种画面可以 `--frames 0 15`（第 15 帧手臂还没进画面时）。
- 输出 `frames/manifest.csv`，带指令和 `task_type`（t1/t2/t3）。
- 推理机上用 `code/deploy/docs/CUSTOM_INFERENCE.md` 第 4 节的只读观测采样抓到的图，也可以放进 `frames/`，
  在 manifest 里手动补一行（dataset 写 `live`，episode_index 随便编个不重复的号）。
- 同一个摆法录了好几条时，在 manifest 的 `layout` 列填同一个编号，切分时加 `--group-col layout`。

## 2. 预标注

```bash
python prelabel.py --weights best.pt --images frames/ --out labels/ --conf 0.25
```
- `best.pt` 用仿真训的 YOLO（**前提是群里确认可以带进场**）。不能带的话用 YOLO-World 预标：
  `--weights yolov8s-worldv2.pt --world-classes classes_en.txt`。
- 输出 `labels/*.txt` 和 `labels/classes.txt`。

## 3. X-AnyLabeling 修正

`pip install x-anylabeling-cvhub[cpu]`（或 `[cuda12]`），打开 `frames/`，导入 YOLO 格式标注（选 `labels/classes.txt`），
逐张修正后导出 YOLO 格式到 `labels/`。也可以直接在软件里加载自定义 YOLO 模型一键预标，见它的 `docs/en/custom_model.md`。

## 4. 质检

```bash
python qa_gallery.py --images frames/ --labels labels/ --classes labels/classes.txt --out qa/
```
- `qa/crops/` 每类一张裁剪图墙：同一面墙里混进别的东西，一眼就能看出来，按格子下面的 `#编号` 回去改。
- `qa/sheets/` 原图加框的缩略图拼图，查漏标。
- `qa/report.txt`：每类数量（0 个样本的类会标出来）、没标注的图、过小框、重复框、两个类别几乎重合的框。
- 裁剪图墙可以发给 Claude 帮忙查。

## 5. 切分（和 LingBot 验证集共用留出条目）

```bash
python make_split.py --frames frames/ --labels labels/ --classes labels/classes.txt \
    --out yolo_ds/ --test-frac 0.25 --holdout holdout.txt
```
`holdout.txt` 每行 `数据集名 episode_index`，就是 LingBot 验证集那几条，它们一律进测试集，两边训练都不用。
按 episode（或 layout）分组、按任务分层，不会出现同一布局同时在训练和测试里。

## 6. 微调和验收

```bash
yolo detect train data=yolo_ds/data.yaml model=best.pt imgsz=640 epochs=80 batch=32
yolo detect val   data=yolo_ds/data.yaml model=runs/detect/train/weights/best.pt
```
`imgsz=640` 和 `experiments/real_detector/code/train_yolo_real.py` 一致。没有可带的仿真权重时 `model=yolo11s.pt`（COCO 预训练）。
mAP 只是参考，开不开框看"按指令选框"在测试集上的正确率（目标框和目的地框都对，用 `cues_real.py` 的
`parse_instruction` + `select_boxes`，和仿真评估同一套），以及易混组各自的正确率。建议门槛 95% 左右，低置信度不画框。

## 标注规范（开标前全队对齐）

1. **类别名和顺序完全沿用仿真 YOLO**，也就是 `experiments/real_detector/code/catalog.py` 里的 `CLASSES`（59 类），
   不改名不重排，否则微调和 `cues_real.py` 的选框映射都会错。本目录的 `classes.txt` 就是从它导出的；catalog 改了要重新导出：
   ```bash
   REAL_ASSETS=experiments/real_detector/assets_meta python3 -c "import sys; sys.path.insert(0,'experiments/real_detector/code'); from catalog import CLASSES; print('\n'.join(CLASSES))" > tools/label_kit/classes.txt
   ```
2. **图里每个属于这 59 类的物体都要标**，不只标目标和目的地。漏标的物体会被当成背景，召回会掉。
3. 框贴紧**可见部分**；被遮挡的只框露出来的部分，露出不到约 1/3 的可以不标（全队统一）；被画面边缘截断的框可见部分。
4. 盘子按颜色分类；同色的两个盘子都要标，都是同一类。
5. 收纳盒按层分三类 `organizer_bottom / organizer_middle / organizer_top`，和仿真一致：仿真里每层是单独一个格子实体，
   框来自它的分割，所以**框的是这一层格子（底板加围墙）的可见范围**，不是只框开口。实物和仿真假设的形状不一样时，
   全队先对着实物定一次，截图存档，所有人按同一个定义标。盒子里放着的化妆品也要标。
6. 机械臂、夹爪、桌布不标。
7. 易混组按 `catalog.py` 的 `CONFUSE` 逐组对照实物确认，有疑问的拍照存档统一判。最容易错的：
   `tangerine`/`orange`、`red_apple`/`red_delicious`、`brush_large`/`brush_small`、
   `highlighter`/`contour`/`glitter_compact`、`lotion_bottle`/`lip_gloss`。质检时这几类的裁剪图墙重点看。
8. 任务 3 的 `brush` 是扫把，任务 2 的化妆刷是 `brush_large` / `brush_small`，别标混。
