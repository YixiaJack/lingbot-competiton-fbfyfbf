# 赛前 yolo11s 在官方 647 条真机数据上的检查（11/14）

**结论：不能直接用。** 能框到大部分物体，但类别基本认不对。任务 1 的目标物只有约 1/3 能认对，任务 2 不到 1/10，盘子颜色、收纳盒层、篮子 / 刷子 / 簸箕都会认错，黄纸团一个都没检出。看板 /field 上用这个检测器得到的"没检测到"类标记，在真机数据上大多是误报。

数据：公共数据集 `7adcd584-…/snapshots/210562dd-…/v1`（开发机 `/mnt/forge/datasets/public/` 下）。检测器：`yolo11s_sim.pt`（赛前只用仿真渲染图训练），默认阈值 0.25，输入是头部相机 `camera_top` 每条的第 0 帧。

## 数据集概况

| | 任务 1 放水果到盘子 | 任务 2 化妆品放进收纳盒 | 任务 3 扫纸团 |
|---|---|---|---|
| 条数 | 215 | 216 | 216 |
| 时长中位数（10%–90%） | 17.3 s（14.6–20.3） | 17.0 s（14.2–21.8） | 27.2 s（22.7–32.8） |
| 物体 | 34 种水果蔬菜，5 色盘子 | 8 种化妆品，3 层透明收纳盒 | 刷子、簸箕、白篮子，红 / 黄纸团 |

- LeRobot v3，30 fps，三路相机（`camera_top`、`camera_wrist_left`、`camera_wrist_right`），都是 640×480。状态拆成 `observation.state.arm.position`（12 个关节）和 `observation.state.effector.position`（2 个夹爪），**没有 14 维的 `observation.state`**，field_build.py 的轨迹分析要改列名才能用。
- 指令在 episodes 表的 `tasks` 列里，`meta/labels.jsonl` 里也有一份，附 `id`（如 `T1_train001`）。任务 3 有 82 条标了"部分成功"、1 条标了"成功"（`T3_train123` 起连续一段），其余没填。这个字段什么意思还没确认。
- 每条结束时机械臂都回到了初始位置，所以首末帧之间只有目标物变了。
- 指令里有两个说法检测器类别里原来没有：`mandarin`、`cherries`。已在 field_build.py 里分别映射到 `tangerine`、`cherry`。

## 怎么测的

没有人工标注的框，所以用首末帧差分自动得到真值：变化区域里，末帧变成空桌面的那一块就是目标物的起点（第 0 帧里的位置），另一块是它最后放下的地方。起点框太大（>20000 像素）或和终点框重叠的不计入（比如葡萄挪到紧挨着的盘子上，两块会粘在一起）。最后任务 1 留下 177 条，任务 2 留下 135 条。随机抽 12 条肉眼核对：8 条准确；1 条起点框偏大（黄瓜框进了旁边盘子的一角，保留在统计里）；1 条起点终点粘连（被上面的规则排除）；2 条没给出真值（自动排除）。见 [figures/gtcheck.jpg](figures/gtcheck.jpg)。

检测框和真值框 IoU ≥ 0.3，或者两个框的中心互相落在对方框里，就算框对了。

## 结果（阈值 0.25）

| | 任务 1（177 条） | 任务 2（135 条） |
|---|---|---|
| 目标物被框到了（不管叫什么） | 79% | 42% |
| 目标类别的框落在目标物上（LingBot 拿到的框是对的） | **33%** | **8%** |
| 目标类别的框落在别的物体上（框给错了） | 3% | 4% |
| 没有目标类别的框 | 63% | 87% |
| 目的地类别的框里装着最后放下的物体 | 40% | 11% |

阈值降到 0.1：任务 1 目标认对 36%、被框到 90%；任务 2 认对 12%、被框到 57%。降阈值只能多框到物体，认对的比例基本不变。

**任务 1 认错的样子**（[figures/vis_t1.jpg](figures/vis_t1.jpg)）：
- 全对或几乎全对：green_apple、tangerine、carrot、potato、grapes、banana、lemon；大多认对：mushroom、pumpkin、cherry、red_chili（约 2/3）。
- 认成别的：orange、persimmon、papaya 基本都叫成 tangerine，tomato、pomegranate、kiwi 也有一半左右叫成 tangerine；red_delicious、passion_fruit 叫成 red_apple；mangosteen 叫成 bell_pepper；bell_pepper 叫成 red_apple；onion 叫成 dragon_fruit；pear 叫成 onion；mango 叫成 lemon。peach、avocado、watermelon、strawberry 约一半没框到（peach 另一半被框成粉盘）。
- 盘子：蓝盘全部、紫盘几乎全部叫成 `plate_white`，黄盘一半叫成白盘，粉盘 2/3 认对。白盘全对，但那是因为它几乎把所有盘子都叫白盘。这个相机下紫盘偏灰或偏粉，蓝盘很浅。

**任务 2**：化妆品很小，眼影盘认对 10/23，其他基本是 0。粉扑都叫成 highlighter，大号刷子多叫成 mushroom，口红叫成 grapes 或没框到。收纳盒各层基本只有 `organizer_bottom` 有框。

**任务 3**（216 条，按框的形状和颜色统计，没有用差分真值）（[figures/vis_t3.jpg](figures/vis_t3.jpg)）：
- 篮子：162 条被叫成 `plate_white`；
- 刷子：159 条被叫成 `basket`；
- 簸箕：112 次认对，61 次叫成 `brush`；
- 红纸团 163/172 检出，黄纸团大约 190 个，**一个都没检出**（仿真里只有红、白两色纸团）。

## 复现

```bash
# 输出放在一个空目录里，在那个目录下运行；V 是数据集的 v1 目录
python -I extract.py $V .                                  # 每条第 0 帧和最后一帧 -> first/ last/，index.json
python -I detect.py yolo11s_sim.pt dets_sim.json first last # 阈值 0.01 全部保存
python -I stats.py dets_sim.json                           # 生成 dets_sim_rows.json：每条的目标类和目的地类
python -I diffgt.py                                        # 差分真值 -> diffgt.json
python -I evalgt.py dets_sim.json 0.25                     # 上面的表
python -I draw.py dets_sim.json out.jpg 216,217,218        # 画框看（红 = 目标类，黄 = 目的地类）
```

需要 ultralytics、opencv、pillow、av、pyarrow（就是 field 的 venv）。`diffgt.json` 和 `index.json` 已经放在这里：换一个检测器（比如自动标注后重训的版本，或者 YOLO-World）时，只要重跑 detect、stats、evalgt 三步，就能和上表直接对比。
