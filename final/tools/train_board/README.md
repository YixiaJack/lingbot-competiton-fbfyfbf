# 训练进度页（看板 /train）

地址：https://lingbot-board.vercel.app/train （GitHub 登录；GPU 看板左边栏"现场"分组里有入口）。

显示训练机 `~/lingbot_train/runs` 里正在跑的训练：步数、预计结束时间、下一个 checkpoint、各项 Loss 曲线、已保存的 checkpoint 和大小、
GPU 状态、磁盘还能存几个 checkpoint（不够时页面顶部报警）、数据和配置、历次运行的状态、自采数据准备（`~/field_training`）的状态。

## 怎么来的数据

- `train_board.py` 在队员电脑上常驻：用队员的 ssh 登录命令原样登录堡垒机，菜单里按下方向键到 001 再回车，保持这一个会话；
  每 2 分钟把 `train_probe.py` 送进训练机执行（只读，只用标准库和 PyYAML），取回 zlib+base64 的 JSON，校验大小和 SHA-256，
  推到私有仓库 `YixiaJack/lingbot-board-data` 的 **train 分支**（只有它写，force-push 单个提交）。会话断了会自动重连。
- 看板这边：`train-data.js`（即 `api/train-data.js`）读 train 分支的文件，`page_index.html`（即 `public/train/index.html`）是页面。

## 启停

```bash
bash ~/LingBotRoboTwin/train_board/train_board.sh start    # 也可以 stop / status / log
```

运行期间用 `caffeinate -i` 防止电脑空闲休眠；合上盖子还是会睡，这时页面顶部会提示"数据 N 分钟没更新"。

## 状态怎么判断

| 状态 | 条件 |
|---|---|
| 运行中 | 进程在，日志 10 分钟内更新过 |
| 正在保存 checkpoint | 进程在，当前步是保存步数的整数倍，日志 45 秒以上没更新 |
| 可能卡住 | 进程在，日志 10 分钟以上没更新 |
| 已完成 / 手动停止 / 出错停止 / 启动失败 | 进程不在：步数到头 / 日志里有 signal 15 / 日志末尾有报错（第 0 步就报错算启动失败） |

预计结束时间按最近 50 步每步耗时的中位数估算，不含存 checkpoint 的时间。
