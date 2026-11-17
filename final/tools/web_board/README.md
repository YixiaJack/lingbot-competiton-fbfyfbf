# lingbot-board（Vercel）

https://lingbot-board.vercel.app ：集群 上的 crontab 每 2 分钟构建一次看板页面，推到私有仓库
`YixiaJack/lingbot-board-data` 的 `board_v2.html`（脚本 集群上的 `~/lingbot-board/runner.sh`）；这里的函数在运行时从仓库读出来返回。
推送频率随有没有人看自动调整：集群 的 cron 每 2 分钟醒一次，最近 10 分钟有人打开过看板就推，没人看就每 10 分钟推一次
（`runner.sh` 开头的判断；`FORCE=1 bash runner.sh` 强制推）。
页面需要登录：用 GitHub 账号登录，只放行 `ALLOWED_GITHUB_IDS` 里的 GitHub 用户数字 ID。

## 文件

```text
middleware.js        第一道检查（Routing Middleware，在 CDN 缓存之前）：没登录 -> /api/auth/login
api/board.js         看板页面；自己再验一次登录；private + no-store，CDN 不缓存
api/auth/login.js    跳到 GitHub 授权页（state + PKCE S256，不申请 scope）
api/auth/callback.js 换 token、查 GitHub 用户、核对允许名单、发 session cookie；用完即作废 GitHub token
api/auth/logout.js   清掉这台设备上的登录
api/field-data.js    现场作战台的数据：field 分支的 field_data.json（只给登录用户）
api/field-state.js   现场作战台的共享记录（自测打分、提交检查清单）：私有 Vercel Blob lingbot-field 的 field/state.json，写入带 ETag 防覆盖
api/field-img.js     现场作战台的图片：field 分支的 thumbs/<n>.jpg 和 bg_t<k>.jpg（只认这两种文件名）
api/train-data.js    训练进度页的数据：train 分支的 train_status.json（只给登录用户）
public/train/        训练进度页（/train）
lib/github_file.js   从数据仓库读文件（默认 field 分支，函数实例内缓存）
public/field/        现场作战台页面（/field）；静态文件也在 middleware 的登录检查之后
api/presence.js      给 集群 的 cron 用：最近一次有人打开看板是多少秒前（PRESENCE_TOKEN 鉴权，不走登录）
lib/session.js       session cookie：base64url(JSON).HMAC-SHA256(SESSION_SECRET)，7 天
lib/presence.js      看板每被打开一次就把时间记进 Vercel Runtime Cache（同区域函数共享，丢了只会让推送变慢）
public/robots.txt    只有 public/ 里的文件会被当静态文件发布（源码不会）
test/auth.test.mjs   node --test test/auth.test.mjs
```

## 环境变量（Production）

| 名字 | 内容 |
|---|---|
| `GITHUB_TOKEN` | 读数据仓库的 fine-grained token（只读、只对 lingbot-board-data） |
| `SESSION_SECRET` | session 签名密钥（随机 48 字节）；换掉它 = 所有设备都要重新登录 |
| `ALLOWED_GITHUB_IDS` | 允许登录的 GitHub 用户数字 ID，逗号分隔 |
| `PRESENCE_TOKEN` | 集群 查询 /api/presence 用的令牌；集群 上存在 `~/.config/lingbot-board/presence_token`（0600） |
| `GITHUB_OAUTH_CLIENT_ID` / `GITHUB_OAUTH_CLIENT_SECRET` | GitHub OAuth App（回调地址 `https://lingbot-board.vercel.app/api/auth/callback`） |

改了环境变量要重新部署才生效。

## 加人 / 删人

```bash
bash allow.sh ls                 # 现在谁能登录
bash allow.sh add <GitHub 用户名>  # 加人（自动重新部署）
bash allow.sh rm <GitHub 用户名>   # 删人：新部署上线后立即失效，已登录的设备也进不来
```

对方只需要有 GitHub 账号：打开网址，用自己的 GitHub 登录，第一次点一下"授权"。

## 部署

一定从这个目录部署（`vercel deploy --prod --yes`）。用别处没有登录的旧副本部署会去掉登录；不过旧代码读的是已经不再推送的
`board.html`，只会显示"取不到数据"，不会把看板公开出去。

## 现场作战台（/field）

决赛现场用的页面：数据覆盖地图和"下一条采什么"的建议、数据质量检查、评测场次和提交前检查、自测打分、规则速查。
GPU 看板左边栏最上面有"现场"分组（`api/board.js` 在返回前插进去，不用改 集群 的构建脚本；找不到左边栏时退回右下角浮动按钮）。

- 数据在私有仓库 `YixiaJack/lingbot-board-data` 的 **field 分支**（main 分支每 2 分钟会被 集群 force-push 覆盖，不能放那里）。
- 11/14 更新为真机 764 条：官方 647 条 + 自采好数据 117 条（任务一 59、任务二 32、任务三 26）。只接入指定好数据集的最新快照，成功／部分成功保留，失败、结果缺失和差质量排除。完整动作／状态哈希去重，稳定编号不复用。
- “位置统计”提供首帧 8×6 网格、来源/任务/目标筛选、目标/目的地候选/全部物体切换以及格子对应的样本。按格内独立记录数计数，兼容不同图像分辨率。当前用真实数据训练的 YOLO v2 自动估计位置，自采目标可定位 56/117 条，其余单列，不将漏检视为 0 次；未自动判定操作成功。背景图接口仅新增固定的 `bg_public_t[1-3].jpg`、`bg_private_t[1-3].jpg` 名称。
- 自动同步：`~/LingBotRoboTwin/final_real/field_board_watch.py` 通过 LaunchAgent `com.lingbot.field-sync` 常驻运行，每 30 秒检查快照；网页每 30 秒刷新。配置及日志在 `~/.cache/field_dev/watch`。采集电脑须开机登录并保持网络；部署后续数据只需推送 field 分支，无需重复部署网页。详见同步脚本目录的 `FIELD_BOARD_SYNC.md`。
- 真机导出：`LingBotRoboTwin/final_real/field_fetch_bastion.py --out NEW_DIR` 通过堡垒机调用只读的 `field_export_real.py`；传输验证 SHA-256。当前快照的句式分布另由 `field_prompt_policy.py` 生成 `prompt_policy.json` 并放入 `field_data.json.prompt_policy`。
- 发布真机数据使用 `field` 分支的普通提交和 fast-forward push；保留历史。旧 `field_sync.sh` 是仿真检测流程，会 force-push，不用于当前真机发布。
- “下一条怎么采”按任务记录本机已展示历史：任务一／二轮换“物体＋目的地”组合，本轮用完前不重复，在剩余组合里优先补自采缺口。任务二覆盖 10 类化妆品 × 3 层共 30 个组合；五种句式按官方权重 48/42/43/40/43 抽取，保留官方大小写。任务三按完整原句权重抽样并避开上一条相同 prompt，摆放建议独立轮换。历史保存在 localStorage，刷新可继续；切换任务不串历史，普通重绘／复制不消耗建议。展示建议不会增加真实采集计数。该任务的数据覆盖或模板策略变化时，自动按最新缺口开启新一轮；仅其他任务或导出时间变化时不会消耗建议。建议卡展示所依据的自采／官方／合计条数。
- 训练机 `~/field_training/field_prepare_training.py --work ~/field_training --watch` 每 60 秒扫描私有挂载。文字质量差、缺失、未知标签及不明确指令分别列入排除报告；数字评分需在 `config.json.minimum_score` 设置阈值后才接受。原始数据保留，输出按质量过滤、按官方比例改写的 LeRobot v3 副本、审计记录和 LingBot 两列数据列表 `train_lingbot.txt`。训练副本的任务、帧与统计量同步更新，视频引用原挂载。
- 私有目录为空时处于 `waiting_private_data`；先在平台建立私有快照并挂载到此训练实例。监听只准备数据，不启动 GPU 训练。作战台中的训练状态为导出时快照，显示检查时间。
- 自测打分和检查清单存在私有 Blob 存储 `lingbot-field`（11/13 用 `vercel blob create-store --access private` 建的，已连到本项目，环境变量 `BLOB_READ_WRITE_TOKEN` 自动加好）。`get()` 返回的是弱 ETag（`W/"…"`），条件写入要去掉前缀。

## 训练进度（/train）

训练机上正在跑的训练：步数和预计结束时间、下一个 checkpoint、各项 Loss 曲线（原始值 + EMA 平滑，Loss 跨度大时用对数坐标）、
已保存的 checkpoint 和大小、8 张卡的利用率 / 显存 / 温度 / 功耗、磁盘还能存几个 checkpoint（不够时页面顶部报警）、
这次用的数据和配置、`~/lingbot_train/runs` 下每一次运行的状态（运行中 / 可能卡住 / 已完成 / 手动停止 / 出错）、自采数据准备的状态。
GPU 看板左边栏"现场"分组里有入口。

- 数据：队员电脑上的 `~/LingBotRoboTwin/train_board/train_board.py` 常驻一个堡垒机会话（队员的 ssh 登录命令原样，菜单里按下方向键到 001 再回车），
  每 2 分钟把 `train_probe.py`（只读，标准库）送进训练机跑一次，取回 zlib+base64 的 JSON（校验大小和 SHA-256），
  推到私有仓库 **train 分支**的 `train_status.json`（这个分支只有它写，force-push 单个提交，不碰 field 和 main）。
- 启停：`bash ~/LingBotRoboTwin/train_board/train_board.sh start|stop|status|log`。运行期间 `caffeinate -i` 不让电脑空闲休眠；合上盖子仍会睡，页面会显示"数据 N 分钟没更新"。
- 页面每 60 秒取一次数据。代码副本在 notes 仓库 `tools/train_board/`。


## 评测难例生成（11/14）

任务一／二的“下一条怎么采”默认进入“评测难例”，各 10 组，可自动逐组轮换或指定一组。目标从该组物品中按当前自采／官方目标组合缺口选择，其他物品作为干扰物；指定组内目标×目的地轮换，不连续重复同一 prompt。原有“补齐覆盖”模式及其独立历史保留。

任务一保留两／三个盘子、长条横放、全橙色／全红色干扰；目标盘颜色由建议补选且一定出现在盘子清单内。第 1 组未记录物品，目标及干扰物明确标成补选；第 8 组未记录盘数，建议交替用两／三个并标明。第 2 组尖椒未注明颜色，红／青轮换。任务二保留收纳盒桌面位置与第 7 组左右分列，桌面左上／右上与目标 top／middle／bottom 层独立。

每组保留用户原始文字。暂定“百花果＝百香果、亮色／浅色＝高光盒、暗色／深色＝阴影盒”；第 10 组 lotion 与乳液合并为同一物品。这些解释在界面明确标注，后续可按团队确认修正。杨桃、青尖椒、小刷子、亮片盒支持生成，即使官方没有对应目标样例。

场景作为人工摆放安排，英文 prompt 保留官方大小写和五种句式权重（T1 41/43/46/43/42，T2 48/42/43/40/43）。可分别复制 prompt 或整份采集安排。20 条评测观察不加入训练记录、数量、成功率或位置统计；推荐中的计数仅对应目标×目的地，并不声称已统计整组难例覆盖。数据更新时重排组内缺口，自动场景遍历不重置；本机保存场景选择、两种模式及各任务轮换历史。

38 项页面与权限测试通过，并用 Chrome 实际点击验证全部 20 组、手选第 7 组和右上角收纳盒的底层目标。
