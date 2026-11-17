# DAgger 接管痕迹（5080 推理服务补丁）

现场采 DAgger 数据时，人用主臂接管从臂、纠正后再交还给模型。录下来的 rollout episode 本身看不出哪一段是人在操作，所以在推理服务里加一个记录函数：推理每次开始、暂停、恢复、停止时，往 `~/workspace/lingbot/inference/logs/takeover_events.jsonl` 追加一行，并在服务日志里写一行 `Takeover trace: event=...`。

每行内容：

- `wall_time`：Unix 时间
- `monotonic`：服务时钟
- `event`：start / pause / resume / stop
- `session_id`、`task`、`task_type_id`
- `obs_sequence`、`obs_captured_at`：当时最新一帧观测的序号和采集时间
- `state`：当时全部 `observation.state.*` 的值

rollout 里从 pause 到下一个 resume 或 stop 之间的帧，就是人工接管的纠正数据。离线按 wall_time 加 state 快照，把这些帧和 episode 对齐（对齐脚本还没写）。

- 改动：`lingbot_infer/lingbot_infer_server.py` 新增 36 行，一个记录函数，加上 start / resume / pause / stop 四处各一行调用。函数从 mailbox 只读最新观测，不碰控制和推理，写失败只记一条异常日志，不会抛出。
- 安装：2026-11-14 16:36 安装。原文件 md5 是 436e9490…，备份在 5080 的 `~/dagger_patch/lingbot_infer_server.py.orig_20261114_163625`；打补丁后 md5 是 25e0d8bd…。
- 测试：`code/dagger_trace_test.py` 用假对象跑过；还缺一次真机验证。
- 重启：guest 没有免密 sudo。服务空闲时 `kill -KILL <MainPID>`，systemd 会按 `Restart=on-failure` 在 5 秒后用 `run.sh start` 重新拉起，服务仍归 systemd 管。不要用 SIGTERM：服务会正常退出，systemd 不会重启它。
- 回滚：把备份拷回原路径，再按上面的方法重启。
- 待验证：跑一次带“人工接管 / 退出接管”的 rollout，看 jsonl 里是否出现 pause / resume。如果平台接管时走的是 stop 再 start，记录里会是 stop / start，同样可用。如果什么都没出现，说明接管不经过推理服务，就改成按指令和实际状态的偏差来检测。

## 接管交还补丁（11/15 04:17 上线，[code/dagger_resume_patch.py](code/dagger_resume_patch.py)）

官方的 RESUME（从 PAUSED 状态 start）会在后移的时间轴上继续执行暂停前那段动作。人接管期间手臂如果被移走，恢复后会朝旧计划猛拉回去；在异步+RTC 下，下一次推理还会把这段过时的动作当作前缀。补丁改为：
- **丢掉旧计划**：恢复时 `action_buffer.clear()`，清掉当前动作段、排队的下一段、时间轴和上次发出的锚点；`pending_request_id` 和 `last_action_index` 也清零。暂停前发出的推理结果因为请求编号对不上，会被丢弃。
- **从当前姿态重新开始**：下一个控制周期像新会话开局一样，以当前观测推理，不带 RTC 前缀，第一段动作以当前姿态为锚点（交还时停顿约 0.45 秒）。
- **同步 yolocue**：调用 `YoloCue.on_resume()`，按当前画面重建框：
  - 目标已在盘子或收纳盒里 → 算放好，不画白框；
  - 目标在桌上别处 → 重新画框，重新等待抓取；
  - 看不到目标 → 视为已抓起，只保留橙框。

PAUSE 不变（官方语义是"冻结"，会继续发冻结的目标）。机器人主机在遥操作接管期间会不会忽略这些指令，需要在真机上验证。

## 自动识别接管（11/15 04:45 上线，[code/takeover_watch.py](code/takeover_watch.py) + [code/takeover_watch_patch.py](code/takeover_watch_patch.py)）

04:28 的真机测试（机器人主机录制 042628/042728）说明：**平台接管不给推理服务发 PAUSE**。策略在接管期间照常规划，规划和手臂偏差 0.43–0.61 rad；交还瞬间手臂以 7.19 rad/s 猛拉 0.5 rad。上面的交还补丁因为收不到 PAUSE/START，根本没被触发。

识别信号是机器人主机 robot_bridge 发布的 `/robby/schema/global/teleop_action` 的频率：
- 策略模式下，它逐条镜像我们发的 `/robby/schema/global/action`，30 次/秒，没有会话时是 0；
- 主手遥操作时，它就是主手指令，约 226 次/秒（接管第一秒 139 条，最后一秒 118 条）。

接管期间这个话题的内容，就是 bridge 正在发给从臂的指令（含夹爪开度）。

补丁做的事：
- **识别接管**：0.25 秒窗口内的频率 ≥100 次/秒并持续 0.1 秒，就调用 `pause()`（轨迹里记 pause）。
- **跟随**：暂停期间每个控制周期，把最新一条 teleop_action 换上新时间戳，原样发到 action 话题，不再发冻结的旧目标。所以交还时指令零跳变，夹爪也保持人手的夹紧指令（发实测开度会松掉夹着的东西）。
- **识别交还**：频率 <60 次/秒持续 0.3 秒，就调用 `start()`，也就是上面的交还逻辑（丢旧计划、从当前姿态重新推理、yolocue on_resume）。交还到恢复之间约 0.5 秒，teleop_action 已经变成我们自己指令的镜像，手臂原地保持。
- **平台自己的暂停**：平台已经暂停时发生接管，只跟随，不自动恢复，等平台 START。自动接管期间平台发 PAUSE，会被接受并转成平台暂停，同样不自动恢复。
- **开关**：在 5080 上 `touch ~/takeover_watch/off` 关闭（只挡新的接管；已经自动暂停的那次照常交还），`rm` 恢复。
- **参数**：`~/takeover_watch/config.json`，可设 on_hz / off_hz / window / on_hold / off_hold；`"auto_resume": false` 表示只暂停加跟随、恢复交给平台。
- **记录**：`~/takeover_watch/events.jsonl` 记 takeover_start / takeover_end，会话中每秒记一次频率；服务日志里搜 `takeover_watch:`。

验证：
- [code/test_takeover_watch.py](code/test_takeover_watch.py) 用录制的时间戳（`teleop_ts_20261115_0426.npy`）离线回放。接管开始后 +0.22 s 识别到，交还后 +0.53 s 识别到。20/50 ms 成批到达、丢 30%、加抖动时，也只有这两个事件。80 ms 的误碰不触发。
- [code/test_takeover_watch_ros.py](code/test_takeover_watch_ros.py) 在 5080 上用隔离域（ROS_DOMAIN_ID=77、localhost only）跑真实话题：30→226→30 次/秒，在 3.22 s / 6.53 s 识别到；跟随消息等于最后一条指令，时间戳是新的。
- [code/test_takeover_server.py](code/test_takeover_server.py) 在打过补丁的服务副本上跑 13 项状态检查：自动暂停、跟随、没有消息时退回冻结目标、自动恢复、平台 PAUSE/START 的各种组合、auto_resume=false、无会话。

回滚：`~/dagger_patch/lingbot_infer_server.py.before_tw_20261115_044500` 拷回 `code/deploy/lingbot_infer/lingbot_infer_server.py`，再执行 `~/rtc/switch.sh rtc`（服务空闲时）。

待真机复测：接管一次再交还，看 events.jsonl 和服务日志，以及交还时有没有抽动。接管前平台还有约 3–4 秒"对齐"阶段（从臂冻结，频率仍是 30），这段识别不到。但交还时旧计划会被整体丢掉，不影响交还。

数据建议（参考 HG-DAgger / Sirius / RaC）：
- 只用人工接管段，加高权重；
- 接管前 1–2 秒不用；
- 推荐 RaC 式操作：先把机器人退回熟悉的状态，再完成当前子任务，然后结束这一条；
- 用 `takeover_events.jsonl` 里记录的关节状态，去机器人主机的录制数据里找吻合的帧来对齐。
