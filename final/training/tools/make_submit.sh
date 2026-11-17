#!/bin/bash
# make_submit.sh ROUND [HF_CKPT] [LIST] — fill /workspace/submit_result/<ROUND> (first|second|third) with the training
# code, the weights, data samples and the inference code of the submitted model. Built on the local disk first, then
# copied to /workspace (OSS mount). log: /data/lingbot/submit_<ROUND>.log
# 11/15 changes (second/third):
#  - HF_CKPT and LIST must be given explicitly (the round-1 defaults v3_step2000 / v3.txt only apply to "first").
#  - inference_code/ is the snapshot of what actually runs on the 5080, made on the Mac with
#    collect_5080_inference.sh ROUND (notes repo experiments/train_real/tools) and uploaded to the OSS workspace:
#    /workspace/上传OSS_inference_<ROUND>/lingbot_inference_<ROUND>.tar + .md5 (INF_TAR=... overrides the path).
#    Missing snapshot = abort; the old copy of code_v4s/deploy is NOT what the 5080 runs and is only kept for "first".
#  - checked before anything is copied: the tar's md5, and that the weights loaded on the 5080 (MANIFEST.json: per-file
#    sizes, source global_step) are HF_CKPT. ALLOW_WEIGHT_MISMATCH=1 skips the weight check.
#  - YOLO weights: uploaded ONCE to /workspace/上传OSS_yolo_weights (YOLO_DIR=... overrides); the file the 5080 runs
#    (MANIFEST yolocue.weights_md5) is copied into inference_code/team_inference_side/yolocue/weights/. Not found = abort.
#  - README: submit_src/README_<ROUND>.md if present, else a generated stub (WARN in the log); the snapshot summary
#    (loaded model, configs, patched files, what is not included) is appended either way.
#  - training code: code_v4r for second/third (every v4 launch from 11/14 22:31 on, i.e. the whole chain of the submitted
#    weights, ran CODE=code_v4r = code_v4s + p10 training-time RTC; TRAIN_CODE=... overrides); code_v4s for "first".
#  - data_prep_1115/ (data-processing code written after 11/14): data_prep_1115.tar next to the inference snapshot
#    (/workspace/上传OSS_inference_<ROUND>/) plus the task-3 takeover label files ~/lingbot_train/labels_t3_*.json.
#  - logs/ (second/third; the guide asks for process logs): /data/lingbot/runs/v4 without checkpoints (launch records,
#    every segment's train.log, TensorBoard events) and ~/lingbot_train/logs (list generation, switch scripts).
#  - before copying to /workspace: abort if anything outside weights/ and data_samples/ looks like a credential.
#  - 11/16 re-runs are incremental: weights already in /workspace with the same SHA-256 per shard are not copied again,
#    and the rest is synced by a small Python step (samples by size, code / logs by md5; only changes are written).
#  - PART=data: like base without the weights (HF_CKPT may be "-"): pre-copies data samples, training code and logs
#    before the round's checkpoint exists; the final PART=base run then only adds the weights and what changed.
#  - PART=base (second/third): everything except inference_code/ and data_prep_1115/ (no snapshot needed), so the slow
#    weight copy can run while the inference side is still changing. PART=inf: only checks the snapshot and replaces
#    inference_code/, training_code/data_prep_1115/ and README.md in /workspace/submit_result/<ROUND>. Default PART=all.
#  - PART=weights (11/16 04:0x, for the last-minute checkpoint swap): only the weights (skipped if /workspace already
#    has the same SHA-256), logs/ (compared by size: rotated logs never change, growing ones change size), the train
#    config of the weights and README.md. Data samples and training code stay as the last data/base run left them.
#    README = submit_src/README_<ROUND>.md + the inference section of the last PART=inf staging, if any.
#  - the final size line is taken from the local staging (du over the OSS mount took 3.5 min).
#  - README_SRC=<file> overrides submit_src/README_<ROUND>.md.
#  - weights go straight from the checkpoint mount to /workspace/submit_result/<ROUND>/weights (no local copy); every
#    shard's SHA-256 is computed on both sides and must match. All tar copies use 1 MiB records (-b 2048): with tar's
#    default 10 KiB records the ossfs2 mounts read at ~1 MB/s, with 1 MiB records at ~150 MB/s.
R=${1:?round: first, second or third}; CK=${2:-}; LIST=${3:-}
case $R in first|second|third) ;; *) echo "round must be first, second or third"; exit 1 ;; esac
if [ $R = first ]; then CK=${CK:-/data/lingbot/init/v3_step2000_hf}; LIST=${LIST:-$HOME/lingbot_train/lists/v3.txt}; fi
[ -n "$CK" ] && [ -n "$LIST" ] || { echo "round $R: give HF_CKPT and LIST explicitly: make_submit.sh $R <hf_ckpt dir> <list>"; exit 1; }
W=$HOME/lingbot_train; D=/data/lingbot; L=$D/submit_stage/$R; S=/workspace/submit_result/$R; X=$D/submit_stage/inf_$R
INF=${INF_TAR:-/workspace/上传OSS_inference_$R/lingbot_inference_$R.tar}
YW=${YOLO_DIR:-/workspace/上传OSS_yolo_weights}     # YOLO weights, uploaded once; the one the 5080 runs is picked by md5
exec 3>&2; exec >> $D/submit_$R.log 2>&1
log() { echo "== $(date '+%F %T') $*"; }
die() { log "ABORT: $*"; echo "make_submit $R ABORTED: $* (log: $D/submit_$R.log)" >&3; exit 1; }
copy() { mkdir -p "$2" && tar -b 2048 -C "$1" --exclude=__pycache__ --exclude='*.pyc' -chf - . | tar -b 2048 -C "$2" -xmf - --no-same-owner; }   # -m: ossfs2 refuses utime
sha() { (cd "$1" && for f in *.safetensors; do echo "$(dd if="$f" bs=16M status=none | sha256sum | cut -d' ' -f1)  $f"; done); }
log "round $R: weights $CK, data list $LIST, inference snapshot $INF"
[ "${PART:-all}" = data ] || [ -f "$CK/config.json" ] || die "no checkpoint at $CK"
[ -f "$LIST" ] || die "no data list at $LIST"
PART=${PART:-all}
case $PART in all|base|inf|data|weights) ;; *) die "PART must be all, base, inf, data or weights" ;; esac
NOINF=0; case $PART in base|data|weights) NOINF=1 ;; esac
[ $PART = all ] || [ $R != first ] || die "PART=$PART is only for second/third"
log "PART=$PART"
[ $PART != inf ] || [ -f $L/weights/SHA256SUMS ] || die "no staging at $L: run PART=base first"
[ $PART != weights ] || { [ -f $L/README.md ] && [ -d $L/training_code ]; } || die "no staging at $L: run PART=data or base first"

NEW_INF=0
[ $PART = weights ] && [ -f $L/inference_code/MANIFEST.json ] && NEW_INF=1   # keep the inference section of the last PART=inf
if [ $NOINF = 0 ] && [ -f "$INF" ]; then
  NEW_INF=1
  want=$(cut -d' ' -f1 "$INF.md5" 2>/dev/null); got=$(md5sum "$INF" | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || die "md5 of $INF is $got but $INF.md5 says '$want' (upload incomplete?)"
  rm -rf $X && mkdir -p $X && tar -C $X -xf "$INF" && [ -f $X/inference_code/MANIFEST.json ] || die "cannot unpack $INF"
  python3 - $X/inference_code/MANIFEST.json "$CK" <<'EOF'
import json, os, re, sys
m = json.load(open(sys.argv[1])); ck = sys.argv[2]; w = m.get('loaded_model', {})
print(f"5080 snapshot {m.get('collected_at')}: loaded {w.get('path')} from {w.get('source')}")
print(f"  inference.yaml={m.get('inference_yaml')} yolocue={m.get('yolocue', {}).get('mode')} {m.get('yolocue', {}).get('weights_file')} "
      f"cue_prompt={m.get('yolocue', {}).get('config', {}).get('cue_prompt', False)} robotharness={m.get('robotharness', {}).get('mode')} "
      f"scheduler={m.get('robotharness', {}).get('scheduler')}")
bad = []
if not w.get('files'):
    bad.append('MANIFEST lists no weight files')
for f in w.get('files', []):
    p = os.path.join(ck, f['name'])
    if not os.path.exists(p):
        bad.append(f"{f['name']} is not in HF_CKPT")
    elif os.path.getsize(p) != f['size']:
        bad.append(f"{f['name']}: {os.path.getsize(p)} bytes in HF_CKPT, {f['size']} on the 5080")
a = re.search(r'global_step_(\d+)', w.get('source') or ''); b = re.search(r'global_step_(\d+)', os.path.realpath(ck))
if a and b and a.group(1) != b.group(1):
    bad.append(f'the 5080 runs global_step_{a.group(1)}, HF_CKPT is global_step_{b.group(1)}')
for x in bad:
    print('WEIGHT MISMATCH:', x)
print('weights: HF_CKPT matches what the 5080 had loaded' if not bad else 'weights: MISMATCH')
sys.exit(1 if bad else 0)
EOF
  [ $? = 0 ] || [ "${ALLOW_WEIGHT_MISMATCH:-0}" = 1 ] || die "the weights loaded on the 5080 are not $CK (details in the log; ALLOW_WEIGHT_MISMATCH=1 to override)"
  python3 - $X/inference_code/MANIFEST.json "$YW" $X/inference_code/team_inference_side/yolocue/weights <<'EOF' || die "YOLO weights the 5080 runs are not in $YW (upload them there once: ~/Desktop/上传OSS_yolo_weights)"
import hashlib, json, os, shutil, sys
m = json.load(open(sys.argv[1])); src, dst = sys.argv[2], sys.argv[3]
want, name = m['yolocue']['weights_md5'], os.path.basename(m['yolocue']['weights_file'] or '')
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
files = sorted(os.listdir(src)) if os.path.isdir(src) else []
cands = [f for f in files if f == name] + [f for f in files if f != name and f.endswith(('.torchscript', '.pt'))]
hit = next((os.path.join(src, f) for f in cands if md5(os.path.join(src, f)) == want), None)
if not want or not hit:
    print(f'YOLO weights {name} (md5 {want}) not found in {src}'); sys.exit(1)
os.makedirs(dst, exist_ok=True); shutil.copy2(hit, os.path.join(dst, name)); got = [name]
stem = os.path.splitext(name)[0]
for extra in (stem + '.pt', 'MD5SUMS', 'README.txt'):
    if os.path.isfile(os.path.join(src, extra)) and extra != name:
        shutil.copy2(os.path.join(src, extra), os.path.join(dst, extra)); got.append(extra)
print('YOLO weights copied (md5 %s matches the 5080):' % want, ', '.join(got))
EOF
elif [ $R != first ] && [ $NOINF = 0 ]; then
  die "no inference snapshot at $INF: run collect_5080_inference.sh $R on the Mac, upload ~/Desktop/上传OSS_inference_$R to the OSS workspace"
fi

if [ $PART != inf ] && [ $PART != weights ]; then
rm -rf $L && mkdir -p $L/training_code $L/weights $L/data_samples $L/inference_code
TC=$W/code_v4s; [ $R = first ] || TC=$W/${TRAIN_CODE:-code_v4r}
[ -d $TC ] || die "no training code at $TC"
log "training code: $TC"
copy $TC $L/training_code/lingbot-vla-v2_patched
copy $W/tools $L/training_code/tools
for d in configs lists rc; do copy $W/$d $L/training_code/$d; done
copy $W/submit_src/data_prep $L/training_code/data_prep
fi
if [ $R != first ] && [ $NOINF = 0 ]; then
  rm -rf $L/training_code/data_prep_1115
  DP=$(dirname "$INF")/data_prep_1115.tar
  if [ -f "$DP" ]; then
    tar -C $L/training_code -xf "$DP" --no-same-owner && mkdir -p $L/training_code/data_prep_1115/labels && cp $W/labels_t3_*.json $L/training_code/data_prep_1115/labels/ \
      && log "data_prep_1115 added ($(find $L/training_code/data_prep_1115 -type f | wc -l) files)" || log "WARN: data_prep_1115 incomplete"
  else
    log "WARN: no $DP, data_prep_1115 not added"
  fi
  GS=$(dirname "$INF")/graspnet_src.tar      # 11/16: training code of the harness grasp-check CNN (model file is in the snapshot)
  rm -rf $L/training_code/graspnet
  GU=$(python3 -c "import json,sys; print(int(bool(json.load(open(sys.argv[1])).get('robotharness', {}).get('graspnet_used'))))" $X/inference_code/MANIFEST.json 2>/dev/null)
  if [ "$GU" != 1 ]; then log "graspnet not used by the snapshot harness: training code not added"
  elif [ -f "$GS" ]; then tar -C $L/training_code -xf "$GS" --no-same-owner && log "graspnet training code added"; else log "no $GS (graspnet training code not added)"; fi
fi
if [ $PART != inf ]; then
if [ $PART = data ]; then
  log "weights: skipped (PART=data)"
else
cp $CK/lingbotvla_cli.yaml $L/training_code/train_config_of_submitted_weights.yaml
log "weights: $CK -> $S/weights/hf_ckpt"
sha $CK > $L/weights/SHA256SUMS
if [ -d $S/weights/hf_ckpt ] && [ "$(ls $S/weights/hf_ckpt | wc -l)" = "$(ls $CK | wc -l)" ] && { [ ! -f $S/weights/SHA256SUMS ] || cmp -s $S/weights/SHA256SUMS $L/weights/SHA256SUMS; } \
   && sha $S/weights/hf_ckpt > $L/weights/SHA256SUMS.dst \
   && cmp -s $L/weights/SHA256SUMS $L/weights/SHA256SUMS.dst; then
  log "weights already in $S with identical SHA-256: not copied again"
else
  mkdir -p $S/weights && rm -rf $S/weights/hf_ckpt && { copy $CK $S/weights/hf_ckpt || log "WARN: tar reported errors copying the weights (checked by SHA-256 below)"; }
  sha $S/weights/hf_ckpt > $L/weights/SHA256SUMS.dst
fi
[ -s $L/weights/SHA256SUMS ] && cmp -s $L/weights/SHA256SUMS $L/weights/SHA256SUMS.dst || die "weights in $S/weights/hf_ckpt differ from $CK (see $L/weights/SHA256SUMS*)"
rm -f $L/weights/SHA256SUMS.dst
log "weights verified (SHA-256 source = /workspace): $(wc -l < $L/weights/SHA256SUMS) shards, $(du -sh $S/weights/hf_ckpt | cut -f1), $(ls $S/weights/hf_ckpt | wc -l) files (source $(ls $CK | wc -l))"
fi
if [ $PART != weights ]; then
log "data samples"
python3 $W/tools/make_samples.py $L/data_samples $LIST 2 || log "samples FAILED"
fi
if [ $R != first ]; then
  log "process logs"
  rm -rf $L/logs; mkdir -p $L/logs/v4_train $L/logs/data_and_switch
  tar -C $D/runs/v4 --exclude=./checkpoints --exclude=./pid -cf - . | tar -C $L/logs/v4_train -xf - --no-same-owner
  find $W/logs -maxdepth 1 -type f ! -name '*.pid' -exec cp -p {} $L/logs/data_and_switch/ \;
  log "logs: $(du -sh $L/logs | cut -f1)"
fi
fi
if [ $NOINF = 1 ]; then
  log "inference code: skipped (PART=$PART)"
elif [ $NEW_INF = 1 ]; then
  log "inference code"
  rm -rf $L/inference_code
  copy $X/inference_code $L/inference_code
  cp $W/code_v4s/scripts/open_loop_eval.py $L/inference_code/ 2>/dev/null || log "WARN: no open_loop_eval.py"
  for f in robot_config.yaml norm_states.json; do cp $CK/$f $L/inference_code/ 2>/dev/null || log "WARN: $f not in $CK"; done
else    # round 1 exactly as submitted on 11/14
  log "inference code"
  copy $W/code_v4s/deploy $L/inference_code/deploy
  cp $W/code_v4s/scripts/open_loop_eval.py $CK/robot_config.yaml $CK/norm_states.json $L/inference_code/
  copy $W/submit_src/inference_side $L/inference_code/team_inference_side
fi
RSRC=${README_SRC:-$W/submit_src/README_$R.md}     # README_SRC=...: a variant README (e.g. README_third_15000.md for a fallback checkpoint)
if [ -f $RSRC ]; then
  cp $RSRC $L/README.md; log "README from $RSRC"
elif [ $R = first ]; then
  cp $W/submit_src/README.md $L/README.md
else
  log "WARN: no submit_src/README_$R.md, writing a stub README (fill in the training history before the round)"
  printf '# %s 轮评测提交\n\n权重：`weights/hf_ckpt/`，来自 `%s`（各分片 SHA-256 见 `weights/SHA256SUMS`）。数据列表：`training_code/lists/%s`。\n\n**TODO：训练经过（起点、各阶段步数和数据列表）。**\n' \
    "$R" "$CK" "$(basename $LIST)" > $L/README.md
fi
if [ $NEW_INF = 1 ]; then
  python3 - $L/inference_code/MANIFEST.json "$CK" >> $L/README.md <<'EOF'
import json, os, re, sys
m = json.load(open(sys.argv[1])); w = m['loaded_model']; y = m['yolocue']; h = m['robotharness']
def overall(ck):   # submitted checkpoint as an OVERALL step: runs/<run>/checkpoints/global_step_N + runs/<run>/step_offset
    r = re.search(r'/runs/([^/]+)/checkpoints/global_step_(\d+)', os.path.realpath(ck or ''))
    if r:
        try: off = int(open(os.path.expanduser(f'~/lingbot_train/runs/{r.group(1)}/step_offset')).read().strip() or 0)
        except (OSError, ValueError): off = 0
        return int(r.group(2)) + off
    r = re.search(r'global_step_(\d+)', ck or ''); return int(r.group(1)) if r else None
sub = overall(sys.argv[2] if len(sys.argv) > 2 else ''); ld = re.search(r'global_step_(\d+)', w.get('source') or ''); ld = int(ld.group(1)) if ld else None
print(f"\n## inference_code/（推理机快照，{m['collected_at']}，自动生成）\n")
print("评测时推理机（RTX 5080）上实际运行的推理代码和配置，由 `collect_5080_inference.sh` 从推理机直接打包；每个文件的 md5 见 `inference_code/MANIFEST.json`。\n")
print("| 路径 | 内容 |\n|---|---|")
print("| `inference_service/` | 平台推理服务（`~/workspace/lingbot/inference`），含队内补丁；未改动的平台代码原样保留，去掉了图片等资源文件 |")
print("| `team_inference_side/yolocue/` | 任务一、二：YOLO 检测目标和目的地，在头部相机画面上画框，并按配置改写指令（与训练数据同一模板） |")
print("| `team_inference_side/robotharness/` | 任务三：子任务调度（抓取 / 扫 / 倒 / 归位），给 VLA 下发子任务指令；动作轨迹全部由 VLA 输出 |")
print("| `team_inference_side/rtc/` | 异步推理 + RTC（实时分块衔接）的切换脚本和补丁 |")
print("| `team_inference_side/actsmooth/` | 动作平滑：对 VLA 输出的每段动作块做机械臂平滑和夹爪去抖，只对配置里的任务生效，不生成新动作 |")
print("| `team_inference_side/dagger_patch/` | 人工接管记录和接管后恢复的补丁脚本（打完补丁的服务代码就在 `inference_service/` 里） |")
print("| `patches/` | RTC 补丁改动的文件与平台原版的 diff |")
if sub and ld and sub != ld:
    print(f"\n- 本轮权重：`weights/hf_ckpt/`（总 global_step_{sub}，平台路径 `/mnt/forge/checkpoints/global_step_{sub}/hf_ckpt`）")
    print(f"- 收快照时（{m['collected_at']}）推理机加载的还是总 global_step_{ld}（`{w.get('source')}`）；本轮评测开始前推理机切换到本轮权重"
          "（`~/rtc/switch.sh rtc <本轮权重在推理机上的路径> --trained`），推理代码、harness 和其余配置与快照相同")
else:
    print(f"\n- 本轮权重：`{w.get('source')}`（推理机路径 `{w.get('path')}`）")
if not (sub and ld and sub != ld) and w.get('given_explicitly') and w.get('service_loaded_at_snapshot') != w.get('path'):
    print(f"- 收快照时推理机加载的还是 `{w.get('service_loaded_at_snapshot')}`；本轮评测开始前切换到本轮权重（`~/rtc/switch.sh rtc {w.get('path')} --trained`），其余代码和配置与快照相同")
if m.get('actsmooth'):
    print(f"- 动作平滑（actsmooth）：{m['actsmooth'].get('mode')}，配置 `{json.dumps(m['actsmooth'].get('config'), ensure_ascii=False)}`")
print(f"- 推理配置：`config/inference.yaml` = {m['inference_yaml']}")
print(f"- yolocue：{y['mode']}，检测权重 `team_inference_side/yolocue/{y['weights_file']}`（md5 `{y['weights_md5']}`；来历见同目录 README.txt：开源 yolo11s COCO 预训练，用赛题数据的人工核对标注微调），cue_prompt = {y['config'].get('cue_prompt', False)}")
print(f"- robotharness：{h['mode']}，调度器 {h['scheduler']}；{h['llm']}")
if h.get('graspnet_md5'):
    print(f"- 抓取判断网络（graspnet，{'harness 在用' if h.get('graspnet_used') else '文件在但 harness 未调用'}）：`team_inference_side/robotharness/graspnet/graspnet.torchscript`（md5 `{h['graspnet_md5']}`）。"
          "任务三里用手腕相机画面判断每只手是否握住工具柄，只给子任务调度做判断，不输出任何动作；训练代码和数据说明见 `training_code/graspnet/`")
print(f"- 相对平台原始代码改动过的文件：{', '.join('`%s`' % x for x in m['modified_vs_platform_bundle']) or '无'}")
print(f"- 未包含：{'；'.join(m['excluded'])}")
EOF
fi
if [ $R != first ]; then
  cat >> $L/README.md <<'EOF'

## logs/（过程日志，自动生成）

- `logs/v4_train/`：v4 训练的启动记录和日志（开发机 `/data/lingbot/runs/v4`，不含 checkpoint）。
  - `launch.txt`：每次启动的命令、环境变量、配置和数据列表；
  - `train.log.<时间戳>`：每段训练的完整输出，时间戳是这一段被下一段替换时的 Unix 时间；
  - `train.log`：收包时仍在运行的一段，可能晚于提交的权重，以上方"v4 各段"表为准；
  - `runs/`：TensorBoard 曲线（loss、学习率等）。
- `logs/data_and_switch/`：开发机 `~/lingbot_train/logs` 里的脚本输出，包括数据列表生成（`mk_v4*.out`）、换列表重启（`*_switch.log`）、checkpoint 守护等。
EOF
fi
if [ $PART = data ]; then
  printf '\n## weights/\n\n（待补：本轮权重保存后，由 `PART=base make_submit.sh` 拷入并核对 SHA-256。）\n' >> $L/README.md
fi
if [ $NOINF = 1 ] && [ $NEW_INF = 0 ]; then
  printf '\n## inference_code/\n\n（待补：推理机上的推理代码定版后，由 `PART=inf make_submit.sh` 收入推理机快照并更新本说明。）\n' >> $L/README.md
fi
hits=$(grep -rIlE '(^|[^A-Za-z0-9])sk-[A-Za-z0-9]{20,}|LTAI[A-Za-z0-9]{12,}|x-oss-signatur[e]=|[Aa]ccess[Kk]ey[Ss]ecret' $L --exclude-dir=weights --exclude-dir=data_samples 2>/dev/null | head -5)
[ -z "$hits" ] || die "possible credentials, not copied to $S: $hits"
log "copy to $S"
if [ $PART = inf ]; then
  [ -d $S/weights/hf_ckpt ] || die "$S has no weights yet: run PART=base (or all) first"
  rm -rf $S/inference_code $S/training_code/data_prep_1115 $S/training_code/graspnet
  copy $L/inference_code $S/inference_code
  [ -d $L/training_code/data_prep_1115 ] && copy $L/training_code/data_prep_1115 $S/training_code/data_prep_1115
  [ -d $L/training_code/graspnet ] && copy $L/training_code/graspnet $S/training_code/graspnet
  cp $L/README.md $S/README.md
else
  # incremental (no rsync on the dev machine): data samples by size, code / logs / README by md5; files that are no
  # longer in the staging copy are removed, but never what PART=inf owns (inference_code, training_code/data_prep_1115)
  mkdir -p $S && PART=$PART python3 - $L $S <<'EOF' || die "incremental copy to $S failed"
import hashlib, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor
L, S = sys.argv[1], sys.argv[2]
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''): h.update(b)
    return h.hexdigest()
def listing(d):
    """one walk over the destination (directory listings) instead of a stat per file: on ossfs2 a lookup of a missing
    file costs a round trip"""
    out = {}
    for root, dirs, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            try: out[os.path.relpath(p, d)] = os.path.getsize(p)
            except OSError: pass
    return out
def put(a, b):
    os.makedirs(os.path.dirname(b), exist_ok=True)
    with open(a, 'rb') as fa, open(b, 'wb') as fb: shutil.copyfileobj(fa, fb, 16 << 20)
def sync(sub, by_size, keep=()):
    src, dst = os.path.join(L, sub), os.path.join(S, sub); have = listing(dst) if os.path.isdir(dst) else {}; todo = []; n_same = 0; seen = set()
    for root, dirs, files in os.walk(src, followlinks=True):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.pyc'): continue
            a = os.path.join(root, f); rel = os.path.relpath(a, src); seen.add(rel); b = os.path.join(dst, rel)
            if have.get(rel) == os.path.getsize(a) and (by_size or md5(a) == md5(b)):
                n_same += 1; continue
            todo.append((a, b))
    for a, b in todo:                                     # sequential: 8 parallel writers stalled ossfs2 (11/16 02:20)
        put(a, b)
    gone = [r for r in have if r not in seen and r.split(os.sep)[0] not in keep]
    for r in gone: os.remove(os.path.join(dst, r))
    print(f'{sub or "."}: {len(todo)} written, {n_same} unchanged, {len(gone)} removed', flush=True)
if os.environ.get('PART') == 'weights':
    sync('logs', True)
    extra = ('README.md', 'weights/SHA256SUMS', 'training_code/train_config_of_submitted_weights.yaml')
else:
    sync('data_samples', True)
    sync('training_code', False, keep=('data_prep_1115', 'graspnet'))
    sync('logs', False)
    extra = ('README.md', 'weights/SHA256SUMS')
for f in extra:
    if os.path.isfile(os.path.join(L, f)):
        os.makedirs(os.path.dirname(os.path.join(S, f)) or S, exist_ok=True); shutil.copyfile(os.path.join(L, f), os.path.join(S, f))
EOF
  log "incremental copy to $S done"
fi
log "done: $S (local staging $(du -sh $L 2>/dev/null | cut -f1))"; (cd $S && find . -maxdepth 2 | sort | head -40)
echo "make_submit $R done: $S (log: $D/submit_$R.log)" >&3
