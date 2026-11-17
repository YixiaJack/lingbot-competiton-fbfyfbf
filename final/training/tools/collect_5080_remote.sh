#!/bin/bash
# Runs ON THE 5080 (copied there and started by collect_5080_inference.sh). Builds
#   $T/lingbot_inference_<ROUND>.tar  ->  inference_code/{inference_service, team_inference_side, patches, MANIFEST.json}
# and prints "TAR <path>" and "MD5 <md5>". With DRY=1 it prints the summary and deletes everything.
#   inference_service/     ~/workspace/lingbot/inference as it runs now: the platform service with the team patches
#                          (code/deploy, code/lingbot-vla-v2 without images/assets, config/, run.sh ...)
#   team_inference_side/   yolocue (box cue + config), robotharness (task-3 subtask scheduler + mode/scheduler files),
#                          rtc (async + RTC switch, allow list), actsmooth (per-chunk action smoothing called from the
#                          patched policy_adapter.postprocess: module, config, mode), robotharness/llm.json (LLM provider)
#   patches/               unified diffs of the RTC-patched files against the originals in ~/rtc/backup
#   MANIFEST.json          what is loaded right now (checkpoint, source global_step, per-file sizes), configs, md5 of every file
# Not in the tar: YOLO weights (make_submit.sh adds them from the one-time upload /workspace/上传OSS_yolo_weights, by md5).
# Never included: ~/.deepseek_env and ~/.astra_env (API keys), the deploy job
# records (signed download URLs), logs, models, backups. The tar is refused if anything looks like a key.
R=${1:?round}; DRY=${2:-0}; MODEL=${3:-}   # MODEL: hf_ckpt path of the round's checkpoint on the 5080 when it is deployed but not loaded (default: the loaded one)
I=$HOME/workspace/lingbot/inference; T=$(mktemp -d /tmp/lbsubmit.XXXXXX); S=$T/inference_code
mkdir -p $S/inference_service $S/team_inference_side/yolocue/weights $S/team_inference_side/robotharness $S/team_inference_side/rtc $S/team_inference_side/actsmooth $S/team_inference_side/dagger_patch $S/patches
X=(--exclude=__pycache__ --exclude='*.pyc' --exclude=.git --exclude='*.orig_*' --exclude='*.bak*' --exclude='*.before_*' --exclude='*.log'
   --exclude='*.jpg' --exclude='*.jpeg' --exclude='*.png' --exclude='*.gif' --exclude='*.pdf' --exclude='*.mp4' --exclude='*.npz')

# 1) the inference service as it runs
tar -C $I -cf - "${X[@]}" --exclude=code/lingbot-vla-v2/assets code config run.sh env.sh check.sh python.sh \
    README.md SERVICE_INTEGRATION.md COMMAND_PROTOCOL.md BUNDLE_STATUS.json environment.json 2>/dev/null \
  | tar -C $S/inference_service -xf -

# 2) team components
Y=$HOME/yolocue
for f in yolocue_bridge.py config.json mode yc_server_patch.py lbcmd.sh test_yolocue.py export_ts.py live_check.py; do
  [ -f $Y/$f ] && cp -p $Y/$f $S/team_inference_side/yolocue/
done
NAMES=$(python3 -c "import json; print(json.load(open('$Y/config.json')).get('names', ''))" 2>/dev/null)
[ -n "$NAMES" ] && [ -f "$Y/$NAMES" ] && cp -p "$Y/$NAMES" $S/team_inference_side/yolocue/weights/
RHF="robotharness_bridge.py code deps mode test restart_when_idle.sh"
grep -q graspnet $HOME/robotharness/code/harness.py && RHF="$RHF graspnet"   # 11/16 grasp-check CNN: only if the running harness loads it
grep -q nudge $HOME/robotharness/code/harness.py && RHF="$RHF nudge patch_nudge.py"   # 11/16 stuck-arm nudge: switch file (absent = on) + its patch, only if the running harness has it
grep -q "make_scheduler('[A-Za-z0-9_]*')" $HOME/robotharness/robotharness_bridge.py || RHF="$RHF scheduler"   # hard-coded scheduler: the file is not read
grep -q "llm.json" $HOME/robotharness/code/harness.py && RHF="$RHF llm.json"                                     # only harness versions that read it
( cd $HOME/robotharness && tar -cf - "${X[@]}" --exclude='backup_*' $(ls -d $RHF 2>/dev/null) ) \
  | tar -C $S/team_inference_side/robotharness -xf -
for f in actsmooth.py config.json mode patch_policy_adapter.py patch_task_arg.py; do   # 11/15: per-chunk action smoothing
  [ -f $HOME/actsmooth/$f ] && cp -p $HOME/actsmooth/$f $S/team_inference_side/actsmooth/
done
for f in dagger_resume_patch.py dagger_trace_patch.py dagger_trace_test.py; do   # takeover trace / resume patch scripts (the patched server is in inference_service)
  [ -f $HOME/dagger_patch/$f ] && cp -p $HOME/dagger_patch/$f $S/team_inference_side/dagger_patch/
done
for f in switch.sh rtc_infer_patch.py p10_rtc.py test_rtc_infer.py allow.txt; do
  [ -f $HOME/rtc/$f ] && cp -p $HOME/rtc/$f $S/team_inference_side/rtc/
done

# 3) RTC diffs (original = oldest backup of each file; live file = same name with the smallest diff) + MANIFEST
python3 - "$R" "$I" "$S" "$Y" "$MODEL" <<'EOF'
import glob, hashlib, json, os, re, subprocess, sys, time
R, I, S, Y = sys.argv[1:5]; MODEL = sys.argv[5] if len(sys.argv) > 5 else ''
H = os.path.expanduser('~')
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

patches = []
orig = {}
for o in sorted(glob.glob(f'{H}/rtc/backup/*.orig_*')):
    orig.setdefault(os.path.basename(o).split('.orig_')[0], o)
for name, o in sorted(orig.items()):
    cands = [p for p in glob.glob(f'{I}/code/**/{name}', recursive=True) if '__pycache__' not in p]
    best = None
    for p in cands:
        d = subprocess.run(['diff', '-u', o, p], capture_output=True, text=True).stdout
        if best is None or len(d) < len(best[1]):
            best = (p, d)
    if best and best[1]:
        rel = os.path.relpath(best[0], I)
        out = os.path.join(S, 'patches', rel.replace('/', '__') + '.rtc.diff')
        open(out, 'w').write(best[1].replace(o, 'original/' + rel).replace(best[0], 'patched/' + rel))
        patches.append(rel)

log = sh(f'ls -t {I}/logs/lingbot_infer_*.log | head -1')
service_loaded = sh(f'grep "Policy loaded" {log} | tail -1 | sed "s/.*model_path=//"') if log else ''
loaded = MODEL.rstrip('/') or service_loaded
dep = re.search(r'/checkpoints/([^/]+)/', loaded or '')
weights = {'path': loaded, 'deployment_id': dep.group(1) if dep else None, 'source': None, 'files': [], 'given_explicitly': bool(MODEL), 'service_loaded_at_snapshot': service_loaded}
mf = f'{I}/models/checkpoints/.deploy/jobs/{weights["deployment_id"]}/manifest.json' if dep else ''
if mf and os.path.exists(mf):
    for f in json.load(open(mf)).get('files', []):
        url = f.get('download_url', '').split('?')[0]            # drop the signed query string
        name = url.split('/hf_ckpt/')[-1] if '/hf_ckpt/' in url else os.path.basename(url)
        weights['files'].append({'name': name, 'size': f.get('size_bytes'), 'etag': (f.get('headers') or {}).get('If-Match', '').strip('"')})
        if '/hf_ckpt/' in url and not weights['source']:
            weights['source'] = url.split('/hf_ckpt/')[0].split('.aliyuncs.com/')[-1] + '/hf_ckpt'
for f in weights['files']:
    p = os.path.join(loaded, f['name'])
    f['local_size_ok'] = os.path.exists(p) and os.path.getsize(p) == f['size']

cfg_yc = json.load(open(f'{Y}/config.json'))
wfile = os.path.join(Y, cfg_yc.get('weights', ''))
rd = lambda p, d='': open(p).read().strip() if os.path.exists(p) else d
inf = f'{I}/config/inference.yaml'
mode_yaml = next((m for m in ('rtc', 'sync') if os.path.exists(f'{I}/config/inference.{m}.yaml') and md5(f'{I}/config/inference.{m}.yaml') == md5(inf)), 'custom')
bundle = json.load(open(f'{I}/BUNDLE_STATUS.json')).get('updated_files_sha256', {}) if os.path.exists(f'{I}/BUNDLE_STATUS.json') else {}
modified = sorted(k for k, v in bundle.items() if os.path.exists(f'{I}/{k}') and sha(f'{I}/{k}') != v)
pid = sh('systemctl show lingbot-infer -p MainPID --value')
files = {}
for root, _, fs in os.walk(S):
    for n in fs:
        p = os.path.join(root, n); files[os.path.relpath(p, S)] = md5(p)
m = {
    'round': R, 'collected_at': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'host': sh('hostname'),
    'service': {'main_pid': pid, 'active': sh('systemctl show lingbot-infer -p ActiveState --value'),
                'started': sh(f'ps -o lstart= -p {pid}') if pid and pid != '0' else None, 'log': os.path.basename(log)},
    'loaded_model': weights,
    'inference_yaml': mode_yaml,
    'rtc_allow_lists_loaded_model': loaded in rd(f'{H}/rtc/allow.txt').splitlines(),
    'yolocue': {'mode': rd(f'{Y}/mode', 'off'), 'config': cfg_yc, 'weights_file': cfg_yc.get('weights'),
                'weights_md5': md5(wfile) if os.path.isfile(wfile) else None,
                'weights_bytes': os.path.getsize(wfile) if os.path.isfile(wfile) else None, 'weights_included': False},
    'robotharness': {'mode': rd(f'{H}/robotharness/mode', 'off'),
                     'graspnet_md5': md5(f'{H}/robotharness/graspnet/graspnet.torchscript') if os.path.isfile(f'{H}/robotharness/graspnet/graspnet.torchscript') and 'graspnet' in rd(f'{H}/robotharness/code/harness.py') else None,
                     'graspnet_used': 'graspnet' in rd(f'{H}/robotharness/code/harness.py'),
                     'nudge': (('off (switch file ~/robotharness/nudge)' if rd(f'{H}/robotharness/nudge', '').strip() == 'off' else 'on') if 'nudge' in rd(f'{H}/robotharness/code/harness.py') else 'not in this harness'),
                     'scheduler': (lambda hc: f"{hc.group(1)} (hard-coded in robotharness_bridge.py)" if hc else rd(f'{H}/robotharness/scheduler', 'proprio (default, no file)'))(
                         re.search(r"make_scheduler\('(\w+)'\)", rd(f'{H}/robotharness/robotharness_bridge.py'))),
                     'llm': (lambda c: f"{c.get('model')} via {c.get('base_url')} (llm.json; key in ~/{c.get('key_file')}, not included)" if c else
                             'DeepSeek-V4.1-Flash via API (key in ~/.deepseek_env, not included)')(
                         (lambda t: json.loads(t) if t.strip() and 'llm.json' in rd(f'{H}/robotharness/code/harness.py') else {})(rd(f'{H}/robotharness/llm.json', '')))},
    'actsmooth': {'mode': rd(f'{H}/actsmooth/mode', 'off'),
                  'config': (lambda t: json.loads(t) if t.strip() else {})(rd(f'{H}/actsmooth/config.json', ''))},
    'rtc_patched_files': patches,
    'modified_vs_platform_bundle': modified,
    'excluded': ['YOLO weights (*.torchscript/*.pt): make_submit.sh adds the one the 5080 runs from the one-time upload /workspace/上传OSS_yolo_weights, matched by md5', '~/.deepseek_env and ~/.astra_env (API keys)',
                 'models/ and the platform deploy job records (signed URLs)', 'logs/', 'backups (*.bak*, *.orig_*)',
                 'code/lingbot-vla-v2/assets and images/PDF/video files', 'test frames (*.npz)'],
    'files': files,
}
json.dump(m, open(os.path.join(S, 'MANIFEST.json'), 'w'), ensure_ascii=False, indent=1)
w = m['loaded_model']
print(f"round model: {w['path']} (explicit: {w['given_explicitly']}; service loaded now: {w['service_loaded_at_snapshot']})\n  source {w['source']} | {len(w['files'])} files, local sizes ok: {all(f['local_size_ok'] for f in w['files'])}")
print(f"inference.yaml = {mode_yaml}, rtc allow-listed: {m['rtc_allow_lists_loaded_model']} | yolocue {m['yolocue']['mode']} {cfg_yc.get('weights')} cue_prompt={cfg_yc.get('cue_prompt', False)} | robotharness {m['robotharness']['mode']} scheduler={m['robotharness']['scheduler']} nudge={m['robotharness'].get('nudge')} graspnet_used={m['robotharness'].get('graspnet_used')}")
print(f"RTC diffs: {patches}\nmodified vs platform bundle: {modified}\n{len(files)} files")
EOF

# 4) refuse anything that looks like a credential
if grep -rIlE 'sk-[A-Za-z0-9]{20,}|LTAI[A-Za-z0-9]{12,}|x-oss-signature=' $S; then
  echo "ABORT: possible credential in the files above"; rm -rf $T; exit 3
fi
tar -C $T -cf $T/lingbot_inference_$R.tar inference_code
echo "size $(du -sh $S | cut -f1) unpacked, $(du -h $T/lingbot_inference_$R.tar | cut -f1) tar"
if [ "$DRY" = 1 ]; then
  (cd $S && find . -maxdepth 3 -type d | sort | head -40); rm -rf $T; echo "dry run: nothing kept"; exit 0
fi
echo "MD5 $(md5sum $T/lingbot_inference_$R.tar | cut -d' ' -f1)"
echo "TAR $T/lingbot_inference_$R.tar"
