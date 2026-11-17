#!/usr/bin/env bash
# Smoke test of all routing arms on one GPU (>= 140 GB for E4 full SFT): stock control 2 steps, each arm 10 steps,
# then smoke_check.py (params frozen/trained, step-0 loss equals the stock model's, deploy load + routing + 15 GB cap).
# Idempotent: arms with smoke_<arm>.json are skipped, trained arms only rerun the check (train_route.sh sees DONE). Called by the kit member or by hand: bash smoke.sh [ARM ...]
set -uo pipefail
R2=$SCRATCH/lingbot/final_route_20261112
H=$SCRATCH/lingbot/LingBotRoboTwin/route
export NGPU=1 MICRO_BS=2 GLOBAL_BS=2 GRAD_CKPT=1 FULL_SHARD=0
ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(E1_frozen E4_fullsft E2_shared_mid E3_task_mid E2_shared_last E3_task_last E2_shared_all E3_task_all)
mkdir -p "$R2/smoke" "$R2/logs"
exec 9> "$R2/smoke/.lock"; flock -n 9 || { echo "another smoke run holds the lock"; exit 0; }   # two queued copies, first one runs
bash "$H/train_route.sh" control 2 2 --smoke >> "$R2/logs/smoke_control.log" 2>&1 || { echo "control failed"; tail -40 "$R2/logs/smoke_control.log"; }
for ARM in "${ARMS[@]}"; do
  [[ -f "$R2/smoke/smoke_$ARM.json" ]] && continue
  rm -f "$R2/smoke/failed_$ARM.json"
  echo "== $ARM $(date)"
  bash "$H/train_route.sh" "$ARM" 10 10 --smoke >> "$R2/logs/smoke_train_$ARM.log" 2>&1
  rc=$?; echo "train rc=$rc"
  if (( rc != 0 )); then tail -60 "$R2/logs/smoke_train_$ARM.log"; echo "{\"arm\": \"$ARM\", \"ok\": false, \"train_rc\": $rc}" > "$R2/smoke/failed_$ARM.json"; continue; fi
  INIT="$R2/init/$ARM"; [[ "$ARM" == E1_frozen || "$ARM" == E4_fullsft ]] && INIT="$R2/init/gates"
  LINGBOT_DIR=$SCRATCH/lingbot/dev/lingbot-vla-v2-route LINGBOT_ROUTE_CONFIG="$R2/configs/$ARM.json" LINGBOT_MEM_CAP_GB=15 \
  PYTHONPATH=$SCRATCH/lingbot/dev/lingbot-vla-v2-route \
    bash $SCRATCH/lingbot/LingBotRoboTwin/scripts/hpc/run_in_env.sh python "$H/smoke_check.py" --arm "$ARM" \
      --ckpt "$R2/runs_smoke/$ARM/checkpoints/global_step_10/hf_ckpt" --init "$INIT" \
      --control_tb "$R2/runs_smoke/control" --arm_tb "$R2/runs_smoke/$ARM" --out "$R2/smoke" > "$R2/logs/smoke_check_$ARM.log" 2>&1
  rc=$?; echo "check rc=$rc"; tail -3 "$R2/logs/smoke_check_$ARM.log"
  if (( rc != 0 )) && [[ ! -f "$R2/smoke/failed_$ARM.json" ]]; then
    echo "{\"arm\": \"$ARM\", \"ok\": false, \"check_rc\": $rc}" > "$R2/smoke/failed_$ARM.json"; fi
done
ls "$R2/smoke"
python3 - "$R2/smoke" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); arms = ["E1_frozen", "E4_fullsft", "E2_shared_mid", "E3_task_mid", "E2_shared_last", "E3_task_last", "E2_shared_all", "E3_task_all"]
res = {a: json.loads((d / f"smoke_{a}.json").read_text()).get("ok") if (d / f"smoke_{a}.json").exists() else None for a in arms}
print(json.dumps(res))
if all(v is True for v in res.values()):
    (d / "SMOKE_OK").write_text(json.dumps(res) + "\n")
PY
