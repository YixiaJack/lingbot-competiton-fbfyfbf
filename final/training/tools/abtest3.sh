#!/bin/bash
# Sequential A/B of the env-gated optimizations on the code_opt branch, same config/seed as smoke4 (baseline).
# Each test: 20 steps on 8 GPUs; summary appended to runs/ab_summary.txt.
W=$HOME/lingbot_train
CFG=$W/configs/smoke.yaml
summ=$W/runs/ab_summary.txt
run_one() {   # name, env assignments...
  local name=$1; shift
  rm -rf "$W/runs/$name"
  env "$@" CODE=$W/code_opt "$W/tools/train_run.sh" "$name" "$CFG" 0,1,2,3,4,5,6,7 --train.max_steps 20
  sleep 20
  local pid; pid=$(cat "$W/runs/$name/pid")
  while kill -0 "$pid" 2>/dev/null; do sleep 10; done
  sleep 5
  {
    echo "== $name  ($*)"
    tr '\r' '\n' < "$W/runs/$name/train.log" | grep StepTime | sed -E 's/.*Loss ([0-9.]+), VLA_Loss ([0-9.]+).*GradNorm ([0-9.]+).*StepTime ([0-9.]+)s, Depth_Forward_Time +([0-9.]+)s.*/\1 \2 \3 \4 \5/' \
      | awk '{n++; l=l" "$1; if(n>=6 && n<=13){t+=$4; c++}; d=$5} END{printf "steps %d  mean_step_s(6-13) %.3f  depth_s %s\n  loss:%s\n", n, (c?t/c:0), d, l}'
    grep -E "OutOfMemory|ChildFailedError|LB_FSDP_VISION|LB_FSDP_PREFETCH" "$W/runs/$name/train.log" | grep -v _dynamo | sort | uniq -c | head -5
  } >> "$summ"
}
echo "== round 3 (all-on first) started $(date -Is)" >> "$summ"
run_one ab_ALL LB_VIT_BATCHED_SDPA=1 LB_GRID_CACHE=1 LB_FSDP_VISION=1 LB_FSDP_PREFETCH=1 LB_FUSED_OPS=1
# decide: all-on if faster than T2 (8.674 s) and first-14-step losses within 0.01 of the baseline
dec=$(python3 - "$W" <<'PY'
import re, sys
W = sys.argv[1]
def steps(name):
    out = []
    for line in open(f"{W}/runs/{name}/train.log", errors="ignore").read().replace("\r", "\n").splitlines():
        m = re.search(r"Loss ([0-9.]+), VLA_Loss.*StepTime ([0-9.]+)s", line)
        if m: out.append((float(m.group(1)), float(m.group(2))))
    return out
b, a = steps("smoke4"), steps("ab_ALL")
n = min(14, len(b), len(a))
dmax = max(abs(x[0] - y[0]) for x, y in zip(b[:n], a[:n])) if n else 9
mean = sum(t for _, t in a[5:13]) / max(1, len(a[5:13]))
ok = len(a) >= 14 and mean < 8.674 and dmax < 0.01
print(("ALL" if ok else "T2") + f" mean_step={mean:.3f} max_loss_diff_vs_base={dmax:.4f} steps={len(a)}")
PY
)
echo "== decision: $dec" >> "$summ"
case "$dec" in ALL*) ENVS="LB_VIT_BATCHED_SDPA=1 LB_GRID_CACHE=1 LB_FSDP_VISION=1 LB_FSDP_PREFETCH=1 LB_FUSED_OPS=1";; *) ENVS="LB_VIT_BATCHED_SDPA=1 LB_GRID_CACHE=1 LB_FSDP_VISION=1";; esac
cd "$W"
env $ENVS FORGE_CKPT_KEEP_DCP=1 CODE=$W/code_opt "$W/tools/train_run.sh" v1 "$W/configs/v1.yaml" 0,1,2,3,4,5,6,7 >> "$summ"
(cd "$W/code_opt" && git log --oneline -1) > "$W/runs/v1/code_commit.txt"
echo "$ENVS FORGE_CKPT_KEEP_DCP=1" > "$W/runs/v1/env.txt"
echo "== v1 launched with: $ENVS  at $(date -Is)" >> "$summ"
