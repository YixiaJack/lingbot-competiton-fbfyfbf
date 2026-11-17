#!/bin/bash
# ckpt_prune.sh CHECKPOINTS_DIR [--dry-run]
# Keep the resumable DCP parts (model/, optimizer/) only for the newest checkpoint; older checkpoints keep hf_ckpt
# (the submission weights, ~12 GB) and extra_state. A checkpoint is touched only when its hf_ckpt is complete
# (config.json + every shard listed in model.safetensors.index.json) and a newer checkpoint is itself complete.
D=$1; DRY=$2
mapfile -t CK < <(ls -d "$D"/global_step_* 2>/dev/null | sort -t_ -k3 -n)
complete() { local h=$1/hf_ckpt; [ -f "$h/config.json" ] || return 1
  if [ -f "$h/model.safetensors.index.json" ]; then
    python3 -c "import json,os,sys;h=sys.argv[1];m=json.load(open(h+'/model.safetensors.index.json'))['weight_map'];sys.exit(0 if all(os.path.exists(os.path.join(h,f)) for f in set(m.values())) else 1)" "$h" || return 1
  fi; return 0; }
n=${#CK[@]}; [ "$n" -ge 2 ] || { echo "nothing to prune ($n checkpoints)"; exit 0; }
complete "${CK[$((n-1))]}" || { echo "newest ${CK[$((n-1))]} not complete yet, nothing pruned"; exit 0; }
for ((i=0; i<n-1; i++)); do c=${CK[$i]}
  [ -d "$c/model" ] || [ -d "$c/optimizer" ] || continue
  complete "$c" || { echo "skip $c: hf_ckpt incomplete"; continue; }
  sz=$(du -sh "$c/model" "$c/optimizer" 2>/dev/null | awk '{print $1}' | tr '\n' '+')
  if [ "$DRY" = "--dry-run" ]; then echo "would remove $c/{model,optimizer} ($sz)"; else rm -rf "$c/model" "$c/optimizer"; echo "removed $c/{model,optimizer} ($sz)"; fi
done
