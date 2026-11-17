#!/usr/bin/env bash
# Login-node gate writer for the detector pipeline: only counts done markers and touches gate files. Bounded by the
# experiment deadline; touches no jobs except stopping this experiment's pool (lb_real_work_1112) at the deadline.
R=$SCRATCH/lingbot/final_real_20261112
DEADLINE=$(date -d '2026-11-13 10:00 UTC' +%s)
echo $$ > $R/controller.pid
while (( $(date +%s) < DEADLINE )); do
  [[ -f $R/PAUSED ]] && { echo "$(date '+%F %T') PAUSED file present, exit"; exit 0; }
  n=$(ls $R/det/items/*.done 2>/dev/null | wc -l)
  (( n >= 150 )) && [[ ! -f $R/det/RENDER_READY ]] && { date > $R/det/RENDER_READY; echo "$(date '+%F %T') RENDER_READY ($n items)"; }
  [[ -f $R/det/selection.json && -f $R/integ/INTEG_OK && -f $R/pool/integ_yolo/done/0 ]] && { echo "$(date '+%F %T') detector pipeline complete"; exit 0; }
  sleep 60
done
echo "$(date '+%F %T') deadline: stopping lb_real_work_1112"
ids=$(squeue -h -u "$USER" -n lb_real_work_1112 -o %i); [[ -n "$ids" ]] && scancel $ids
