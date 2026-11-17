"""Mirror a running LingBot training's TensorBoard scalars to Weights & Biases.

The trainer writes tfevents under <run_dir>/runs/ (use_wandb: false in the config). This polls those files and
logs every new scalar to one wandb run whose id is derived from the run name, so restarts (preemption, requeue of
this job) resume the same wandb run instead of creating duplicates. It only reads the training output.

  python src/tb_to_wandb.py --run_dir $RUN_DIR/main_b200x4 --project lingbot-robotwin --stop_file /tmp/stop
(normally launched by scripts/hpc/wandb_sync.sbatch)
"""

import argparse
import glob
import os
import time
from collections import defaultdict

import numpy as np

np.string_, np.unicode_ = np.bytes_, np.str_  # tensorboard's tf stub still references the NumPy 1 names
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

import wandb  # noqa: E402


def read_scalars(run_dir, since=0):
    """Scalars from the run's tfevents files written at or after unix time `since` (the number after
    "events.out.tfevents." in the file name). A run that restarted from step 0 (no checkpoint survived a preemption)
    re-logs the same steps; skip the old attempt's files with --since and mirror the restart to a new wandb run."""
    by_step = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(run_dir, "runs", "**", "events.out.tfevents.*"), recursive=True)):
        if int(os.path.basename(path).split(".")[3]) < since:
            continue
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for ev in acc.Scalars(tag):
                by_step[ev.step][tag] = ev.value
    return by_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--project", default="lingbot-robotwin")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--stop_file", default="", help="exit after the pass that follows this file appearing")
    ap.add_argument("--since", type=int, default=0, help="ignore tfevents files created before this unix time")
    ap.add_argument("--id_suffix", default="", help="new wandb run for a restart, e.g. r2 -> <name>-r2")
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.run_dir))
    if args.id_suffix:
        name = f"{name}-{args.id_suffix}"
    run = wandb.init(project=args.project, name=name, id=f"{name}-tb", resume="allow")
    last = int(run.summary.get("_mirrored_step", -1))
    print(f"wandb run {run.url} | resuming after step {last}", flush=True)
    while True:
        by_step = read_scalars(args.run_dir, args.since)
        new = sorted(s for s in by_step if s > last)
        for step in new:
            wandb.log({**by_step[step], "_mirrored_step": step}, step=step)
        if new:
            last = new[-1]
            print(f"{time.strftime('%F %T')} logged steps up to {last}", flush=True)
        if args.stop_file and os.path.exists(args.stop_file):
            break  # set by wandb_sync.sbatch once the training job left the queue; this pass caught its last scalars
        time.sleep(args.interval)
    run.finish()


if __name__ == "__main__":
    main()
