"""Per-arm training yaml for the routing experiment, from the prelim from-base recipe (configs/robotwin_clean.yaml).

  python route/make_yaml.py ARM STEPS SAVE_STEPS OUT_YAML [--smoke]

Changes against the prelim recipe: init checkpoint (base + task routers [+ LoRA]), the 4 proxy tasks, steps,
freeze flags of the arm (E4 trains everything; all other arms freeze the whole VLM, LoRA excepted), MoE metrics every
100 steps. Everything else (Muon, lr 1e-4 -> 5e-5 cosine, depth / video alignment losses, MoE losses) is unchanged.
"""
import os
import sys
from pathlib import Path

R2 = Path(os.path.expandvars("$SCRATCH/lingbot/final_route_20261112"))
TEMPLATE = Path(os.path.expandvars("$SCRATCH/lingbot/LingBotRoboTwin/configs/robotwin_clean.yaml"))

arm, steps, save, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
smoke = "--smoke" in sys.argv
init = R2 / "init" / (arm if arm.startswith(("E2", "E3")) else "gates")
if arm == "control":            # stock model, no routing: reference for the step-0 equivalence check
    init = Path(os.path.expandvars("$SCRATCH/lingbot/ckpts/lingbot-vla-v2-6b"))
full = arm == "E4_fullsft"
runs = R2 / ("runs_smoke" if smoke else "runs") / arm
t = TEMPLATE.read_text()
reps = [
    (os.path.expandvars("  model_path: $SCRATCH/lingbot/ckpts/lingbot-vla-v2-6b\n"), f"  model_path: {init}\n"),
    (os.path.expandvars("  train_path: $SCRATCH/lingbot/LingBotRoboTwin/configs/robotwin_clean.txt\n"),
     f"  train_path: {R2}/lists/train_4tasks.txt\n"),
    (os.path.expandvars("  output_dir: $SCRATCH/lingbot/runs/robotwin_clean\n"), f"  output_dir: {runs}\n"),
    ("  moe_monitor_interval: 1000\n", "  moe_monitor_interval: 100\n"),
    ("  freeze_vision_encoder: false\n",
     f"  freeze_vision_encoder: {'false' if full else 'true'}\n  train_expert_only: {'false' if full else 'true'}\n"),
    ("  max_steps: 40000 # submitted run; upstream 50000\n", f"  max_steps: {steps}\n"),
    ("  save_steps: 2500\n", f"  save_steps: {save}\n"),
]
for old, new in reps:
    assert t.count(old) == 1, old
    t = t.replace(old, new)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(t)
print(out)
