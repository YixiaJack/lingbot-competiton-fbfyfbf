"""Smoke checks for one arm after its short training run. Writes <out>/smoke_<arm>.json.

1. Frozen / trained parameters: compares the saved hf_ckpt with the init checkpoint on layers 0, 17 and 35.
2. Step-0 equivalence: training/vla_loss at step 1 equals the stock model's (control run) on the same batch.
3. Inference: the deploy server loads the checkpoint, routes four instructions (T1, T2, T3, unknown) and returns
   finite action chunks; latency and peak memory are recorded (run with LINGBOT_MEM_CAP_GB=15 for the 16 GB check).

  python route/smoke_check.py --arm ARM --ckpt <hf_ckpt> --init <init dir> --control_tb <dir> --arm_tb <dir> --out <dir>
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

PROMPTS = {
    0: "Grab the red apple using the left arm and place it left of the blue cup.",
    1: "Open the drawer of the wooden cabinet and place the mouse into it.",
    2: "Grab the tabletrashbin and pour the balls into the big bin.",
    -1: "Wave hello to the camera.",
}


def tensors(ckpt: Path, keys):
    wm = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for k in keys:
        if k in wm:
            with safe_open(str(ckpt / wm[k]), "pt") as f:
                out[k] = f.get_tensor(k).float()
    return out


def param_check(ckpt: Path, init: Path, arm: str):
    wm = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    pick = [k for k in wm if any(f".layers.{i}." in k for i in (0, 17, 35)) or "visual.blocks.0." in k]
    a, b = tensors(ckpt, pick), tensors(init, pick)
    cat = {}
    for k in pick:
        if k not in a or k not in b:
            continue
        frac = float((a[k] != b[k].to(a[k].dtype)).float().mean())
        if ".task_gates." in k: c = "task_gates"
        elif k.endswith(".mlp.gate.weight") and "qwen_expert" in k: c = "pretrained_gate"
        elif ".kv_lora." in k: c = "lora_" + k.split(".kv_lora.")[1].split(".")[0]
        elif "qwen_expert" in k: c = "action_expert"
        elif ".visual." in k: c = "vit"
        elif "qwenvl" in k: c = "vlm_llm"
        else: c = "other"
        cat.setdefault(c, []).append(frac)
    summary = {c: {"tensors": len(v), "changed_tensors": sum(x > 0 for x in v), "mean_frac_changed": float(np.mean(v))}
               for c, v in cat.items()}
    full = arm == "E4_fullsft"
    expect = {"task_gates": True, "pretrained_gate": False, "action_expert": True,
              "vlm_llm": full, "vit": full, "lora_k_B": True, "lora_v_B": True}
    ok = all((summary[c]["changed_tensors"] > 0) == want for c, want in expect.items() if c in summary)
    ok &= "task_gates" in summary and (("lora_k_B" in summary) == arm.startswith(("E2", "E3")))
    return {"by_group": summary, "expected_trained": {c: w for c, w in expect.items() if c in summary}, "ok": bool(ok)}


def tb_scalar(d: str, tag: str):
    # the env has numpy 2, whose removed aliases tensorboard's tf stub still uses
    for old, new in (("string_", "bytes_"), ("unicode_", "str_")):
        if not hasattr(np, old):
            setattr(np, old, getattr(np, new))
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(glob.glob(f"{d}/**/events.out.tfevents.*", recursive=True))
    vals = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            for e in ea.Scalars(tag):
                vals[e.step] = e.value
    return vals


def infer_check(ckpt: Path):
    sys.path.insert(0, os.environ["LINGBOT_DIR"]); os.chdir(os.environ["LINGBOT_DIR"])
    cap = os.environ.get("LINGBOT_MEM_CAP_GB")
    if cap:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 2**30 / total), 0)
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.models.vla.lingbot_vla import task_routing as tr
    t0 = time.time()
    srv = LingbotVLAv2Server(path_to_pi_model=str(ckpt), use_length=50, chunk_ret=True, use_bf16=True,
                             use_fp32=False, use_compile=False)
    load_s = time.time() - t0
    srv.infer({"reset": True, "robo_name": "robotwin_clean"})
    rng = np.random.default_rng(0)
    obs = {f"observation.images.{c}": rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
           for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    obs["observation.state"] = np.zeros(14, dtype=np.float32)
    rows = []
    for want, prompt in PROMPTS.items():
        torch.manual_seed(7); torch.cuda.manual_seed_all(7)
        s = time.time(); out = srv.infer(dict(obs, task=prompt)); ms = 1000 * (time.time() - s)
        act = np.concatenate([np.asarray(v).reshape(-1) for k, v in out.items() if isinstance(v, np.ndarray)])
        got = int(tr.CTX.ids.reshape(-1)[0]) if tr.CTX.ids is not None else None
        rows.append({"prompt": prompt, "expected_task": want, "routed_task": got, "finite": bool(np.isfinite(act).all()),
                     "infer_ms": ms})
    return {"load_s": load_s, "calls": rows, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "mem_cap_gb": cap,
            "ok": all(r["finite"] and r["routed_task"] == r["expected_task"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    for k in ("arm", "ckpt", "init", "control_tb", "arm_tb", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    res = {"arm": a.arm}

    def stage(name, fn):   # a crashing stage is recorded as failed instead of losing the whole result
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            res[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}

    def step0():
        c, m = tb_scalar(a.control_tb, "training/vla_loss"), tb_scalar(a.arm_tb, "training/vla_loss")
        first = min(set(c) & set(m)) if set(c) & set(m) else None
        res["arm_vla_loss_curve"] = m
        return {"step": first, "control_vla_loss": c.get(first), "arm_vla_loss": m.get(first),
                "ok": first is not None and abs(c[first] - m[first]) <= 1e-3 * max(1.0, abs(c[first]))}

    stage("params", lambda: param_check(Path(a.ckpt), Path(a.init), a.arm))
    stage("step0", step0)
    stage("infer", lambda: infer_check(Path(a.ckpt)))
    res["ok"] = all(res[k].get("ok") for k in ("params", "step0", "infer"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{'smoke' if res['ok'] else 'failed'}_{a.arm}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v["ok"] if isinstance(v, dict) and "ok" in v else v) for k, v in res.items()
                      if k != "arm_vla_loss_curve"}))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
