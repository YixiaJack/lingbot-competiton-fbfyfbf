#!/usr/bin/env python3
"""Seed-list evaluation of one checkpoint on the AMD server: every (setting, task) of the seed list on the given GPUs.

  python eval_amd/run_seedlist_eval.py --model-path <hf_ckpt> --run-name <name> --gpus 0,1,2,3 [--base-port 13600]
      [--seed-list eval_amd/seeds_q100_s22k8k_8k_20ep.json] [--episodes 20] [--tasks a,b] [--check]

Same protocol as the NVIDIA evaluation of the reference run, as far as this machine allows:
  * seeds: exactly the reference run's (EVAL_SEED_LIST, no expert planning; eval_amd/patch_xpolicylab_seedlist.py)
  * policy server: the hosts' LingBot deploy, fp32, use_length 50, no compile (the hosts' launch_official_server.sh
    uses bf16 / 25); norm stats from LINGBOTVLA_TRAINING_CONFIG = our training config (clean-only stats, checked)
  * renderer: SAPIEN 3.0.0b1 (RoboTwin's pinned version) from /root/robotwin-scratch/sapien_b1_isolated; it only works
    with one visible GPU per process (DRI_PRIME=<that GPU's PCI address> + MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1).
    The camera near plane must be 0.01 (/RoboTwin/envs/camera/camera.py, see amd_fix/revert_fix.sh), checked here.
  * instructions: seen for demo_clean, unseen for demo_randomized
Output (LingBotRoboTwin/src/episode_table.py and compare_paired.py read it): <out>/<run>/<setting>/eval_logs/<task>.log,
<setting>/<task>.result ("successes episodes"), <setting>/state/<task>.json (resume), server_gpu<i>.log, launcher.log;
every finished item is also copied to <persist>/<run>/. Items whose .result already has --episodes are skipped, so the
same command resumes a stopped run.
"""

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROBOTWIN = Path("/RoboTwin")
HOSTS = ROBOTWIN / "experiments/lingbot_vla_v2_6b_robotwin"
PY = "/opt/robotwin-env/bin/python"
B1 = "/root/robotwin-scratch/sapien_b1_isolated"
NORM_SHA = "92b37257234b37c8642894997847da496c6e6bb550c2968eced4439c8b6bed31"
PCI = ["03", "23", "43", "63", "83", "a3", "c3", "e3"]  # HIP device order on this server (lspci, rocm-smi)
CONFIGS = ("demo_clean", "demo_randomized")
RATE = re.compile(r"Success rate: (\d+)/(\d+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def last_rate(path):
    try:
        text = ANSI.sub("", path.read_text(errors="ignore"))
    except FileNotFoundError:
        return None
    m = RATE.findall(text)
    return tuple(map(int, m[-1])) if m else None


def wait_port(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server on port {port} exited ({proc.returncode})")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(5)
    raise TimeoutError(f"server on port {port} not ready after {timeout}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--base-port", type=int, default=13600)
    ap.add_argument("--seed-list", default=str(PKG / "eval_amd/seeds_q100_s22k8k_8k_20ep.json"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--training-config", default=str(PKG / "configs/cont_amd_nodepth3k.yaml"),
                    help="yaml whose data.norm_stats_file the server uses (any of our AMD training configs)")
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--use-length", type=int, default=50)
    ap.add_argument("--out-root", default=str(PKG / "eval_runs"))
    ap.add_argument("--persist-root", default="/workspace/lingbot_eval")
    ap.add_argument("--check", action="store_true", help="EVAL_SEED_LIST_CHECK=1: also run the expert, compare info")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    out = Path(args.out_root) / args.run_name
    persist = Path(args.persist_root) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    seeds = json.load(open(args.seed_list))
    configs = args.configs.split(",")
    tasks = args.tasks.split(",") if args.tasks else list(seeds[configs[0]])

    # preconditions
    import yaml
    norm = yaml.safe_load(open(args.training_config))["data"]["norm_stats_file"]
    assert hashlib.sha256(open(norm, "rb").read()).hexdigest() == NORM_SHA, f"{norm} is not the clean-only stats"
    cam = (ROBOTWIN / "envs/camera/camera.py").read_text()
    assert re.search(r"^\s*near, far = 0\.01, 100", cam, re.M), "camera near plane is not 0.01 (amd_fix)"
    assert (Path(B1) / "sapien").is_dir(), f"SAPIEN 3.0.0b1 not installed in {B1}"
    assert (Path(args.model_path) / "model.safetensors.index.json").exists(), args.model_path
    client = ROBOTWIN / "scripts/eval_policy_xpolicylab_seedlist.py"
    subprocess.run([PY, str(PKG / "eval_amd/patch_xpolicylab_seedlist.py"),
                    str(ROBOTWIN / "scripts/eval_policy_xpolicylab.py"), str(client)], check=True)
    (out / "run.json").write_text(json.dumps(vars(args), indent=1))
    log(f"run {args.run_name}: model {args.model_path}, GPUs {gpus}, {len(tasks)} tasks x {configs}, "
        f"{args.episodes} episodes, {args.precision}, use_length {args.use_length}, norm {norm}")

    items = queue.Queue()
    for t in tasks:  # both settings of a task next to each other: paired results arrive early
        for c in configs:
            rf = out / c / f"{t}.result"
            if rf.exists() and int(rf.read_text().split()[1]) >= args.episodes:
                continue
            items.put((c, t))
    log(f"{items.qsize()} items to run")

    procs, lock, stop = [], threading.Lock(), threading.Event()

    def start_server(g):
        port = args.base_port + g
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(g), QWEN3VL_PATH=str(HOSTS / "models/Qwen3-VL-4B-Instruct-config-tokenizer"),
                   LINGBOTVLA_TRAINING_CONFIG=args.training_config, AITER_TRITON_ONLY="1",
                   FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE", SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0",
                   TMPDIR=str(out / "tmp"), TRITON_CACHE_DIR=str(out / "tmp/triton"))
        for k in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            env.pop(k, None)
        (out / "tmp/triton").mkdir(parents=True, exist_ok=True)
        fp32 = args.precision == "fp32"
        cmd = [PY, "-m", "deploy.lingbot_vla_v2_policy", "--model_path", args.model_path, "--use_length",
               str(args.use_length), "--use_bf16", str(not fp32), "--use_fp32", str(fp32), "--use_compile", "False",
               "--port", str(port)]
        fh = open(out / f"server_gpu{g}.log", "ab")
        p = subprocess.Popen(cmd, cwd=HOSTS / "source/lingbot-vla-v2", env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        with lock:
            procs.append(p)
        wait_port(port, p)
        log(f"GPU {g}: server ready on port {port}")
        return port

    def run_item(g, port, cfg, task):
        d = out / cfg
        (d / "eval_logs").mkdir(parents=True, exist_ok=True); (d / "state").mkdir(exist_ok=True)
        env = dict(os.environ, HIP_VISIBLE_DEVICES=str(g), ROBOTWIN_DISABLE_CUROBO="1", ROBOTWIN_EE_PLANNER="mplib",
                   PYOPENGL_PLATFORM="egl", PYTHONUNBUFFERED="1", PYTHONPATH=B1 + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   DRI_PRIME=f"pci-0000_{PCI[g]}_00_0", MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE="1",
                   EVAL_SEED_LIST=args.seed_list, EVAL_STATE_FILE=str(d / "state" / f"{task}.json"))
        if args.check:
            env["EVAL_SEED_LIST_CHECK"] = "1"
        for k in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            env.pop(k, None)
        cmd = [PY, str(client), "--task_name", task, "--task_config", cfg, "--policy_name", "LingBot-VLA-v2",
               "--protocol", "lingbot_vla_v2", "--host", "127.0.0.1", "--port", str(port), "--device_id", "0",
               "--seed", "0", "--test_num", str(args.episodes),
               "--instruction_type", "seen" if cfg == "demo_clean" else "unseen", "--expert_check", "false",
               "--eval_batch", "false", "--additional_info", "eval_video_log=false"]
        t0 = time.time()
        lp = d / "eval_logs" / f"{task}.log"
        with open(lp, "ab") as fh:
            p = subprocess.Popen(cmd, cwd=ROBOTWIN, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            with lock:
                procs.append(p)
            rc = p.wait()
        r = last_rate(lp)
        if rc == 0 and r and r[1] >= args.episodes:
            (d / f"{task}.result").write_text(f"{r[0]} {r[1]}\n")
        log(f"GPU {g}: {cfg}/{task} rc={rc} {r[0] if r else '-'}/{r[1] if r else '-'} in {time.time() - t0:.0f}s")
        try:  # small files only: logs, results, state
            for sub in ("eval_logs", "state"):
                (persist / cfg / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy(lp, persist / cfg / "eval_logs" / lp.name)
            for f in (d / f"{task}.result", d / "state" / f"{task}.json"):
                if f.exists():
                    shutil.copy(f, persist / cfg / (f"{task}.result" if f.suffix == ".result" else f"state/{f.name}"))
        except OSError as e:
            log(f"persist copy failed: {e}")
        return rc == 0

    def worker(g):
        try:
            port = start_server(g)
        except Exception as e:
            log(f"GPU {g}: server failed: {e}")
            return
        fails = 0
        while not stop.is_set():
            try:
                cfg, task = items.get_nowait()
            except queue.Empty:
                return
            if not run_item(g, port, cfg, task):
                fails += 1
                items.put((cfg, task))  # leave it to another GPU / a later turn (the state file resumes it)
                if fails >= 3:
                    log(f"GPU {g}: 3 failed items, stopping this worker")
                    return

    def shutdown(*_):
        stop.set()
        with lock:
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, lambda *a: (shutdown(), sys.exit(1)))
    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    try:
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        shutdown()
    done = {(c, t): (out / c / f"{t}.result").read_text().split() for t in tasks for c in configs
            if (out / c / f"{t}.result").exists()}
    for c in configs:
        s = sum(int(v[0]) for (cc, _), v in done.items() if cc == c); n = sum(int(v[1]) for (cc, _), v in done.items() if cc == c)
        log(f"{c}: {sum(1 for k in done if k[0] == c)}/{len(tasks)} tasks, {s}/{n}" + (f" = {100 * s / n:.1f}%" if n else ""))


if __name__ == "__main__":
    main()
