"""Apply the DINOv2-fusion changes to a SEPARATE checkout of lingbot-vla-v2 (never the live one). Idempotent.

  python dino/patch_dino.py $SCRATCH/lingbot/dev/lingbot-vla-v2-dino

  * copies dino/dino_fusion.py to lingbotvla/models/vla/lingbot_vla/dino_fusion.py
  * FlowMatchingV2.__init__: builds `self.dino_fusion` when config.dino_fusion_path is set
  * FlowMatchingV2.embed_prefix: img_emb += dino_fusion(pixel_values) (zero at init)
  * train_lingbotvla.py: training arguments dino_fusion_path / dino_fusion_input_size / dino_fusion_grid, copied onto
    the model config (LingbotVLAConfig.__init__ silently drops unknown kwargs)
  * optimizer.py: LINGBOT_DINO_LR_SCALE multiplies the lr of the (newly initialized) dino_fusion parameters
  * train_lingbotvla.py resume: keep the config's lr. LambdaLR.load_state_dict restores the checkpoint's base lrs, so a
    resubmission with a new --train.lr kept the old one (dino_cont8k_6k from step 500: 1e-5 instead of 5e-6, 10/24);
    LINGBOT_RESUME_KEEP_LR=1 keeps the checkpoint's
  * train_lingbotvla.py time-based saves: preemptible jobs may be ended 30 min after they start (PreemptExemptTime,
    GraceTime 0), so a member slower than save_steps per ~25 min never saved (2xA100, 10/24); at every multiple of
    LINGBOT_TIME_SAVE_MIN minutes (default 24) after the process started a resume-only checkpoint (no hf_ckpt) is
    saved as well, unless a save finished under LINGBOT_TIME_SAVE_SKIP_MIN (default 3) minutes earlier; after each
    save rank 0 keeps only that checkpoint, the newest one with a complete hf_ckpt (to evaluate) and any holding a
    file named `keep` (e.g. one being evaluated), and deletes the rest (user request 10/24: keep only the newest)
  * resume across a world size change: a checkpoint saved by 2 ranks has no extra_state file for ranks 2-3 of a 4-GPU
    job; those ranks fell back to an older checkpoint on their own while ranks 0-1 waited in dist.barrier(), and the
    4xB200 member hung (10/24). Such ranks now load a saved rank's file, and the per-rank dataloader state is not
    restored when the saved and current world sizes differ (the data order restarts; everything else resumes)
  * resume agreement (user requests 10/24): every rank must load a candidate checkpoint before it counts, else all
    ranks go on to the next older one together; when none loads, the job does not start from scratch by itself but
    writes its decision to <run>/.resume_failed/<jobid> for scripts/hpc/train.sbatch: "requeue" (hand the run to the
    other members) when a checkpoint was saved with another GPU count and this job has not failed before, else
    "restart" (checkpoints moved aside, training again from the initial model, i.e. 22k+8k + zero-init fusion). The
    cleanup deletes only checkpoints older than the one just saved, and keeps the newest one of every saved world
    size, so a 4-GPU member that cannot resume from a 2-rank checkpoint falls back to a 4-rank one
  * per-rank random streams (10/25): every rank had the same seed, so with LINGBOT_TASK_SAMPLING=temp:T (task and
    frame drawn by the worker's RNG, the sampler index unused) all ranks trained on the same samples, augmentation and
    noise, and every resume re-drew the batches drawn since its dataloader iterator was created (the base seed is
    restored, the worker RNG restarts). Before each iterator the workers and the main process are reseeded from
    (seed, dp rank, global step, epoch)
The env-gated augmentation / sampling changes come from src/patch_upstream.py, applied to the same checkout.
"""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MV2 = "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"

# time-based save schedule, as first applied (counted from the last save: a step save at minute ~15 pushed the next
# time save past the preemption at ~30 min, 10/24) and as now (fixed grid from the process start)
_TIME_SAVE_V1 = (
    "    # same checkpoint every time. LINGBOT_TIME_SAVE_MIN=m (default 24, 0 = off) also saves m minutes after the\n"
    "    # process started and m minutes after each save (resume state only, no hf_ckpt). Rank 0's clock decides\n"
    "    # (broadcast every step) so that all ranks enter the collective save together.\n"
    "    _time_save_min = float(os.environ.get(\"LINGBOT_TIME_SAVE_MIN\", \"24\"))\n"
    "    _last_save_time = [_PROCESS_START]\n"
    "\n"
    "    def _time_save_due() -> bool:\n"
    "        if _time_save_min <= 0:\n"
    "            return False\n"
    "        due = torch.tensor([int(time.time() - _last_save_time[0] >= 60 * _time_save_min)], dtype=torch.int32,\n"
    "                           device=\"cuda\")\n"
    "        dist.broadcast(due, src=0)\n"
    "        return bool(due.item())\n"
)
_TIME_SAVE = (
    "    # same checkpoint every time. LINGBOT_TIME_SAVE_MIN=m (default 24, 0 = off) also saves at m, 2m, ... minutes\n"
    "    # after the process started (resume state only, no hf_ckpt), whatever step saves happened in between; skipped\n"
    "    # when a save finished under LINGBOT_TIME_SAVE_SKIP_MIN (default 3) minutes earlier. Rank 0's clock decides\n"
    "    # (broadcast every step) so that all ranks enter the collective save together.\n"
    "    _time_save_min = float(os.environ.get(\"LINGBOT_TIME_SAVE_MIN\", \"24\"))\n"
    "    _time_save_skip_min = float(os.environ.get(\"LINGBOT_TIME_SAVE_SKIP_MIN\", \"3\"))\n"
    "    _last_save_time = [_PROCESS_START]\n"
    "    _next_time_save = [_PROCESS_START + 60 * _time_save_min]\n"
    "\n"
    "    def _time_save_due() -> bool:\n"
    "        if _time_save_min <= 0:\n"
    "            return False\n"
    "        now = time.time()\n"
    "        due = now >= _next_time_save[0] and now - _last_save_time[0] >= 60 * _time_save_skip_min\n"
    "        while _next_time_save[0] <= now:  # next slot on the grid counted from the process start\n"
    "            _next_time_save[0] += 60 * _time_save_min\n"
    "        due = torch.tensor([int(due)], dtype=torch.int32, device=\"cuda\")\n"
    "        dist.broadcast(due, src=0)\n"
    "        return bool(due.item())\n"
)

# no checkpoint could be loaded: as first applied (always requeue, cancel on a second failure) and now (requeue or
# restart from the initial model, decided here and carried out by scripts/hpc/train.sbatch)
_NOT_LOADED_V1 = (
    "        if not loaded:\n"
    "            # LingBotRoboTwin: never start from scratch while checkpoints exist (its first save would prune them\n"
    "            # all); leave a note for scripts/hpc/train.sbatch, which requeues this job (cancels it on a 2nd failure)\n"
    "            if args.train.global_rank == 0:\n"
    "                os.makedirs(os.path.dirname(_resume_note), exist_ok=True)\n"
    "                with open(_resume_note, \"a\") as _f:\n"
    "                    _f.write(f\"{os.path.basename(candidates[0])} {time.strftime('%F %T')} {last_err!r}\\n\")\n"
)
_NOT_LOADED = (
    "        if not loaded:\n"
    "            # LingBotRoboTwin: never start from scratch by itself while checkpoints exist. Decision for\n"
    "            # scripts/hpc/train.sbatch: \"requeue\" hands the run to the other members when a checkpoint was saved\n"
    "            # with another GPU count (one of them may read it) and this job has not failed before; else \"restart\":\n"
    "            # the checkpoints are moved aside and training starts again from the initial model (user request 10/24)\n"
    "            if args.train.global_rank == 0:\n"
    "                _ws = lambda p: (sum(f.startswith(\"extra_state_rank_\") for f in os.listdir(os.path.join(p, \"extra_state\")))\n"
    "                                 if os.path.isdir(os.path.join(p, \"extra_state\")) else 0)\n"
    "                _other_ws = any(_ws(c) not in (0, dist.get_world_size()) for c in candidates)\n"
    "                _action = \"requeue\" if _other_ws and not os.path.exists(_resume_note) else \"restart\"\n"
    "                os.makedirs(os.path.dirname(_resume_note), exist_ok=True)\n"
    "                with open(_resume_note, \"a\") as _f:\n"
    "                    _f.write(f\"{os.path.basename(candidates[0])} {_action} {time.strftime('%F %T')} {last_err!r}\\n\")\n"
)

# text left by an older version of this script -> its current form (applied before EDITS where the old text is found
# and the new one is not; a new text may contain its old one)
UPGRADES = {
    "tasks/vla/train_lingbotvla.py": [
        (_TIME_SAVE_V1, _TIME_SAVE),
        (  # cleanup: also keep the newest checkpoint of every saved world size (10/24)
            "        # keeps `keep`, the newest checkpoint with a complete hf_ckpt (for evaluation) and any checkpoint holding a\n"
            "        # file named `keep`; deletes the rest, including newer ones left by a save that a preemption cut off\n",
            "        # keeps `keep`, the newest checkpoint with a complete hf_ckpt (for evaluation), the newest one saved by each\n"
            "        # world size (fallback for a member with another GPU count) and any checkpoint holding a file named\n"
            "        # `keep`; deletes the rest, including newer ones left by a save that a preemption cut off\n",
        ),
        (
            "            newest_hf = max((st for st, p in steps.items() if st <= keep_step and hf_ok(p)), default=None)\n",
            "            newest_hf = max((st for st, p in steps.items() if st <= keep_step and hf_ok(p)), default=None)\n"
            "            n_ranks = lambda p: (sum(f.startswith(\"extra_state_rank_\") for f in os.listdir(os.path.join(p, \"extra_state\")))\n"
            "                                 if os.path.isdir(os.path.join(p, \"extra_state\")) else 0)\n"
            "            newest_ws = {n_ranks(steps[st]): st for st in sorted(steps) if st <= keep_step}  # world size -> step\n"
            "            newest_ws.pop(0, None)\n",
        ),
        (
            "                if st in (keep_step, newest_hf) or os.path.exists(os.path.join(path, \"keep\")):\n",
            "                if st in (keep_step, newest_hf, *newest_ws.values()) or os.path.exists(os.path.join(path, \"keep\")):\n",
        ),
        (
            "                logger.info_rank0(f\"deleting checkpoint {path} (keeping {keep_step}, hf {newest_hf})\")\n",
            "                logger.info_rank0(f\"deleting checkpoint {path} (keeping {keep_step}, hf {newest_hf}, \"\n"
            "                                  f\"per world size {newest_ws})\")\n",
        ),
        (  # cleanup: never delete a checkpoint newer than the one just saved (user request 10/24)
            "        # keeps `keep`, the newest checkpoint with a complete hf_ckpt (for evaluation), the newest one saved by each\n"
            "        # world size (fallback for a member with another GPU count) and any checkpoint holding a file named\n"
            "        # `keep`; deletes the rest, including newer ones left by a save that a preemption cut off\n",
            "        # deletes only checkpoints older than `keep` (user request 10/24), except the newest one with a complete\n"
            "        # hf_ckpt (for evaluation), the newest one saved by each world size (fallback for a member with another\n"
            "        # GPU count) and any checkpoint holding a file named `keep`\n",
        ),
        (
            "                if st in (keep_step, newest_hf, *newest_ws.values()) or os.path.exists(os.path.join(path, \"keep\")):\n",
            "                if st >= keep_step or st in (newest_hf, *newest_ws.values()) or os.path.exists(os.path.join(path, \"keep\")):\n",
        ),
        (_NOT_LOADED_V1, _NOT_LOADED),
    ],
}

EDITS = {
    MV2: [
        (
            "from .configuration_lingbot_vla import LingbotVLAV2Config\n",
            "from .configuration_lingbot_vla import LingbotVLAV2Config\n"
            "from .dino_fusion import DinoFusion  # LingBotRoboTwin\n",
        ),
        (
            "        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)\n",
            "        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)\n"
            "        # LingBotRoboTwin: frozen DINOv2 features added to the visual tokens (zero-initialized projection)\n"
            "        if getattr(self.config, \"dino_fusion_path\", None):\n"
            "            self.dino_fusion = DinoFusion(\n"
            "                self.config.dino_fusion_path,\n"
            "                self.qwenvl_with_expert.qwenvl.config.text_config.hidden_size,\n"
            "                input_size=getattr(self.config, \"dino_fusion_input_size\", 224),\n"
            "                grid=getattr(self.config, \"dino_fusion_grid\", 8),\n"
            "            )\n",
        ),
        (
            "        embed_dtype = img_emb.dtype\n        num_patch = img_emb.shape[1]\n",
            "        embed_dtype = img_emb.dtype\n        num_patch = img_emb.shape[1]\n"
            "        # LingBotRoboTwin: residual DINOv2 features on the visual tokens (same (b n), token order)\n"
            "        if getattr(self, \"dino_fusion\", None) is not None:\n"
            "            img_emb = img_emb + self.dino_fusion(images).to(img_emb.dtype)\n",
        ),
    ],
    "tasks/vla/train_lingbotvla.py": [
        (
            "    align_params: Optional[Dict[str, Any]] = field(\n",
            "    dino_fusion_path: Optional[str] = field(  # LingBotRoboTwin\n"
            "        default=None,\n"
            "        metadata={\"help\": \"Frozen DINOv2 dir (config.json); its weights must be in model_path.\"},\n"
            "    )\n"
            "    dino_fusion_input_size: int = field(default=224, metadata={\"help\": \"DINOv2 input resolution.\"})\n"
            "    dino_fusion_grid: int = field(default=8, metadata={\"help\": \"Token grid per camera (Qwen merger grid).\"})\n"
            "    align_params: Optional[Dict[str, Any]] = field(\n",
        ),
    ],
    "tasks/vla/train_lingbotvla.py#config": [
        (
            "        config_cls = config_cls(**config_kwargs)\n",
            "        config_cls = config_cls(**config_kwargs)\n"
            "        # LingBotRoboTwin: LingbotVLAConfig.__init__ drops unknown kwargs; carry the DINO fusion options over\n"
            "        for _key in (\"dino_fusion_path\", \"dino_fusion_input_size\", \"dino_fusion_grid\"):\n"
            "            if config_kwargs.get(_key) is not None:\n"
            "                setattr(config_cls, _key, config_kwargs[_key])\n",
        ),
    ],
    "tasks/vla/train_lingbotvla.py#resume_lr": [
        (
            "        last_err = None\n",
            "        fresh_base_lrs = list(lr_scheduler.base_lrs)  # LingBotRoboTwin: the config's lr, before a resume\n"
            "        last_err = None\n",
        ),
        (
            "                lr_scheduler.load_state_dict(state[\"extra_state\"][\"lr_scheduler\"])\n",
            "                lr_scheduler.load_state_dict(state[\"extra_state\"][\"lr_scheduler\"])\n"
            "                # LingBotRoboTwin: load_state_dict brings back the checkpoint's base lrs; use the config's\n"
            "                if (os.environ.get(\"LINGBOT_RESUME_KEEP_LR\", \"0\") != \"1\"\n"
            "                        and len(lr_scheduler.base_lrs) == len(fresh_base_lrs)\n"
            "                        and lr_scheduler.base_lrs != fresh_base_lrs):\n"
            "                    logger.info_rank0(f\"resume: base lrs {lr_scheduler.base_lrs} -> {fresh_base_lrs} (config)\")\n"
            "                    lr_scheduler.base_lrs = fresh_base_lrs\n"
            "                    for _g, _base, _fn in zip(optimizer.param_groups, fresh_base_lrs, lr_scheduler.lr_lambdas):\n"
            "                        _g[\"initial_lr\"] = _base\n"
            "                        _g[\"lr\"] = _base * _fn(lr_scheduler.last_epoch)\n"
            "                    lr_scheduler._last_lr = [_g[\"lr\"] for _g in optimizer.param_groups]\n",
        ),
    ],
    "tasks/vla/train_lingbotvla.py#time_save": [
        (
            "import re\nimport time\n",
            "import re\nimport time\n"
            "_PROCESS_START = time.time()  # LingBotRoboTwin: time-based saves count from here\n",
        ),
        (
            "    def save_hf_checkpoint_best_effort(\n",
            "    # LingBotRoboTwin: preemptible partitions may end a job 30 min after it starts (PreemptExemptTime) with no\n"
            "    # grace time, so a member slower than save_steps per ~25 min (2xA100: 200 steps x 15 s) restarted from the\n"
            + _TIME_SAVE +
            "\n"
            "    def save_hf_checkpoint_best_effort(\n",
        ),
        (
            "            if args.train.save_steps and global_step % args.train.save_steps == 0:\n"
            "                helper.empty_cache()\n",
            "            _step_save = bool(args.train.save_steps and global_step % args.train.save_steps == 0)\n"
            "            _time_only = _time_save_due() and not _step_save  # LingBotRoboTwin (every rank, every step)\n"
            "            if _step_save or _time_only:\n"
            "                helper.empty_cache()\n",
        ),
        (
            "                if args.train.global_rank == 0:\n"
            "                    writer.flush()\n"
            "                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)\n"
            "                dist.barrier()\n"
            "                logger.info_rank0(f\"Distributed checkpoint saved at {save_checkpoint_path} successfully!\")\n"
            "                save_hf_checkpoint_best_effort(\n"
            "                    save_checkpoint_path,\n",
            "                if args.train.global_rank == 0:\n"
            "                    writer.flush()\n"
            "                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)\n"
            "                dist.barrier()\n"
            "                _last_save_time[0] = time.time()  # LingBotRoboTwin\n"
            "                logger.info_rank0(f\"Distributed checkpoint saved at {save_checkpoint_path} successfully!\"\n"
            "                                  + (\" (time-based, no hf_ckpt)\" if _time_only else \"\"))\n"
            "                save_hf_checkpoint_best_effort(\n"
            "                    None if _time_only else save_checkpoint_path,  # LingBotRoboTwin: time-based = resume only\n",
        ),
    ],
    "tasks/vla/train_lingbotvla.py#prune": [  # anchored outside the #time_save texts (idempotent re-runs)
        (
            "    # LingBotRoboTwin: preemptible partitions may end a job 30 min after it starts (PreemptExemptTime) with no\n",
            "    def _prune_resume_only(keep: str) -> None:  # rank 0, after `keep` was saved\n"
            "        # deletes only checkpoints older than `keep` (user request 10/24), except the newest one with a complete\n"
            "        # hf_ckpt (for evaluation), the newest one saved by each world size (fallback for a member with another\n"
            "        # GPU count) and any checkpoint holding a file named `keep`\n"
            "        import shutil, threading\n"
            "        try:\n"
            "            root, keep_step = os.path.dirname(keep), int(os.path.basename(keep).rsplit(\"_\", 1)[1])\n"
            "            steps = {int(m.group(1)): os.path.join(root, m.group(0)) for m in\n"
            "                     (re.fullmatch(r\"global_step_(\\d+)\", n) for n in os.listdir(root)) if m}\n"
            "            hf_ok = lambda p: os.path.exists(os.path.join(p, \"hf_ckpt\", \"model.safetensors.index.json\"))\n"
            "            newest_hf = max((st for st, p in steps.items() if st <= keep_step and hf_ok(p)), default=None)\n"
            "            n_ranks = lambda p: (sum(f.startswith(\"extra_state_rank_\") for f in os.listdir(os.path.join(p, \"extra_state\")))\n"
            "                                 if os.path.isdir(os.path.join(p, \"extra_state\")) else 0)\n"
            "            newest_ws = {n_ranks(steps[st]): st for st in sorted(steps) if st <= keep_step}  # world size -> step\n"
            "            newest_ws.pop(0, None)\n"
            "            doomed = [os.path.join(root, n) for n in os.listdir(root) if n.endswith(\".deleting\")]\n"
            "            for st, path in steps.items():\n"
            "                if st >= keep_step or st in (newest_hf, *newest_ws.values()) or os.path.exists(os.path.join(path, \"keep\")):\n"
            "                    continue\n"
            "                os.rename(path, path + \".deleting\")  # the resume scan only matches global_step_<n>\n"
            "                logger.info_rank0(f\"deleting checkpoint {path} (keeping {keep_step}, hf {newest_hf}, \"\n"
            "                                  f\"per world size {newest_ws})\")\n"
            "                doomed.append(path + \".deleting\")\n"
            "            for path in doomed:\n"
            "                threading.Thread(target=shutil.rmtree, args=(path,), kwargs={\"ignore_errors\": True},\n"
            "                                 daemon=True).start()\n"
            "        except Exception as e:  # never stop training over a cleanup\n"
            "            logger.info_rank0(f\"checkpoint cleanup failed: {e!r}\")\n"
            "\n"
            "    # LingBotRoboTwin: preemptible partitions may end a job 30 min after it starts (PreemptExemptTime) with no\n",
        ),
        (
            "                    None if _time_only else save_checkpoint_path,  # LingBotRoboTwin: time-based = resume only\n"
            "                    state,\n"
            "                    global_step,\n"
            "                    current_epoch_for_eval,\n"
            "                    current_epoch_step_for_eval,\n"
            "                )\n",
            "                    None if _time_only else save_checkpoint_path,  # LingBotRoboTwin: time-based = resume only\n"
            "                    state,\n"
            "                    global_step,\n"
            "                    current_epoch_for_eval,\n"
            "                    current_epoch_step_for_eval,\n"
            "                )\n"
            "                if args.train.global_rank == 0:\n"
            "                    _prune_resume_only(save_checkpoint_path)  # LingBotRoboTwin\n",
        ),
    ],
    "lingbotvla/checkpoint/checkpointer.py": [
        (
            "            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))\n"
            "            state[\"extra_state\"] = torch.load(\n"
            "                extra_state_path,\n"
            "            )\n",
            "            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))\n"
            "            # LingBotRoboTwin: saved by fewer ranks (2xA100 -> 4xB200) -> no file for this rank; take a saved one\n"
            "            _saved = sorted(f for f in os.listdir(extra_state_dir) if f.startswith(\"extra_state_rank_\"))\n"
            "            if not os.path.exists(extra_state_path) and _saved:\n"
            "                extra_state_path = os.path.join(extra_state_dir, _saved[dist.get_rank() % len(_saved)])\n"
            "            state[\"extra_state\"] = torch.load(\n"
            "                extra_state_path,\n"
            "            )\n"
            "            state[\"extra_state\"][\"_saved_world_size\"] = len(_saved)  # LingBotRoboTwin\n",
        ),
    ],
    "tasks/vla/train_lingbotvla.py#world_size": [
        (
            "                if start_step > 0 and args.train.resume_dataloader_state:\n"
            "                    train_dataloader.load_state_dict(state[\"extra_state\"][\"train_dataloader\"])\n",
            "                # LingBotRoboTwin: the dataloader state is per rank; after a world size change start it afresh\n"
            "                _saved_ws = state[\"extra_state\"].get(\"_saved_world_size\", dist.get_world_size())\n"
            "                if _saved_ws != dist.get_world_size():\n"
            "                    logger.info_rank0(f\"resume: checkpoint saved by {_saved_ws} ranks, running \"\n"
            "                                      f\"{dist.get_world_size()}: dataloader state not restored\")\n"
            "                if start_step > 0 and args.train.resume_dataloader_state and _saved_ws == dist.get_world_size():\n"
            "                    train_dataloader.load_state_dict(state[\"extra_state\"][\"train_dataloader\"])\n",
        ),
    ],
    "tasks/vla/train_lingbotvla.py#resume_agree": [
        (
            "        loaded = False\n"
            "        for cp in candidates:\n",
            "        loaded = False\n"
            "        _resume_note = os.path.join(args.train.output_dir, \".resume_failed\", os.environ.get(\"SLURM_JOB_ID\", \"local\"))\n"
            "        for cp in candidates:\n",
        ),
        (
            "                dist.barrier()\n"
            "                logger.info_rank0(f\"Load distributed checkpoint from {cp} successfully!\")\n"
            "                loaded = True\n"
            "                break\n"
            "            except Exception as e:\n"
            "                last_err = e\n"
            "                logger.info_rank0(f\"Failed to load checkpoint {cp}: {repr(e)}. Trying older one...\")\n"
            "                continue\n"
            "        if not loaded:\n"
            "            logger.info_rank0(\"Starting training from scratch. No valid checkpoint could be loaded.\")\n",
            "                _ok = 1  # LingBotRoboTwin: counts only if every rank loaded it (below)\n"
            "            except Exception as e:\n"
            "                last_err, _ok = e, 0\n"
            "                logger.info(f\"rank {dist.get_rank()}: failed to load checkpoint {cp}: {e!r}\")\n"
            "            # LingBotRoboTwin: a rank that failed alone went on to an older checkpoint (a collective) while the\n"
            "            # others waited in dist.barrier() -> NCCL timeout, job FAILED and out of the queue (4xB200, 10/24);\n"
            "            # now the ranks agree on every candidate and move on together\n"
            "            _agree = torch.tensor([_ok], dtype=torch.int32, device=\"cuda\")\n"
            "            dist.all_reduce(_agree, op=dist.ReduceOp.MIN)\n"
            "            if _agree.item() == 1:\n"
            "                logger.info_rank0(f\"Load distributed checkpoint from {cp} successfully!\")\n"
            "                if args.train.global_rank == 0 and os.path.exists(_resume_note):\n"
            "                    os.remove(_resume_note)  # earlier failed resumes of this job no longer count\n"
            "                loaded = True\n"
            "                break\n"
            "            logger.info_rank0(f\"Failed to load checkpoint {cp} on at least one rank. Trying older one...\")\n"
            + _NOT_LOADED +
            "            dist.barrier()\n"
            "            raise SystemExit(f\"no checkpoint in {args.train.output_dir}/checkpoints could be loaded: {last_err!r}\")\n",
        ),
    ],
    "lingbotvla/optim/optimizer.py": [
        (
            "        lr_to_params.setdefault(lr_for_param, []).append(p)\n",
            "        # LingBotRoboTwin: newly initialized DINOv2 fusion layers may need a larger lr than the continuation\n"
            "        if \"dino_fusion\" in name:\n"
            "            lr_for_param = lr_for_param * float(os.environ.get(\"LINGBOT_DINO_LR_SCALE\", \"1\"))\n"
            "        lr_to_params.setdefault(lr_for_param, []).append(p)\n",
        ),
    ],
    "tasks/vla/train_lingbotvla.py#rank_seed": [
        (
            "\n\ndef main():\n",
            "\n\ndef _lb_reseed(key, worker_id=None):\n"
            "    # LingBotRoboTwin: seeds torch (CPU + CUDA), random and numpy from `key` (+ a dataloader worker id); see main()\n"
            "    import hashlib\n"
            "    import random\n"
            "    s = int.from_bytes(hashlib.sha256(repr((*key, worker_id)).encode()).digest()[:8], \"little\") >> 1\n"
            "    torch.manual_seed(s)\n"
            "    random.seed(s)\n"
            "    np.random.seed(s % 2**32)\n"
            "\n\ndef main():\n",
        ),
        (
            "        data_iterator = iter(train_dataloader)\n",
            "        # LingBotRoboTwin (10/25): set_seed gives every rank the same seed, so the dataloader workers of all ranks drew\n"
            "        # the same temp:2 (task, frame) and augmentation and the flow-matching noise was the same on every rank (a step\n"
            "        # had 128 / NGPU distinct samples); a resume also restored the workers' base seed but not their position, so it\n"
            "        # drew again what its iterator had drawn since it was created. Workers and the main process (noise) now get\n"
            "        # their own streams from (seed, dp rank, step, epoch); a rerun from the same checkpoint draws the same.\n"
            "        _seed_key = (args.train.seed, get_parallel_state().dp_rank, global_step, epoch)\n"
            "        train_dataloader.worker_init_fn = partial(_lb_reseed, _seed_key)\n"
            "        _lb_reseed(_seed_key + (\"main\",))\n"
            "        logger.info_rank0(f\"reseeded dataloader workers and noise per rank (step {global_step}, epoch {epoch})\")\n"
            "        data_iterator = iter(train_dataloader)\n",
        ),
    ],
}


def main():
    root = Path(sys.argv[1])
    assert (root / ".git").exists(), f"{root} is not a git checkout"
    assert "dino" in root.name, "refusing to patch a checkout whose name does not mark it as the DINO dev copy"
    dst = root / "lingbotvla/models/vla/lingbot_vla/dino_fusion.py"
    shutil.copy2(HERE / "dino_fusion.py", dst)
    print("copied", dst)
    for rel, upgrades in UPGRADES.items():
        path = root / rel
        code = path.read_text()
        for old, new in upgrades:
            if old in code and new not in code:
                assert code.count(old) == 1, f"{rel}: old text found more than once: {old[:80]!r}"
                code = code.replace(old, new)
                path.write_text(code)
                print("upgraded", rel)
    for rel, edits in EDITS.items():
        path = root / rel.split("#")[0]  # "file#tag" allows several edit groups per file
        code = path.read_text()
        changed = False
        for old, new in edits:
            if new in code:
                continue
            assert code.count(old) == 1, f"{rel}: anchor not found exactly once: {old!r}"
            code = code.replace(old, new)
            changed = True
        if changed:
            path.write_text(code)
        print("patched" if changed else "already patched", rel)


if __name__ == "__main__":
    main()
