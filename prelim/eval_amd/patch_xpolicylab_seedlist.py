"""Write a seed-list copy of the AMD image's RoboTwin eval client (the original file is left untouched).

  python eval_amd/patch_xpolicylab_seedlist.py /RoboTwin/scripts/eval_policy_xpolicylab.py \
         /RoboTwin/scripts/eval_policy_xpolicylab_seedlist.py

Same two additions as LingBotRoboTwin/src/patch_eval_client.py on the NVIDIA side (the copy has to sit next to the
original: the client finds RoboTwin from its own path):
  * EVAL_SEED_LIST=<json {task_config: {task: [[seed, no_scene], ...]}}>: evaluate exactly the listed seeds, in order,
    with no expert planning (the AMD planner is mplib instead of CuRobo and picks other seeds; it is also slow). The
    episode info for the instruction (object ids, which arm; scene-only in all 50 tasks) comes from play_once run with
    plan_success set to False beforehand, so every move returns at once; if that raises, the real expert is run for
    the info. no_scene = 1 (the reference run had no runnable scene) and a listed scene this machine cannot set up count
    as failures. EVAL_SEED_LIST_CHECK=1 also runs the real expert on every scene and prints whether its info matches.
    Pass --test_num = the list length.
  * EVAL_STATE_FILE: the episode loop state (next seed, successes, episodes, episode id) after every episode; a
    restarted client continues from it. Printed as "resumed from <file>: {...}" like the NVIDIA client, so
    LingBotRoboTwin/src/episode_table.py reads both logs.
"""

import sys

src, dst = sys.argv[1], sys.argv[2]
code = open(src).read()

SAVE_STATE = (
    "if state_file:\n"
    '    with open(state_file + ".tmp", "w") as _f:\n'
    '        json.dump({"now_seed": now_seed, "now_id": now_id, "suc": task_env.suc, "test_num": task_env.test_num}, _f)\n'
    '    os.replace(state_file + ".tmp", state_file)\n'
)


def indent(block, n):
    return "".join(" " * n + line if line.strip() else line for line in block.splitlines(True))


FAIL_EPISODE = (  # count the current listed episode as a failure and move on (inside the while loop)
    "{count}"
    "task_env.test_num += 1\n"
    "now_id += 1\n"
    "print(\n"
    "    f\"{task_name} | {args['policy_name']} | {args['task_config']} | episode counted as failure\\n\"\n"
    '    f"Success rate: {task_env.suc}/{task_env.test_num} => "\n'
    '    f"{round(task_env.suc / task_env.test_num * 100, 1)}%, current seed: {now_seed}\\n", flush=True)\n'
    + SAVE_STATE
    + 'args["render_freq"] = render_freq\n'
    "continue\n"
)

PATCHES = [
    (  # 1) after the loop variables are set: seed list + resume state
        "    clear_cache_freq = args[\"clear_cache_freq\"]\n"
        "    args[\"eval_mode\"] = True\n"
        "\n"
        "    while succ_seed < test_num:\n",
        "    clear_cache_freq = args[\"clear_cache_freq\"]\n"
        "    args[\"eval_mode\"] = True\n"
        "\n"
        "    # LingBotRoboTwin: EVAL_SEED_LIST / EVAL_STATE_FILE (see eval_amd/patch_xpolicylab_seedlist.py)\n"
        "    state_file = os.environ.get(\"EVAL_STATE_FILE\")\n"
        "    seed_entries = None\n"
        "    if os.environ.get(\"EVAL_SEED_LIST\"):\n"
        "        with open(os.environ[\"EVAL_SEED_LIST\"]) as _f:\n"
        "            seed_entries = json.load(_f)[args[\"task_config\"]][task_name]\n"
        "        test_num = min(test_num, len(seed_entries))\n"
        "        expert_check = False\n"
        "        print(f\"seed list {os.environ['EVAL_SEED_LIST']}: {len(seed_entries)} seeds, evaluating {test_num}\",\n"
        "              flush=True)\n"
        "    if state_file and os.path.exists(state_file):\n"
        "        with open(state_file) as _f:\n"
        "            state = json.load(_f)\n"
        "        now_seed, now_id = state[\"now_seed\"], state[\"now_id\"]\n"
        "        task_env.suc, task_env.test_num = state[\"suc\"], state[\"test_num\"]\n"
        "        succ_seed = state[\"test_num\"]\n"
        "        print(f\"resumed from {state_file}: {state}\", flush=True)\n"
        "\n"
        "    while succ_seed < test_num:\n",
    ),
    (  # 2) top of the loop: the listed seed and its scene info, without planning
        "        args[\"render_freq\"] = 0\n"
        "        episode_info = {\"info\": {}}\n"
        "\n"
        "        if expert_check:\n"
        "            expert_plan_success = False\n",
        "        args[\"render_freq\"] = 0\n"
        "        episode_info = {\"info\": {}}\n"
        "\n"
        "        if seed_entries is not None:  # LingBotRoboTwin: listed seed, scene info without planning\n"
        "            if task_env.test_num >= len(seed_entries):\n"
        "                break\n"
        "            now_seed, no_scene = seed_entries[task_env.test_num]\n"
        "            episode_info = None\n"
        "            for dry in ((True, False) if not no_scene else ()):\n"
        "                try:  # dry: with plan_success False every move returns at once\n"
        "                    task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
        "                    if dry:\n"
        "                        task_env.plan_success = False\n"
        "                    episode_info = {\"info\": dict(task_env.play_once().get(\"info\", {}))}\n"
        "                    task_env.close_env()\n"
        "                    break\n"
        "                except Exception as e:\n"
        "                    try:\n"
        "                        task_env.close_env()\n"
        "                    except Exception:\n"
        "                        pass\n"
        "                    print(f\"error occurs ! scene info ({'dry' if dry else 'expert'}) for seed {now_seed}: \"\n"
        "                          f\"{type(e).__name__}: {e}\", flush=True)\n"
        "            if episode_info is not None and os.environ.get(\"EVAL_SEED_LIST_CHECK\") == \"1\":\n"
        "                try:\n"
        "                    task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
        "                    check = dict(task_env.play_once().get(\"info\", {}))\n"
        "                    task_env.close_env()\n"
        "                    print(f\"scene info check seed {now_seed}: \" + (\"same\" if check == episode_info[\"info\"] else\n"
        "                          f\"DIFFERENT dry {episode_info['info']} expert {check}\"), flush=True)\n"
        "                except Exception as e:\n"
        "                    try:\n"
        "                        task_env.close_env()\n"
        "                    except Exception:\n"
        "                        pass\n"
        "                    print(f\"scene info check seed {now_seed}: expert raised {type(e).__name__}: {e}\", flush=True)\n"
        "            if episode_info is None:  # no_scene in the reference run, or no scene info here: a failure\n"
        "                if not no_scene:\n"
        "                    print(f\"seed list: no scene info for seed {now_seed}\", flush=True)\n"
        + indent(FAIL_EPISODE.replace("{count}", "succ_seed += 1\n"), 16)
        + "\n"
        "        if expert_check:\n"
        "            expert_plan_success = False\n",
    ),
    (  # 3) a listed scene this machine cannot set up for the policy: a failure (upstream would retry the next seed)
        "        except UnStableError:\n"
        "            task_env.close_env()\n"
        "            if not expert_check:\n"
        "                succ_seed -= 1\n",
        "        except UnStableError:\n"
        "            task_env.close_env()\n"
        "            if seed_entries is not None:  # LingBotRoboTwin\n"
        "                print(f\"policy episode could not be set up on seed {now_seed} (UnStableError)\", flush=True)\n"
        + indent(FAIL_EPISODE.replace("{count}", ""), 16)  # succ_seed was counted before the setup
        + "            if not expert_check:\n"
        "                succ_seed -= 1\n",
    ),
    (  # 4) state after every finished episode
        "        now_seed += 1\n"
        "\n"
        "    return now_seed, task_env.suc\n",
        "        now_seed += 1\n"
        + indent(SAVE_STATE, 8)
        + "\n"
        "    return now_seed, task_env.suc\n",
    ),
]

for old, new in PATCHES:
    assert code.count(old) == 1, f"anchor not found exactly once: {old[:120]!r}"
    code = code.replace(old, new)
open(dst, "w").write(code)
print("wrote", dst)
