"""Write a patched copy of LingBot's RoboTwin eval client (upstream file is left untouched).

  * EVAL_TEST_NUM   overrides the hardcoded test_num = 100 (default 100 = official; the submitted results use 10).
  * EVAL_STATE_FILE checkpoints the episode loop after every finished episode (next seed, successes, episodes,
    episode id) and restores it on start. Preemptible Slurm jobs get killed and requeued; with this the requeued
    run continues at the next seed instead of starting over. Every episode's scene is set up from its seed, so the
    100 evaluated seeds are exactly the ones an uninterrupted run would use.
  * Expert attempts per episode are capped (organizers' Q&A 13): upstream skips seeds until the scripted expert
    succeeds, without limit. Here each episode gets at most EVAL_EXPERT_MAX_ATTEMPTS (default 5) expert runs, each on
    the next seed; if all fail, the policy is still evaluated on the last scene and the episode counts. If that scene
    cannot be set up at all (the expert run raised, e.g. UnStableError) or no instruction can be generated for it, the
    episode is recorded as a failure. EVAL_EXPERT_MAX_ATTEMPTS=0 restores upstream's unlimited skipping.

  python src/patch_eval_client.py <upstream eval_policy_client_lingbotvla.py> <output path>
"""

import sys

src, dst = sys.argv[1], sys.argv[2]
code = open(src).read()

SAVE_STATE = (
    "        if state_file:\n"
    '            with open(state_file + ".tmp", "w") as f:\n'
    '                json.dump({"now_seed": now_seed, "now_id": now_id, "suc": TASK_ENV.suc,\n'
    '                           "test_num": TASK_ENV.test_num}, f)\n'
    '            os.replace(state_file + ".tmp", state_file)\n'
)

PATCHES = [
    (
        "    test_num = 100\n",
        '    test_num = int(os.environ.get("EVAL_TEST_NUM", "100"))\n',
    ),
    (
        '    args["eval_mode"] = True\n',
        '    args["eval_mode"] = True\n'
        "\n"
        '    state_file = os.environ.get("EVAL_STATE_FILE")\n'
        "    if state_file and os.path.exists(state_file):\n"
        "        with open(state_file) as f:\n"
        "            state = json.load(f)\n"
        '        now_seed, now_id = state["now_seed"], state["now_id"]\n'
        '        TASK_ENV.suc, TASK_ENV.test_num = state["suc"], state["test_num"]\n'
        '        succ_seed = state["test_num"]\n'
        '        print(f"resumed from {state_file}: {state}", flush=True)\n'
        "    # organizers' Q&A 13: at most this many expert runs per episode (0 = unlimited, upstream behaviour)\n"
        '    expert_max_attempts = int(os.environ.get("EVAL_EXPERT_MAX_ATTEMPTS", "5"))\n'
        "    expert_attempts = 0\n",
    ),
    (
        "        # TASK_ENV._take_picture()\n        now_seed += 1\n",
        "        # TASK_ENV._take_picture()\n        now_seed += 1\n" + SAVE_STATE,
    ),
]

# Replace upstream's expert check + episode setup (from "if expert_check:" through the instruction choice).
OLD_START = "        if expert_check:\n            try:\n"
OLD_END = "        instruction = np.random.choice(results[0][instruction_type])\n"
NEW_BLOCK = (
    "        expert_failed_hard = False\n"
    "        if expert_check:\n"
    "            try:\n"
    "                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
    "                episode_info = TASK_ENV.play_once()\n"
    "                TASK_ENV.close_env()\n"
    "            except Exception as e:  # UnStableError or any other error: the expert could not run this scene\n"
    "                TASK_ENV.close_env()\n"
    '                print(f"error occurs ! {e}")\n'
    "                expert_failed_hard = True\n"
    "            expert_ok = (not expert_failed_hard) and TASK_ENV.plan_success and TASK_ENV.check_success()\n"
    "            expert_attempts += 1\n"
    "            if not expert_ok and (expert_max_attempts <= 0 or expert_attempts < expert_max_attempts):\n"
    "                now_seed += 1\n"
    '                args["render_freq"] = render_freq\n'
    "                continue\n"
    "            if not expert_ok:\n"
    '                print(f"expert failed {expert_attempts} times (last seed {now_seed}); "\n'
    '                      f"evaluating the policy on this scene anyway (Q&A 13)", flush=True)\n'
    "            expert_attempts = 0\n"
    "\n"
    "        succ_seed += 1\n"
    "        suc_test_seed_list.append(now_seed)\n"
    '        args["render_freq"] = render_freq\n'
    "\n"
    "        instruction = None\n"
    "        if not expert_failed_hard:\n"
    "            try:\n"
    "                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n"
    '                episode_info_list = [episode_info["info"]]\n'
    '                results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)\n'
    "                instruction = np.random.choice(results[0][instruction_type])\n"
    "            except Exception as e:\n"
    '                print(f"policy episode could not be set up on seed {now_seed}: {e}", flush=True)\n'
    "                try:\n"
    "                    TASK_ENV.close_env()\n"
    "                except Exception:\n"
    "                    pass\n"
    "        if instruction is None:\n"
    "            # no runnable scene / instruction after the expert attempts: the episode counts as a failure\n"
    "            TASK_ENV.test_num += 1\n"
    "            now_id += 1\n"
    "            print(\n"
    "                f\"{task_name} | {args['policy_name']} | {args['task_config']} | episode counted as failure\\n\"\n"
    '                f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} => "\n'
    '                f"{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%, current seed: {now_seed}\\n", flush=True)\n'
    "            now_seed += 1\n"
    + SAVE_STATE.replace("        ", "            ", 1).replace("\n            ", "\n                ")
    + "            continue\n"
)

for old, new in PATCHES:
    assert code.count(old) == 1, f"anchor not found exactly once: {old!r}"
    code = code.replace(old, new)
assert code.count(OLD_START) == 1 and code.count(OLD_END) == 1, "expert-check block anchors not found exactly once"
a = code.index(OLD_START)
b = code.index(OLD_END) + len(OLD_END)
assert a < b and b - a < 4000, "unexpected expert-check block span"
code = code[:a] + NEW_BLOCK + code[b:]
if "\nimport json" not in code:
    code = "import json\n" + code
if "\nimport os" not in code and not code.startswith("import os"):
    code = "import os\n" + code
open(dst, "w").write(code)
print("patched", dst)
