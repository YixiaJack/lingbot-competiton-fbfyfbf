"""Skill library shared by the scripted expert (labels), the LeRobot conversion (prompts), and the harness (what the
LLM may choose from). No simulator imports.

The VLA is fine-tuned on these exact prompts, so the harness never sends free text to it: the LLM plans and schedules by
choosing skill ids, and the harness looks up the prompt. Which skills to use, in what order, how many sweeps, and when to
retry one are the LLM's decisions.
"""

FULL_INSTRUCTION = 'With the brush and dustpan, sweep up all the paper balls and empty the dustpan into the basket.'

# id -> prompt fed to LingBot, and what "done" looks like (shown to the LLM, never used by the harness itself)
SKILLS = {
    'grasp_dustpan': {
        'prompt': 'Grasp the dustpan handle with the left gripper.',
        'done_when': 'The left gripper is closed around the dustpan handle. The dustpan stays on the table.'},
    'grasp_brush': {
        'prompt': 'Grasp the brush handle with the right gripper.',
        'done_when': 'The right gripper is closed around the brush handle and the brush is lifted slightly off the table.'},
    'sweep_red': {
        'prompt': 'Sweep the red paper ball into the dustpan.',
        'done_when': 'The red paper ball is inside the dustpan and the brush is lifted away from it.'},
    'sweep_white': {
        'prompt': 'Sweep the white paper ball into the dustpan.',
        'done_when': 'The white paper ball is inside the dustpan and the brush is lifted away from it.'},
    'sweep_both': {
        'prompt': 'Sweep both paper balls into the dustpan.',
        'done_when': 'Both paper balls are inside the dustpan and the brush is lifted away. Only sensible when the two '
                     'balls lie side by side, close enough for the brush head to push both at once.'},
    'carry_to_basket': {
        'prompt': 'Lift the dustpan and hold it above the basket.',
        'done_when': 'The dustpan, still holding the paper balls, is held above the round basket.'},
    'pour': {
        'prompt': 'Tip the dustpan to empty the paper balls into the basket.',
        'done_when': 'Both paper balls have fallen out of the dustpan into the basket.'},
    'put_down_and_home': {
        'prompt': 'Put down the brush and the dustpan, then return both arms to the home position.',
        'done_when': 'The dustpan is turned back level and set down, the brush is set down, both grippers are open, and '
                     'both arms are back at their start pose and not moving.'},
}
SKILL_IDS = list(SKILLS)
PROMPT = {k: v['prompt'] for k, v in SKILLS.items()}


def relabel(skill, flags, margin=15):
    """Per-frame skill labels with 'pour' ending `margin` frames (~1 s) after both balls are in the basket; the rest of
    the expert's pour (holding, rolling the dustpan back level) becomes part of put_down_and_home. Why: the balls drop in
    the first 2-3 s of the expert's 13 s pour, and the LLM (both versions, judgement test 11/14) reads "balls in the
    basket" as the end of pouring; the labels now match the moment a harness switches. flags columns: in_pan red/white,
    in_basket red/white, ... (collect.FLAG_NAMES)."""
    import numpy as np
    lab = np.array(skill, dtype=object).copy()
    idx = np.where(lab == 'pour')[0]
    inb = np.asarray(flags)[:, 2].astype(bool) & np.asarray(flags)[:, 3].astype(bool)
    hit = [j for j in idx if inb[j]]
    if hit:
        lab[[j for j in idx if j >= hit[0] + margin]] = 'put_down_and_home'
    return lab


def segments(labels):
    """[(skill, start, end)] runs of equal labels."""
    out, start = [], 0
    for j in range(1, len(labels) + 1):
        if j == len(labels) or labels[j] != labels[start]:
            out.append((str(labels[start]), start, j)); start = j
    return out
