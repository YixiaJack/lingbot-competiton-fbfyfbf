"""Instruction phrasings for the VLA, and the fixed subtask list of harness V2. No simulator imports.

Training data (prepare_data.py): each skill segment gets the canonical prompt (skills.PROMPT) 40% of the time, else one of
the variants below, filled from the scene at the segment's first frame. So the VLA learns the canonical prompt (what V1
and the oracle/timed harnesses send) and also varied, scene-grounded wording (what V2's LLM writes).
"""
import numpy as np
from skills import PROMPT

# {side}: where the named ball lies relative to the other ball, from the head camera ("on the left" / "on the right")
VARIANTS = {
    'grasp_dustpan': [
        'Pick up the dustpan by its handle with the left arm.',
        'Use the left gripper to hold the blue dustpan by the handle.',
        'Left arm: grab the handle of the dustpan.',
        'Take hold of the dustpan handle with your left hand and keep the dustpan on the table.'],
    'grasp_brush': [
        'Pick up the brush by its handle with the right arm.',
        'Use the right gripper to grab the wooden brush.',
        'Right arm: take the brush by the handle and lift it a little.',
        'Grab the brush with your right hand.'],
    'sweep_red': [
        'Use the brush to push the red paper ball into the dustpan.',
        'Sweep the red paper ball {side} into the dustpan.',
        'Brush the red ball into the dustpan.',
        'With the brush, sweep the red crumpled paper into the dustpan.'],
    'sweep_white': [
        'Use the brush to push the white paper ball into the dustpan.',
        'Sweep the white paper ball {side} into the dustpan.',
        'Brush the white ball into the dustpan.',
        'With the brush, sweep the white crumpled paper into the dustpan.'],
    'sweep_both': [
        'Sweep the red and white paper balls into the dustpan together.',
        'Use the brush to push both paper balls into the dustpan in one stroke.',
        'Brush the two paper balls into the dustpan.',
        'Sweep all the paper balls into the dustpan at once.'],
    'carry_to_basket': [
        'Carry the dustpan over the basket.',
        'Lift the dustpan with the paper balls and move it above the round basket.',
        'Bring the dustpan to the basket without spilling.',
        'Move the dustpan over the basket on the left, keeping the paper inside.'],
    'pour': [
        'Empty the dustpan into the basket.',
        'Turn the dustpan over the basket so the paper balls fall in.',
        'Pour the paper balls from the dustpan into the basket.',
        'Dump the paper balls from the dustpan into the basket.'],
    'put_down_and_home': [
        'Level the dustpan, put both tools back on the table and move the arms to their home position.',
        'Place the brush and the dustpan down, release them, and return to home.',
        'Set down the dustpan and the brush, then go back to the starting pose.',
        'Release the tools on the table and return both arms home.'],
}


def ball_side(color, balls_xy):
    """'on the left' / 'on the right' as seen from the robot (smaller world x is the robot's left)."""
    other = [c for c in balls_xy if c != color][0]
    return 'on the left' if balls_xy[color][0] < balls_xy[other][0] else 'on the right'


def sample_prompt(skill, rng, balls_xy=None, p_canonical=0.4):
    if rng.uniform() < p_canonical:
        return PROMPT[skill]
    v = VARIANTS[skill][rng.integers(len(VARIANTS[skill]))]
    if '{side}' in v:
        color = skill.split('_')[1]
        v = v.format(side=ball_side(color, balls_xy)) if balls_xy else v.replace(' {side}', '')
    return v


# ---------------------------------------------------------------------------------------------- harness V2: fixed split
# The five steps of the official reference collection procedure, in order. The LLM only says which one the robot is in
# and writes the instruction for the VLA inside it.
SUBTASKS = [
    {'id': 'grasp', 'name': 'Grasp the tools',
     'goal': 'The left gripper holds the dustpan by its handle (the dustpan stays on the table), then the right gripper '
             'holds the brush by its handle.',
     'steps': 'Two instructions in order: first the dustpan with the left gripper, then the brush with the right gripper.',
     'examples': [PROMPT['grasp_dustpan'], VARIANTS['grasp_dustpan'][1], PROMPT['grasp_brush'], VARIANTS['grasp_brush'][0]]},
    {'id': 'sweep', 'name': 'Sweep the paper balls into the dustpan',
     'goal': 'Both paper balls (red and white) are inside the dustpan, and the brush is lifted away.',
     'steps': 'One ball per instruction, or both in one instruction when they lie side by side close enough for the '
              'brush head to push both. Name the ball by colour; its side (left/right) helps. If a ball rolls out or '
              'is missed, sweep it again.',
     'examples': [PROMPT['sweep_red'], 'Sweep the white paper ball on the left into the dustpan.', PROMPT['sweep_both'],
                  VARIANTS['sweep_red'][0]]},
    {'id': 'carry', 'name': 'Carry the dustpan over the basket',
     'goal': 'The dustpan, still holding both paper balls, is held above the round basket.',
     'steps': 'One instruction.',
     'examples': [PROMPT['carry_to_basket'], VARIANTS['carry_to_basket'][1]]},
    {'id': 'pour', 'name': 'Pour into the basket',
     'goal': 'Both paper balls have fallen out of the dustpan into the basket (none outside it).',
     'steps': 'One instruction.',
     'examples': [PROMPT['pour'], VARIANTS['pour'][2]]},
    {'id': 'home', 'name': 'Put the tools down and return home',
     'goal': 'The dustpan is turned back level and set down, the brush is set down, both grippers are open, both arms '
             'are back at their start pose and still.',
     'steps': 'One instruction. When this is done the whole task is complete.',
     'examples': [PROMPT['put_down_and_home'], VARIANTS['put_down_and_home'][1]]},
]
SUBTASK_IDS = [s['id'] for s in SUBTASKS]
# which training skills a V2 subtask covers (for logging and the offline judgement test)
SUBTASK_OF_SKILL = {'grasp_dustpan': 'grasp', 'grasp_brush': 'grasp', 'sweep_red': 'sweep', 'sweep_white': 'sweep',
                    'sweep_both': 'sweep', 'carry_to_basket': 'carry', 'pour': 'pour', 'put_down_and_home': 'home'}
