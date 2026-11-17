"""Team patch 11/16 for ~/robotharness/code/harness.py (Fixed harness on the 5080). Idempotent (marker 'lifting them is
part of sweeping'). User request 11/16 ~02:00: a firm grasp does not have to be followed by a lift; lifting must not be
a precondition for leaving the grasp subtask. The grasp goal never mentioned lifting, but in replays the LLM kept the
robot in grasp ("readings low ... tools not yet lifted, grasp not confirmed") until the tools left the table.
Adds one sentence after the subtask list of FIXED_SYSTEM; no logic changes.
usage: python3 patch_grasp_nolift.py <harness.py>"""
import sys
F = sys.argv[1]
s = open(F).read()
if 'lifting them is part of sweeping' in s:
    print('already patched'); sys.exit(0)
OLD = '''The four subtasks, in order:
{subtasks}

Each time you are asked:'''
NEW = '''The four subtasks, in order:
{subtasks}
The grasp goal is reached as soon as both tools are held firmly by their handles. The tools do not have to be lifted \\
off the table first: lifting them is part of sweeping.

Each time you are asked:'''
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)
open(F, 'w').write(s)
print('patched')
