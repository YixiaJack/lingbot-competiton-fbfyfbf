"""Team patch 11/16 for ~/robotharness/code/harness.py (the 11/14 20:06 Fixed harness on the 5080). Idempotent
(marker 'hard crumpled balls'). User request 11/16 ~01:00: add only the paper-ball facts to the prompt, nothing else:
the balls are hard; one red and one yellow at most, never more, possibly only one; at the start every ball is fully
visible in the top view. Changes exactly one sentence of FIXED_SYSTEM's scene paragraph; no logic changes.
usage: python3 patch_ball_desc.py <harness.py>"""
import sys
F = sys.argv[1]
s = open(F).read()
if 'hard crumpled balls' in s:
    print('already patched'); sys.exit(0)
OLD = '''one on the left and one on the right (either way round). One or two crumpled paper balls (yellow and/or red) lie \\
between them. A round white basket stands at the back.'''
NEW = '''one on the left and one on the right (either way round). The paper balls are hard crumpled balls: one red and one \\
yellow at most, never more than these two, and a scene may have only one of them. At the start they lie between the \\
tools, and every paper ball of the scene is fully visible in the top image. A round white basket stands at the back.'''
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)
open(F, 'w').write(s)
print('patched')
