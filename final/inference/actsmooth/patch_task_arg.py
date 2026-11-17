import os, sys
F = os.path.expanduser('~/workspace/lingbot/inference/code/deploy/policy_adapter/policy_adapter.py')
s = open(F).read()
if 'task=getattr(self, "_task", None)' in s: print('already'); sys.exit(0)
old = '''                [ACTION_KEYS[f"{s}_gripper.position"] for s in self._arm_sides],
            )'''
new = '''                [ACTION_KEYS[f"{s}_gripper.position"] for s in self._arm_sides],
                task=getattr(self, "_task", None),
            )'''
assert s.count(old) == 1, s.count(old)
open(F, 'w').write(s.replace(old, new)); print('patched')
