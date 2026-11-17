"""Idempotent: insert the actsmooth call at the end of LingBotVLA2Policy.postprocess (5080)."""
import os, sys
F = os.path.expanduser('~/workspace/lingbot/inference/code/deploy/policy_adapter/policy_adapter.py')
s = open(F).read()
if 'actsmooth' in s:
    print('already patched'); sys.exit(0)
old = '''        if chunk.chunk_length < self._use_length:
            raise ValueError("Native output is shorter than use_length")
        return ActionChunk(
            actions={
                key: value[: self._use_length].copy()
                for key, value in chunk.actions.items()
            }
        )
'''
new = '''        if chunk.chunk_length < self._use_length:
            raise ValueError("Native output is shorter than use_length")
        out_actions = {
            key: value[: self._use_length].copy()
            for key, value in chunk.actions.items()
        }
        # Team patch (actsmooth, 11/15): per-chunk arm smoothing + gripper gate. Mode file ~/actsmooth/mode.
        try:
            if getattr(self, "_actsmooth", None) is None:
                import sys as _sys
                _p = os.path.expanduser("~/actsmooth")
                if _p not in _sys.path:
                    _sys.path.insert(0, _p)
                import actsmooth as _actsmooth_mod
                self._actsmooth = _actsmooth_mod.Smoother(os.path.expanduser("~/workspace/lingbot/inference/logs"))
            out_actions = self._actsmooth.apply(
                out_actions,
                [ACTION_KEYS[f"{s}_arm.position"] for s in self._arm_sides],
                [ACTION_KEYS[f"{s}_gripper.position"] for s in self._arm_sides],
            )
        except Exception:
            _rtc_logging.getLogger("lingbot_infer").exception("actsmooth failed; raw chunk used")
        return ActionChunk(actions=out_actions)
'''
assert s.count(old) == 1, s.count(old)
open(F, 'w').write(s.replace(old, new)); print('patched')
