"""Patches on lingbotvla/distributed/torch_parallelize.py (code_opt branch): suggestion 4 (FSDP granularity + comm).

p03 LB_FSDP_VISION (default 0 until measured): the vision tower is not sharded per block in the official code (the
    vlm_fsdp loop only walks the LLM layers although Qwen3VLVisionBlock is listed), so all ViT blocks, the patch embed,
    the merger and the deepstack mergers sit in the root FSDP group and are all-gathered together with it. Shard every
    vision block and deepstack merger as its own unit, then the vision tower, then (as before) the root.
    Mixed precision as the suggestion says: same BF16 params / FP32 reduce as the root, cast_forward_inputs=False (the
    rotary cos/sin handed to each block stay in their dtype) and no output cast, so numerics match the root-group run.
p04 LB_FSDP_PREFETCH=N (default 0): explicit forward prefetch of the next N FSDP units in the real call order
    (LLM layer i, action-expert layer i, LLM layer i+1, ...) and backward prefetch of the previous N. With
    reshard_after_forward=False the backward needs no re-gather, so mostly the forward overlap matters.
"""
from pathlib import Path

p = Path.home() / 'lingbot_train/code_opt/lingbotvla/distributed/torch_parallelize.py'
s = p.read_text()
if '\nimport os\n' not in s:
    s = s.replace('\nimport types\n', '\nimport os\nimport types\n', 1)

old = '''            fully_shard(model, **mp_fsdp_kwargs)

            if kwargs.get("init_device") == "meta":'''
new = '''            if os.environ.get("LB_FSDP_VISION", "0") == "1":
                _qwenvl = model.model.qwenvl_with_expert.qwenvl
                _vision = getattr(getattr(_qwenvl, "model", None), "visual", None)
                if _vision is not None:
                    _vision_mp = MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        output_dtype=None,
                        cast_forward_inputs=False,
                    )
                    _vkw = {k: v for k, v in mp_fsdp_kwargs.items() if k != "mp_policy"}
                    _vkw["mp_policy"] = _vision_mp
                    for _blk in _vision.blocks:
                        fully_shard(_blk, **_vkw)
                    _dsm = list(getattr(_vision, "deepstack_merger_list", None) or [])
                    for _m in _dsm:
                        fully_shard(_m, **_vkw)
                    fully_shard(_vision, **_vkw)
                    logger.info_rank0(
                        f"[LB_FSDP_VISION] vision tower: {len(_vision.blocks)} block units + {len(_dsm)} deepstack "
                        f"merger units + vision root, split from the model root group"
                    )

            fully_shard(model, **mp_fsdp_kwargs)

            _pf = int(os.environ.get("LB_FSDP_PREFETCH", "0") or 0)
            if _pf > 0:
                try:
                    _qwenvl = model.model.qwenvl_with_expert.qwenvl
                    _llm = list(_qwenvl.model.language_model.layers)
                    _exp = list(model.model.qwenvl_with_expert.qwen_expert.model.layers)
                    _order = []
                    for _a, _b in zip(_llm, _exp):
                        _order += [_a, _b]
                    _order = [m for m in _order if hasattr(m, "set_modules_to_forward_prefetch")]
                    for _i, _m in enumerate(_order):
                        _nxt = _order[_i + 1:_i + 1 + _pf]
                        if _nxt:
                            _m.set_modules_to_forward_prefetch(_nxt)
                        _prv = _order[max(0, _i - _pf):_i][::-1]
                        if _prv:
                            _m.set_modules_to_backward_prefetch(_prv)
                    logger.info_rank0(f"[LB_FSDP_PREFETCH] depth {_pf} over {len(_order)} decoder units")
                except Exception as e:
                    logger.warning_rank0(f"[LB_FSDP_PREFETCH] not configured: {e!r}")

            if kwargs.get("init_device") == "meta":'''
assert s.count(old) == 1, ('anchor', s.count(old))
s = s.replace(old, new)
p.write_text(s)
print('patched p03+p04', p)
