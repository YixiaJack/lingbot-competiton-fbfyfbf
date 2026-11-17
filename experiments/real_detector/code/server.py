"""Stock LingBot server with paired per-episode RNG seeds and memory diagnostics."""
import os
import runpy
import sys
import time
import traceback
import subprocess
sys.path.insert(0, os.environ['LINGBOT_DIR'])
os.chdir(os.environ['LINGBOT_DIR'])
import numpy as np
import torch
import websockets
import deploy.websocket_policy_server as wps
from deploy.msgpack_numpy import Packer, unpackb

CAP_GIB = float(os.environ.get('LINGBOT_EVAL_MEMORY_LIMIT_GIB', '0'))
if CAP_GIB:
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1., CAP_GIB * 2**30 / props.total_memory), device=0)


def process_gpu_memory():
    """Sample this server's driver-visible memory; includes more than PyTorch's allocator."""
    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                        '--format=csv,noheader,nounits'], text=True, timeout=5)
        for row in rows.splitlines():
            pid, mib = [x.strip() for x in row.split(',')]
            if pid == str(os.getpid()):
                return float(mib) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


class PairedServer(wps.WebsocketPolicyServer):
    async def _handler(self, websocket):
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        try:
            while True:
                obs = unpackb(await websocket.recv())
                if obs.get('reset'):
                    if CAP_GIB:
                        self._prior_peak = max(getattr(self, '_prior_peak', 0), torch.cuda.max_memory_allocated()/2**30)
                    seed = int(obs.pop('rng_seed', 42))
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                action = self._policy.infer(obs)
                action['server_timing'] = {'infer_ms': (time.monotonic()-start)*1000,
                                          'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                                          'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30}
                if CAP_GIB:
                    action['memory_diagnostics'] = {
                        'allocator_limit_gib': CAP_GIB, 'gpu': props.name,
                        'physical_memory_gib': props.total_memory/2**30,
                        'model_dtype': str(next(self._policy.vla.parameters()).dtype),
                        'lifetime_peak_allocated_gib': max(getattr(self, '_prior_peak', 0), torch.cuda.max_memory_allocated()/2**30),
                        'nvidia_process_after_call_gib': process_gpu_memory()}
                await websocket.send(packer.pack(action))
        except websockets.ConnectionClosed:
            pass
        except Exception:
            traceback.print_exc()
            await websocket.send(traceback.format_exc())
            await websocket.close(code=1011)


# LINGBOT_MEM_CAP_GB=15: cap this process's GPU memory (the venue's RTX 5080 has 16 GB, minus driver/display headroom).
# Set before the model loads; an out-of-memory error then shows up exactly as it would on the 5080.
_cap = os.environ.get('LINGBOT_MEM_CAP_GB')
if _cap:
    _total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(_cap) * 2**30 / _total), 0)
    print(f'GPU memory capped at {_cap} GB of {_total / 2**30:.0f} GB', flush=True)

wps.WebsocketPolicyServer = PairedServer
runpy.run_module('deploy.lingbot_vla_v2_policy', run_name='__main__', alter_sys=True)
