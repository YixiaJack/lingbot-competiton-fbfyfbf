"""Stock LingBot server with paired per-episode RNG seeds and memory diagnostics."""
import os
import runpy
import sys
import time
import traceback
sys.path.insert(0, os.environ['LINGBOT_DIR'])
os.chdir(os.environ['LINGBOT_DIR'])
import numpy as np
import torch
import websockets
import deploy.websocket_policy_server as wps
from deploy.msgpack_numpy import Packer, unpackb


class PairedServer(wps.WebsocketPolicyServer):
    async def _handler(self, websocket):
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        try:
            while True:
                obs = unpackb(await websocket.recv())
                if obs.get('reset'):
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
                await websocket.send(packer.pack(action))
        except websockets.ConnectionClosed:
            pass
        except Exception:
            traceback.print_exc()
            await websocket.send(traceback.format_exc())
            await websocket.close(code=1011)


wps.WebsocketPolicyServer = PairedServer
runpy.run_module('deploy.lingbot_vla_v2_policy', run_name='__main__', alter_sys=True)
