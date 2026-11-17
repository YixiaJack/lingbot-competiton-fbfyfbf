"""One LingBot-VLA policy server shared by several RoboTwin eval clients on the same GPU (eval_pool.sbatch
SHARED_SERVER=1, started through src/shared_server.sh).

Same model, flags and protocol as upstream's `python -m deploy.lingbot_vla_v2_policy` (its main() is run unchanged);
only the websocket handler differs: every connection (one eval client process) keeps its own copy of the policy's
per-episode inference state (global_step, last_action_chunk, last_normalized_action_chunk), swapped in around each
infer call. infer() blocks the event loop, so calls from different clients are served one at a time and never see
each other's state. (With the eval's default chunk_ret=True every call is a fresh forward pass anyway; the swap keeps
the server correct for use_length stepping too.) Run from $LINGBOT_DIR, which is put on sys.path like `python -m`.

  cd $LINGBOT_DIR && python $WORK_DIR/src/shared_policy_server.py --model_path <hf_ckpt> --port <port> ...
"""
import logging
import os
import runpy
import sys
import time
import traceback

sys.path.insert(0, os.getcwd())

import deploy.websocket_policy_server as wps  # noqa: E402
import websockets  # noqa: E402
import websockets.frames  # noqa: E402
from deploy.msgpack_numpy import Packer, unpackb  # noqa: E402

logger = logging.getLogger(__name__)
STATE = ("global_step", "last_action_chunk", "last_normalized_action_chunk")


class PerConnectionPolicyServer(wps.WebsocketPolicyServer):
    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        policy, packer = self._policy, Packer()
        mine = {"global_step": 0, "last_action_chunk": None, "last_normalized_action_chunk": None}
        await websocket.send(packer.pack(self._metadata))
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())
                infer_time = time.monotonic()
                for k in STATE:
                    setattr(policy, k, mine[k])
                try:
                    action = policy.infer(obs)
                finally:
                    for k in STATE:
                        mine[k] = getattr(policy, k)
                infer_time = time.monotonic() - infer_time
                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # unlike upstream, keep serving the other clients: report to this one and close only its connection
                logger.error(traceback.format_exc())
                try:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                          reason="Internal server error. Traceback included in previous frame.")
                except websockets.ConnectionClosed:
                    pass
                break


wps.WebsocketPolicyServer = PerConnectionPolicyServer  # picked up by the module's own import below
runpy.run_module("deploy.lingbot_vla_v2_policy", run_name="__main__", alter_sys=True)
