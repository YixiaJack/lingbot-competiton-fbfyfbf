"""Paired BF16 rollout; first-approach proxy + task success, with raw evidence."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scene import TargetCue, config, boxes, TRAIN_COMBOS, HOLDOUT_COMBOS, tcp_positions
from cues import apply_cue, detected_boxes
sys.path.insert(0, os.environ['LINGBOT_DIR'])
from deploy.websocket_client_policy import WebsocketClientPolicy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--mode',choices=['text','box','coords'],required=True)
    p.add_argument('--source',choices=['oracle','yolo','jitter','wrong10'],default='oracle')
    p.add_argument('--item',type=int,required=True)
    p.add_argument('--port',type=int,required=True)
    a=p.parse_args()
    key=f'{a.run}_{a.mode}_{a.source}'
    out=a.root/'results'/key/f'{a.item:04d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        prior=json.loads((out/'result.json').read_text())
        if prior.get('metric_version')==2 and prior.get('scene_fingerprint'): return
    heldout=a.item>=30
    pairs=HOLDOUT_COMBOS if heldout else TRAIN_COMBOS
    target,dest=pairs[(a.item-30 if heldout else a.item)%len(pairs)]
    cfg=config(os.environ['ROBOTWIN_DIR'],eval_mode=True)
    requested_seed=100000+a.item
    scene_rejections=[]
    # Filter only incomplete visibility, independently of the policy and expert success.
    # The fixed seed sequence gives every condition the same accepted initial scene.
    for attempt in range(16):
        seed=requested_seed+1000*attempt
        env=TargetCue()
        env.setup_demo(target=target,destination=dest,seed=seed,now_ep_num=a.item,**cfg)
        obs=env.get_obs()
        try:
            all_boxes=boxes(env)
            break
        except ValueError as exc:
            scene_rejections.append({'seed':seed,'reason':str(exc)})
            env.close_env(clear_cache=True)
    else:
        raise RuntimeError(f'No fully visible scene for item {a.item}: {scene_rejections}')
    initial_image=obs['observation']['head_camera']['rgb']
    target_box,dest_box=all_boxes[target],all_boxes[3+dest]
    instruction=env.get_instruction()
    detector_ms=None
    if a.source=='yolo':
        from ultralytics import YOLO
        detector=YOLO(str(a.root/'detector/yolo11n/weights/best.pt'))
        start=time.perf_counter()
        target_box,dest_box=detected_boxes(detector,initial_image,instruction)
        detector_ms=1000*(time.perf_counter()-start)
    elif a.source=='jitter':
        rng=np.random.default_rng(seed+777)
        def jitter(b):
            x1,y1,x2,y2=b; shift=rng.uniform(-.1,.1,2)*[x2-x1,y2-y1]
            return [max(0,x1+shift[0]),max(0,y1+shift[1]),min(320,x2+shift[0]),min(240,y2+shift[1])]
        target_box,dest_box=jitter(target_box),jitter(dest_box)
    elif a.source=='wrong10' and a.item%10==0:
        target_box=all_boxes[(target+1)%3]
    displayed,prompt=apply_cue(initial_image,instruction,target_box,dest_box,a.mode)
    Image.fromarray(initial_image).save(out/'frame0.png')
    Image.fromarray(displayed).save(out/'cue.png')
    policy=WebsocketClientPolicy(port=a.port)
    policy.infer(dict(reset=True,robo_name='robotwin_clean',path_to_pi_model=None,rng_seed=seed+17))
    initial_positions=np.array([b.get_pose().p for b in env.blocks])
    initial_scene={'objects':initial_positions.round(4).tolist(),
                   'plates':np.array([plate.get_pose().p for plate in env.plates]).round(4).tolist(),
                   'joints':np.asarray(obs['joint_action']['vector']).round(4).tolist()}
    scene_fingerprint=hashlib.sha256(json.dumps(initial_scene,sort_keys=True).encode()).hexdigest()
    initial_ee=tcp_positions(env)
    initial_dist=np.linalg.norm(initial_ee[:,None,:]-initial_positions[None,:,:],axis=-1)
    first=None; trace=[]; latencies=[]; memory=[]; frames=[]
    start=time.time()
    while env.take_action_cnt<env.step_lim and not env.eval_success:
        obs=env.get_obs()
        high,prompt=apply_cue(obs['observation']['head_camera']['rgb'],instruction,target_box,dest_box,a.mode)
        ret=policy.infer({'observation.images.cam_high':high,
                          'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
                          'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb'],
                          'observation.state':obs['joint_action']['vector'],'task':prompt})
        timing=ret['server_timing'];latencies.append(timing['infer_ms']);memory.append(timing)
        action=np.asarray(ret['action'])
        if action.ndim==1: action=action[None]
        for act in action:
            env.take_action(act)
            ee=tcp_positions(env)
            dist=np.linalg.norm(ee[:,None,:]-initial_positions[None,:,:],axis=-1)
            candidates=np.argwhere((dist<.10)&((initial_dist-dist)>.04)&(ee[:,2,None]<.90))
            if first is None and len(candidates):
                arm,obj=min(candidates,key=lambda x:dist[tuple(x)])
                first={'object':int(obj),'arm':('left','right')[arm],'step':env.take_action_cnt,'distance_m':float(dist[arm,obj])}
            trace.append({'step':env.take_action_cnt,'ee':ee.tolist(),'distances':dist.tolist()})
            if env.take_action_cnt%10==0:
                frames.append(env.get_obs()['observation']['head_camera']['rgb'])
            if env.eval_success or env.take_action_cnt>=env.step_lim: break
    import cv2
    if frames:
        h,w=frames[0].shape[:2]
        video=cv2.VideoWriter(str(out/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(w,h))
        for frame in frames: video.write(frame[...,::-1])
        video.release()
    result={'run':a.run,'mode':a.mode,'source':a.source,'item':a.item,'seed':seed,
            'requested_seed':requested_seed,'scene_rejections':scene_rejections,
            'split':'heldout_combo' if heldout else 'seen_combo','target':target,'destination':dest,
            'instruction':instruction,'prompt':prompt,'boxes_used':[target_box,dest_box],
            'cue_applied':a.mode!='text' and target_box is not None and dest_box is not None,
            'first_approach':first,'first_approach_correct':first is not None and first['object']==target,
            'success':bool(env.eval_success),'steps':env.take_action_cnt,'seconds':time.time()-start,
            'detector_ms':detector_ms,'latency_ms':latencies,'memory':memory,'precision':'BF16',
            'metric_version':2,
            'initial_scene':initial_scene,'scene_fingerprint':scene_fingerprint,
            'metric_definition':'first gripper TCP within 10cm of an initial candidate, >4cm progress, TCP z<0.90m; no approach counts incorrect'}
    (out/'trace.json').write_text(json.dumps(trace))
    tmp=out/'result.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'result.json')
    env.close_env(clear_cache=True)
    print(json.dumps({k:result[k] for k in ('run','mode','source','item','first_approach_correct','success','seconds')}),flush=True)


if __name__=='__main__': main()
