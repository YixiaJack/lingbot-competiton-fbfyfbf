"""Pre-event target-cue experiment: three colored blocks and three colored plates.

This is a controlled RoboTwin proxy for target selection, not competition data.
Training excludes (red,yellow), (green,cyan), (blue,magenta) combinations.
"""
import copy
from pathlib import Path

import numpy as np
import sapien
import yaml
from envs._base_task import Base_Task
from envs.utils import ArmTag, create_box

OBJECTS = ('red', 'green', 'blue')
PLATES = ('yellow', 'cyan', 'magenta')
COLORS = ((1., .05, .05), (.05, .8, .05), (.05, .1, 1.),
          (1., .9, .05), (.05, .9, .9), (.9, .05, .9))
TRAIN_COMBOS = [(i, j) for i in range(3) for j in range(3) if i != j]
HOLDOUT_COMBOS = [(i, i) for i in range(3)]


def config(robotwin, eval_mode=False):
    root = Path(robotwin)
    cfg = yaml.safe_load((root / 'task_config/demo_clean.yml').read_text())
    emb = yaml.safe_load((root / 'task_config/_embodiment_config.yml').read_text())
    robot = emb[cfg['embodiment'][0]]['file_path']
    robot_cfg = yaml.safe_load((Path(robot) / 'config.yml').read_text())
    cfg.update(task_name='target_cue', task_config='demo_clean',
               left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True,
               left_embodiment_config=copy.deepcopy(robot_cfg),
               right_embodiment_config=copy.deepcopy(robot_cfg), eval_mode=eval_mode,
               eval_video_log=False, save_data=False, render_freq=0)
    return cfg


class TargetCue(Base_Task):
    def setup_demo(self, target=0, destination=0, **kwargs):
        self.target_index, self.destination_index = target, destination
        self.frames = []
        super()._init_task_env_(**kwargs)
        self.step_lim = 600
        self.set_instruction(f'Pick up the {OBJECTS[target]} block and place it on the {PLATES[destination]} plate.')

    def load_actors(self):
        # Randomize left/right workspace and color positions independently of the instruction.
        side = np.random.choice([-1, 1])
        xs = side * np.array([.065, .160, .255])
        np.random.shuffle(xs)
        self.blocks = []
        for i in range(3):
            block = create_box(self, pose=sapien.Pose([xs[i], np.random.uniform(.02, .075), .765]),
                               half_size=(.022, .022, .022), color=COLORS[i], name=f'{OBJECTS[i]}_block')
            self.blocks.append(block)
        pxs = side * np.array([.065, .160, .255])
        np.random.shuffle(pxs)
        self.plates = []
        for i in range(3):
            builder = self.scene.create_actor_builder()
            # SAPIEN cylinder axis is x; rotate it upright.
            rot = sapien.Pose(q=[.70710678, 0, .70710678, 0])
            builder.add_cylinder_collision(pose=rot, radius=.038, half_length=.004)
            material = sapien.render.RenderMaterial(base_color=[*COLORS[3+i], 1])
            builder.add_cylinder_visual(pose=rot, radius=.038, half_length=.004, material=material)
            plate = builder.build_static(name=f'{PLATES[i]}_plate')
            plate.set_pose(sapien.Pose([pxs[i], np.random.uniform(-.16, -.12), .744]))
            self.plates.append(plate)

    def play_once(self):
        block = self.blocks[self.target_index]
        arm = ArmTag('left' if block.get_pose().p[0] < 0 else 'right')
        self.move(self.grasp_actor(block, arm_tag=arm, pre_grasp_dis=.09))
        self.move(self.move_by_displacement(arm_tag=arm, z=.09))
        p = self.plates[self.destination_index].get_pose().p
        self.move(self.place_actor(block, target_pose=[float(p[0]), float(p[1]), .751, 0, 1, 0, 0],
                                   arm_tag=arm, functional_point_id=0, pre_dis=.09, dis=.02, constrain='align'))
        self.move(self.move_by_displacement(arm_tag=arm, z=.08))
        return {'info': {}}

    def check_success(self):
        obj = self.blocks[self.target_index].get_pose().p
        dst = self.plates[self.destination_index].get_pose().p
        return bool(np.linalg.norm(obj[:2]-dst[:2]) < .038 and .75 < obj[2] < .82
                    and self.is_left_gripper_open() and self.is_right_gripper_open())

    def _take_picture(self):
        if self.save_data:
            obs = self.get_obs()
            self.frames.append({
                'state': np.asarray(obs['joint_action']['vector'], dtype=np.float32),
                'ee': tcp_positions(self),
                **{k: obs['observation'][c]['rgb'].copy() for k, c in
                   [('cam_high', 'head_camera'), ('cam_left_wrist', 'left_camera'),
                    ('cam_right_wrist', 'right_camera')]},
            })
            self.FRAME_IDX += 1


def boxes(env):
    """Visible-pixel oracle boxes from simulator entity segmentation, never future frames."""
    camera = env.cameras.static_camera_list[env.cameras.head_camera_id]
    seg = camera.get_picture('Segmentation')[..., 1]
    entities = [b.actor for b in env.blocks] + list(env.plates)
    result = []
    for entity in entities:
        yy, xx = np.where(seg == entity.per_scene_id)
        if len(xx) < 4:
            raise ValueError(f'Object {entity.name} not sufficiently visible in frame zero')
        if xx.min()<2 or xx.max()>=seg.shape[1]-2 or yy.min()<2 or yy.max()>=seg.shape[0]-2:
            raise ValueError(f'Object {entity.name} truncated at the frame boundary')
        result.append([int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1])
    return result


def tcp_positions(env):
    """World-frame gripper centers; get_arm_pose is the wrist, offset by 12 cm."""
    return np.array([env.robot.get_left_tcp_pose()[:3], env.robot.get_right_tcp_pose()[:3]])
