"""Bimanual sweeping scene (final-round task 3) in RoboTwin, aloha-agilex, competition-like look (green backdrop and
cloth, head D435 320x240). Objects as in the official list: one dustpan + brush set, a red and a white paper ball, one
round basket.

Tools are built from boxes so that their collision shapes work for sweeping (a convex hull of a dustpan fills the scoop,
and a ball could never get in). Layout, all ours (the venue layout is unknown):
  dustpan  centre-back, mouth toward the robot (-y), handle to the left (-x): the left arm holds it
  brush    right, handle toward the robot: the right arm pushes balls forward (+y) into the mouth
  balls    in front of the mouth, side by side
  basket   left-front: the left arm carries the dustpan there and rolls it about the handle to pour
Rolling about the handle is the strong direction for a two-finger grasp (the pads hold the square handle); the weight of
the pan acts about the finger axis and is held by friction, as for RoboTwin's hammer.

World frame (RoboTwin): x right, y away from the robot, z up, robot base at y = -0.65, table top z = 0.74.
"""
import os
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d
from envs._base_task import Base_Task
from envs.utils import ArmTag
from envs.utils.action import Action
from envs.utils.actor_utils import Actor
from envs.utils.create_actor import create_box, create_table

from skills import FULL_INSTRUCTION

TABLE_TOP = 0.74
GREEN_WALL, GREEN_TABLE = (0.16, 0.55, 0.38), (0.20, 0.60, 0.40)
ASSETS = Path(os.environ.get('REAL_ASSETS', os.path.expandvars('$SCRATCH/lingbot/final_real_20261112/assets')))

# dustpan, local frame: origin at the floor centre on the table, mouth toward local -y, handle along local -x
PAN_W, PAN_D, PAN_H, PAN_T = 0.19, 0.11, 0.045, 0.004       # inner width (x), depth (y), wall height, wall thickness
PAN_FLOOR = 0.003
HANDLE_L, HANDLE_R, HANDLE_Z = 0.09, 0.011, 0.052            # handle length, half width (across the fingers), centre height
HANDLE_HZ = 0.014                                            # handle half height: taller pads contact resist slip
HANDLE_X = -(PAN_W / 2 + 2 * PAN_T + HANDLE_L / 2)           # handle centre x (local)
PAN_GRIP = (HANDLE_X + 0.015, 0.0, HANDLE_Z)          # near the socket: shorter lever for the pour                  # where the left gripper closes
# brush, local frame: origin under the head centre, sweeping face toward local +y, handle along local -y
HEAD_HW, HEAD_HD, BRISTLE_H, HEAD_H = 0.040, 0.015, 0.016, 0.028
BRUSH_HANDLE_L, BRUSH_HANDLE_R, BRUSH_HANDLE_Z = 0.09, 0.009, 0.035
BRUSH_GRIP = (0.0, -(HEAD_HD + 0.055), BRUSH_HANDLE_Z)
# basket and balls
BASKET_R, BASKET_H, BASKET_T = 0.085, 0.075, 0.006           # inner radius, wall height, wall thickness
BALL_R = 0.028
BALLS = ('red', 'white')

PLASTIC_BLUE, HANDLE_BLUE = (0.16, 0.36, 0.78, 1), (0.10, 0.25, 0.60, 1)
BRISTLE, WOOD = (0.12, 0.12, 0.12, 1), (0.62, 0.46, 0.30, 1)
WICKER = (0.56, 0.42, 0.27, 1)


def yaw_pose(x, y, z, yaw):
    return sapien.Pose([x, y, z], t3d.euler.euler2quat(0, 0, yaw))


def mat(pose):
    return pose.to_transformation_matrix() if isinstance(pose, sapien.Pose) else np.asarray(pose)


def pose7(m):
    """4x4 -> [x, y, z, qw, qx, qy, qz] (RoboTwin's target-pose format)."""
    return list(m[:3, 3]) + list(t3d.quaternions.mat2quat(m[:3, :3]))


def ee_mat(robot, arm):
    p = robot.get_left_ee_pose() if arm == 'left' else robot.get_right_ee_pose()
    m = np.eye(4); m[:3, :3] = t3d.quaternions.quat2mat(p[3:]); m[:3, 3] = p[:3]
    return m


def rot_about(axis, angle, point):
    """4x4 rotation by angle about the line through point along axis."""
    m = np.eye(4); m[:3, :3] = t3d.axangles.axangle2mat(axis, angle)
    m[:3, 3] = np.asarray(point) - m[:3, :3] @ np.asarray(point)
    return m


def _topdown(point, yaws):
    """Contact frames for a top-down grasp at point (actor frame), one per gripper yaw (RoboTwin convention: the grasp
    pose is contact @ [[0,0,1],[-1,0,0],[0,-1,0]], approach along its x axis, end effector 0.12 m behind)."""
    m0 = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    out = []
    for t in yaws:
        rz = np.eye(4); rz[:3, :3] = t3d.euler.euler2mat(0, 0, t)
        m = rz @ m0; m[:3, 3] = point; out.append(m.tolist())
    return out


def _actor_data(grip):
    return {'center': [0, 0, 0], 'extents': [0.1, 0.1, 0.1], 'scale': [1, 1, 1],
            'target_pose': [np.eye(4).tolist()], 'contact_points_pose': _topdown(grip, [0, np.pi / 2, np.pi, -np.pi / 2]),
            'transform_matrix': np.eye(4).tolist(), 'functional_matrix': [np.eye(4).tolist()],
            'contact_points_description': [], 'contact_points_group': [[0, 1, 2, 3]], 'contact_points_mask': [True],
            'target_point_description': ['origin']}


def compound(scene, name, boxes, pose, static=False, mass=None, friction=None):
    """One rigid body from boxes [(centre, half_size, rgba)]; density set so the whole body weighs `mass`."""
    b = scene.create_actor_builder()
    b.set_physx_body_type('static' if static else 'dynamic')
    vol = sum(8 * np.prod(h) for _, h, _ in boxes)
    dens = (mass / vol) if mass else 1000.0
    pm = sapien.physx.PhysxMaterial(*friction) if friction else None
    for c, h, rgba in boxes:
        b.add_box_collision(pose=sapien.Pose(c), half_size=h, material=pm, density=dens)
        b.add_box_visual(pose=sapien.Pose(c), half_size=h, material=sapien.render.RenderMaterial(base_color=list(rgba), roughness=0.6))
    e = b.build(name=name)
    e.set_pose(pose)
    return e


def build_dustpan(scene, pose):
    W, D, H, T = PAN_W, PAN_D, PAN_H, PAN_T
    boxes = [([0, 0, PAN_FLOOR / 2], [W / 2 + T, D / 2, PAN_FLOOR / 2], PLASTIC_BLUE),                       # floor
             ([0, D / 2 + T / 2, H / 2], [W / 2 + T, T / 2, H / 2], PLASTIC_BLUE),                            # back wall
             ([-(W / 2 + T / 2), 0, H / 2], [T / 2, D / 2, H / 2], PLASTIC_BLUE),                            # left wall
             ([W / 2 + T / 2, 0, H / 2], [T / 2, D / 2, H / 2], PLASTIC_BLUE),                               # right wall
             ([-(W / 2 + T * 1.5), 0, HANDLE_Z - 0.008], [T / 2, 0.012, 0.014], HANDLE_BLUE),                 # handle socket
             ([HANDLE_X, 0, HANDLE_Z], [HANDLE_L / 2, HANDLE_R, HANDLE_HZ], HANDLE_BLUE)]                    # handle
    # light, high-friction (rubberised) handle: the pan's weight acts about the finger axis and is held by friction only
    return compound(scene, 'dustpan', boxes, pose, mass=0.030, friction=(2.0, 1.8, 0.0))


def build_brush(scene, pose):
    hl = BRUSH_HANDLE_L
    boxes = [([0, 0, BRISTLE_H / 2], [HEAD_HW, HEAD_HD, BRISTLE_H / 2], BRISTLE),                             # bristles
             ([0, 0, (BRISTLE_H + HEAD_H) / 2], [HEAD_HW + 0.002, HEAD_HD + 0.002, (HEAD_H - BRISTLE_H) / 2], WOOD),
             ([0, -(HEAD_HD + hl / 2), BRUSH_HANDLE_Z], [BRUSH_HANDLE_R, hl / 2, BRUSH_HANDLE_R], WOOD),        # handle
             ([0, -(HEAD_HD + hl - 0.008), (BRUSH_HANDLE_Z - BRUSH_HANDLE_R) / 2],
              [0.008, 0.008, (BRUSH_HANDLE_Z - BRUSH_HANDLE_R) / 2], WOOD)]                                    # foot
    return compound(scene, 'brush', boxes, pose, mass=0.035, friction=(1.0, 0.9, 0.0))


def build_basket(scene, xy):
    n, r = 24, BASKET_R + BASKET_T / 2
    b = scene.create_actor_builder(); b.set_physx_body_type('static')
    m = sapien.render.RenderMaterial(base_color=list(WICKER), roughness=0.8)
    floor = sapien.Pose([0, 0, 0.002], [0.70710678, 0, 0.70710678, 0])   # cylinder axis x -> z
    b.add_cylinder_collision(pose=floor, radius=BASKET_R + BASKET_T, half_length=0.002)
    b.add_cylinder_visual(pose=floor, radius=BASKET_R + BASKET_T, half_length=0.002, material=m)
    half = [BASKET_T / 2, r * np.tan(np.pi / n) + 0.002, BASKET_H / 2]
    for i in range(n):
        a = 2 * np.pi * i / n
        p = sapien.Pose([r * np.cos(a), r * np.sin(a), BASKET_H / 2], t3d.euler.euler2quat(0, 0, a))
        b.add_box_collision(pose=p, half_size=half); b.add_box_visual(pose=p, half_size=half, material=m)
    e = b.build(name='basket')
    e.set_pose(sapien.Pose([xy[0], xy[1], TABLE_TOP]))
    return e


def build_ball(scene, color, xy):
    b = scene.create_actor_builder(); b.set_physx_body_type('dynamic')
    b.add_sphere_collision(radius=BALL_R, material=sapien.physx.PhysxMaterial(0.8, 0.6, 0.0), density=0.006 / (4 / 3 * np.pi * BALL_R ** 3))
    b.add_visual_from_file(filename=str(ASSETS / f'paper_ball_{color}.glb'))
    e = b.build(name=f'paper_ball_{color}')
    e.set_pose(sapien.Pose([xy[0], xy[1], TABLE_TOP + BALL_R + 0.001]))
    body = e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    body.linear_damping, body.angular_damping = 1.0, 6.0   # crumpled paper barely rolls (same damping as the fruit assets)
    return e


def set_finger_friction(robot, static, dynamic):
    """Rubber-pad friction on both grippers' finger links (fl/fr_link7, 8). PhysX averages the two materials in a
    contact, so a high-friction handle alone was not enough: the dustpan still slipped about the finger axis."""
    m = sapien.physx.PhysxMaterial(static, dynamic, 0.0)
    n = 0
    for ent in (robot.left_entity, robot.right_entity):
        for link in ent.get_links():
            if link.get_name() in ('fl_link7', 'fl_link8', 'fr_link7', 'fr_link8'):
                for shape in link.get_collision_shapes():
                    shape.set_physical_material(m); n += 1
    return n


def sample_layout(rng):
    """Randomised layout; every object stays inside the head camera's view (x in [-0.31, 0.24] on the table).
    Balls (pan frame, in front of the mouth): 40% side by side (one stroke of the brush head covers both), 60% apart
    (one stroke each, and a stroke at one ball never touches the other)."""
    for _ in range(4000):
        pan = (rng.uniform(0.00, 0.03), rng.uniform(0.05, 0.08), rng.uniform(0.0, 0.08))   # x >= 0.034 with yaw <= 0: no left grasp (seeds 11, 14, 16, 20, 21)
        basket = (rng.uniform(-0.21, -0.19), rng.uniform(-0.13, -0.09))
        brush = (rng.uniform(0.185, 0.195), rng.uniform(-0.03, 0.0), rng.uniform(-0.15, 0.15))
        if rng.uniform() < 0.4:
            c, dx, cy = rng.uniform(-0.03, 0.03), rng.uniform(2 * BALL_R + 0.004, 2 * BALL_R + 0.010), rng.uniform(-0.16, -0.11)
            dy = rng.uniform(-0.015, 0.015)
            loc = [(c - dx / 2, cy - dy / 2), (c + dx / 2, cy + dy / 2)]
        else:
            loc = [(rng.uniform(-0.055, -0.040), rng.uniform(-0.17, -0.10)), (rng.uniform(0.040, 0.055), rng.uniform(-0.17, -0.10))]
        if rng.uniform() < 0.5:
            loc = loc[::-1]
        cs, sn = np.cos(pan[2]), np.sin(pan[2])
        balls = [(pan[0] + cs * lx - sn * ly, pan[1] + sn * lx + cs * ly) for lx, ly in loc]
        if any(np.hypot(b[0] - basket[0], b[1] - basket[1]) < BASKET_R + BASKET_T + BALL_R + 0.012 for b in balls):
            continue
        if any(b[0] > brush[0] - HEAD_HW - BALL_R - 0.02 for b in balls):
            continue
        return {'pan': pan, 'basket': basket, 'brush': brush, 'balls': {'red': balls[0], 'white': balls[1]}}
    raise ValueError('no layout')


class SweepTask(Base_Task):
    """Final-round task 3 in simulation. setup_demo(seed=..., **config(...))."""

    def setup_demo(self, **kwargs):
        self.frames, self.cur_skill, self.skill_log = [], None, []
        super()._init_task_env_(**kwargs)
        self.step_lim = 100000            # the harness enforces the 2-minute clock, not RoboTwin
        self.set_instruction(FULL_INSTRUCTION)
        self.home_q = np.asarray(self.get_obs()['joint_action']['vector'], dtype=np.float64)

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        self.wall_texture, self.table_texture = None, None
        self.wall = create_box(self.scene, sapien.Pose(p=[0, 1, 1.5]), half_size=[3, 0.6, 1.5], color=GREEN_WALL,
                               name='wall', is_static=True)
        self.table = create_table(self.scene, sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
                                  length=1.2, width=0.7, height=table_height, thickness=0.05, color=GREEN_TABLE,
                                  is_static=True)

    def load_actors(self):
        lay = sample_layout(np.random)   # np.random is seeded by _init_task_env_
        self.layout = lay
        px, py, pyaw = lay['pan']
        self.pan_start = yaw_pose(px, py, TABLE_TOP, pyaw)
        self.pan_e = build_dustpan(self.scene, self.pan_start)
        bx, by, byaw = lay['brush']
        self.brush_start = yaw_pose(bx, by, TABLE_TOP, byaw)
        self.brush_e = build_brush(self.scene, self.brush_start)
        self.basket_xy = np.array(lay['basket'])
        self.basket_e = build_basket(self.scene, self.basket_xy)
        self.balls = {c: build_ball(self.scene, c, lay['balls'][c]) for c in BALLS}
        mass = lambda e: e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass
        self.pan = Actor(self.pan_e, _actor_data(PAN_GRIP), mass=mass(self.pan_e))
        self.brush = Actor(self.brush_e, _actor_data(BRUSH_GRIP), mass=mass(self.brush_e))
        set_finger_friction(self.robot, 2.0, 1.8)

    # ------------------------------------------------------------------ privileged state (evaluation / oracle only)
    def gripper_contacts(self):
        """{'dustpan': {'left': n, 'right': n}, 'brush': {...}}: contact points between each tool and each gripper."""
        # aloha-agilex: left arm links fl_link1..8, right fr_link1..8; 7 and 8 are the fingers, 6 the gripper base
        left, right = {f'fl_link{i}' for i in (6, 7, 8)}, {f'fr_link{i}' for i in (6, 7, 8)}
        out = {'dustpan': {'left': 0, 'right': 0}, 'brush': {'left': 0, 'right': 0}}
        for c in self.scene.get_contacts():
            a, b = c.bodies[0].entity.name, c.bodies[1].entity.name
            for tool in out:
                if tool in (a, b):
                    other = b if a == tool else a
                    if other in left: out[tool]['left'] += len(c.points)
                    elif other in right: out[tool]['right'] += len(c.points)
        return out

    def ball_local(self, color):
        p = self.balls[color].get_pose().p
        return (np.linalg.inv(mat(self.pan_e.get_pose())) @ np.append(p, 1))[:3]

    def in_pan(self, color):
        x, y, z = self.ball_local(color)
        return bool(abs(x) < PAN_W / 2 and -PAN_D / 2 + 0.008 < y < PAN_D / 2 and -0.01 < z < PAN_H + 0.03)

    def in_basket(self, color):
        p = self.balls[color].get_pose().p
        return bool(np.hypot(*(p[:2] - self.basket_xy)) < BASKET_R and TABLE_TOP < p[2] < TABLE_TOP + BASKET_H + 0.01)

    def progress(self):
        """Per-frame flags for the rubric (never shown to the harness's LLM)."""
        g = self.gripper_contacts()
        lv, rv = self.robot.get_left_gripper_val(), self.robot.get_right_gripper_val()
        pan_p = np.array(self.pan_e.get_pose().p)
        pan_up = mat(self.pan_e.get_pose())[:3, 2]
        brush_p = np.array(self.brush_e.get_pose().p)
        q = np.asarray(self.robot.get_left_arm_jointState() + self.robot.get_right_arm_jointState())
        dq = np.abs(q - self.home_q)
        pan_hold = {a: bool(g['dustpan'][a] > 0 and v < 0.8) for a, v in (('left', lv), ('right', rv))}
        brush_hold = {a: bool(g['brush'][a] > 0 and v < 0.8) for a, v in (('left', lv), ('right', rv))}
        f = {
            'pan_held': pan_hold, 'brush_held': brush_hold,
            'pan_lifted': bool(pan_p[2] > TABLE_TOP + 0.03),
            'in_pan': {c: self.in_pan(c) for c in BALLS},
            'in_basket': {c: self.in_basket(c) for c in BALLS},
            'pan_over_basket': bool(np.hypot(*(pan_p[:2] - self.basket_xy)) < BASKET_R + 0.04 and pan_p[2] > TABLE_TOP + BASKET_H),
            'ball_z': {c: float(self.balls[c].get_pose().p[2]) for c in BALLS},
            'pan_resting': bool(pan_p[2] < TABLE_TOP + 0.01 and pan_up[2] > 0.97),
            'brush_resting': bool(brush_p[2] < TABLE_TOP + 0.01),
            'grippers_open': bool(lv > 0.8 and rv > 0.8),
            'arms_home': bool(max(dq[:6].max(), dq[7:13].max()) < 0.08),
            'pan_xyz': pan_p.round(4).tolist(), 'brush_xyz': brush_p.round(4).tolist(),
            'balls_xyz': {c: np.round(self.balls[c].get_pose().p, 4).tolist() for c in BALLS},
        }
        f['all_in_pan'] = all(f['in_pan'].values())
        f['all_in_basket'] = all(f['in_basket'].values())
        f['tools_down'] = bool(f['pan_resting'] and f['brush_resting'] and not any(pan_hold.values()) and not any(brush_hold.values()))
        return f

    def check_success(self):
        f = self.progress()
        return f['all_in_basket']

    # ------------------------------------------------------------------ data recording
    def skill(self, name):
        self.cur_skill = name
        self.skill_log.append({'skill': name, 'frame': len(self.frames)})

    def _take_picture(self):
        if self.save_data:
            obs = self.get_obs()
            f = self.progress()
            self.frames.append({
                'state': np.asarray(obs['joint_action']['vector'], dtype=np.float32),
                'skill': self.cur_skill,
                'flags': np.array([f['in_pan']['red'], f['in_pan']['white'], f['in_basket']['red'], f['in_basket']['white'],
                                   f['pan_held']['left'], f['brush_held']['right'], f['pan_lifted'], f['pan_over_basket']]),
                'balls_xy': np.array([f['balls_xyz'][c][:2] for c in BALLS], dtype=np.float32),
                **{k: obs['observation'][c]['rgb'].copy() for k, c in
                   [('cam_high', 'head_camera'), ('cam_left_wrist', 'left_camera'), ('cam_right_wrist', 'right_camera')]},
            })
            self.FRAME_IDX += 1

    # ------------------------------------------------------------------ scripted expert
    def held(self, tool, arm):
        """Tool pose in the end-effector frame while grasped (rigid as long as the grasp holds)."""
        return np.linalg.inv(ee_mat(self.robot, arm)) @ mat(tool.get_pose())

    def tool_target(self, arm, T_tool_ee, tool_world):
        return pose7(tool_world @ np.linalg.inv(T_tool_ee))

    def move_tool(self, arm, T_tool_ee, tool_world, tag=''):
        ok = self.move(self.move_to_pose(ArmTag(arm), self.tool_target(arm, T_tool_ee, tool_world)))
        self.log_move(tag or f'{self.cur_skill}:{arm}', ok, tool_world[:3, 3])
        if getattr(self, 'debug_moves', None) is not None:   # commanded vs actual tool pose, ball positions
            tool = self.brush_e if arm == 'right' else self.pan_e
            act = mat(tool.get_pose())
            ang = np.degrees(np.arccos(np.clip((np.trace(act[:3, :3].T @ tool_world[:3, :3]) - 1) / 2, -1, 1)))
            self.debug_moves.append({'tag': tag, 'ok': bool(ok), 'cmd': np.round(tool_world[:3, 3], 3).tolist(),
                                     'err_mm': np.round(1000 * (act[:3, 3] - tool_world[:3, 3]), 1).tolist(), 'rot_err_deg': round(float(ang), 1),
                                     'balls': {c: np.round(self.balls[c].get_pose().p, 3).tolist() for c in BALLS},
                                     'in_pan': {c: self.in_pan(c) for c in BALLS}})
        return ok

    def log_move(self, tag, ok, where=None):
        if not ok and not getattr(self, 'first_fail', None):
            self.first_fail = {'tag': tag, 'skill': self.cur_skill, 'frame': len(self.frames),
                               'where': None if where is None else np.round(where, 3).tolist()}

    def brush_world(self, pan_m, lx, ly, h):
        """Brush pose with its head centre at (lx, ly) in the pan frame, h above the table, facing into the mouth."""
        m = pan_m.copy(); m[:3, 3] = (pan_m @ np.array([lx, ly, 0, 1]))[:3]; m[2, 3] = TABLE_TOP + h
        return m

    def sweep_plan(self):
        """Which strokes the expert makes: one stroke when the head at one ball would touch the other (side by side),
        else one per ball in random order. Returns [(skill, [colours])]."""
        out = [c for c in BALLS if not self.in_pan(c)]
        if len(out) == 2:
            xs = [self.ball_local(c)[0] for c in out]
            hx = float(np.clip(xs[0], -self.lim(), self.lim()))
            if abs(xs[1] - hx) < HEAD_HW + BALL_R + 0.004:
                return [('sweep_both', out)]
            if np.random.uniform() < 0.5:
                out = out[::-1]
        return [(f'sweep_{c}', [c]) for c in out]

    @staticmethod
    def lim():
        return PAN_W / 2 - HEAD_HW - 0.006   # head centre range that fits between the pan's side walls

    def do_sweep(self, colours, T_brush):
        pan_m = mat(self.pan_e.get_pose())
        loc = np.array([self.ball_local(c) for c in colours])
        lx = float(np.clip(loc[:, 0].mean(), -self.lim(), self.lim()))
        y0 = float(loc[:, 1].min()) - BALL_R - HEAD_HD - 0.015
        y1 = -PAN_D / 2 + 0.005   # head just past the lip: the balls are fully in; deeper, they ride up onto the head
        self.move_tool('right', T_brush, self.brush_world(pan_m, lx, y0, 0.10), 'sweep:above')    # clear of both balls
        self.move_tool('right', T_brush, self.brush_world(pan_m, lx, y0, 0.006), 'sweep:down')
        n = max(2, int(np.ceil((y1 - y0) / 0.025)))
        for y in np.linspace(y0, y1, n + 1)[1:]:
            h = 0.006 if y < -PAN_D / 2 - 0.01 else 0.008   # over the pan floor (3 mm): keep clear of it
            self.move_tool('right', T_brush, self.brush_world(pan_m, lx, y, h), 'sweep:push')
        # back off along the table first (lifting at the balls scoops them up), then straight up: the planner does not
        # know the dustpan, so a diagonal path would cross it
        self.move_tool('right', T_brush, self.brush_world(pan_m, lx, y1 - 0.04, 0.008), 'sweep:back')
        self.move_tool('right', T_brush, self.brush_world(pan_m, lx, y1 - 0.04, 0.10), 'sweep:lift')

    def park_brush(self, T_brush):
        m = mat(self.brush_start).copy(); m[:3, 3] = [0.15, -0.09, TABLE_TOP + 0.10]
        self.move_tool('right', T_brush, m, 'park')

    def pan_pose(self, grip_world, roll, T_grip_pan):
        """Pan pose with its grip point at grip_world, rolled about the handle axis by `roll` from level."""
        yaw = self.layout['pan'][2]
        R = np.eye(4); R[:3, :3] = t3d.euler.euler2mat(0, 0, yaw) @ t3d.axangles.axangle2mat([1, 0, 0], roll)
        m = R.copy(); m[:3, 3] = np.asarray(grip_world) - R[:3, :3] @ T_grip_pan
        return m

    def plan_ok(self, arm, T_tool_ee, tool_world):
        f = self.robot.left_plan_path if arm == 'left' else self.robot.right_plan_path
        return f(self.tool_target(arm, T_tool_ee, tool_world))['status'] == 'Success'

    def first_feasible(self, arm, T_tool_ee, candidates, tag):
        """Execute the first candidate tool pose the planner can reach (probe_reach.py: the far, high poses are not)."""
        if not self.plan_success:
            return False
        for m in candidates:
            if self.plan_ok(arm, T_tool_ee, m):
                return self.move_tool(arm, T_tool_ee, m, tag)
        self.log_move(tag, False, candidates[0][:3, 3]); self.plan_success = False
        return False

    def play_once(self):
        L, R = ArmTag('left'), ArmTag('right')
        self.skill('grasp_dustpan')
        self.log_move('grasp_pan', self.move(self.grasp_actor(self.pan, arm_tag=L, pre_grasp_dis=0.08, contact_point_id=self.PAN_CP)))
        self.delay(1, save_freq=-1)
        self.skill('grasp_brush')
        self.log_move('grasp_brush', self.move(self.grasp_actor(self.brush, arm_tag=R, pre_grasp_dis=0.08, contact_point_id=self.BRUSH_CP)))
        self.log_move('lift_brush', self.move(self.move_by_displacement(R, z=0.04)))
        if not self.plan_success:
            return {'info': {'failed': 'grasp'}}
        T_brush = self.held(self.brush, 'right')
        for attempt in range(3):
            plan = self.sweep_plan()
            if not plan:
                break
            for name, colours in plan:
                self.skill(name)
                self.do_sweep(colours, T_brush)
            self.park_brush(T_brush)   # part of the last sweep segment: the brush leaves the dustpan's path
        if not all(self.in_pan(c) for c in BALLS) or not self.plan_success:
            return {'info': {'failed': 'sweep'}}
        # carry: low lift toward the robot (high far poses are out of reach), then over the basket; mouth tilted up
        # (negative roll) where the planner allows it
        self.skill('carry_to_basket')
        T_pan = self.held(self.pan, 'left')
        grip_local = (np.linalg.inv(mat(self.pan_e.get_pose())) @ np.append(self.robot.get_left_tcp_pose()[:3], 1))[:3]
        g0 = np.array(self.robot.get_left_tcp_pose()[:3])
        # pour position: the balls start to roll out at about 15-25 degrees of roll and leave the mouth moving along its
        # outward direction d (toward the robot). At 25 degrees the mouth edge is 2.5 cm above the rim and 2 cm past the
        # basket centre on the far side; rolling further moves the mouth farther back, so the balls leave 1-4 cm past the
        # centre and land across it (computed: 70 degrees put them over the near rim, 12/13 Nov seeds 12 and 18)
        mouth = np.array([0, -PAN_D / 2, 0.01])
        yaw = self.layout['pan'][2]
        d = np.array([np.sin(yaw), -np.cos(yaw)])
        R_pour = t3d.euler.euler2mat(0, 0, yaw) @ t3d.axangles.axangle2mat([1, 0, 0], np.deg2rad(25))
        off = R_pour @ (mouth - grip_local)
        tgt = self.basket_xy - 0.02 * d
        g_pour = np.array([tgt[0] - off[0], tgt[1] - off[1], TABLE_TOP + BASKET_H + 0.025 - off[2]])
        tilts = [np.deg2rad(d) for d in (-12, -6, 0, 5)]
        g1 = g0 + [0, -0.05, 0.04]
        self.first_feasible('left', T_pan, [self.pan_pose(g1, t, grip_local) for t in tilts], 'carry:lift')
        g2 = (g1 + g_pour) / 2 + [0, 0, 0.03]
        self.first_feasible('left', T_pan, [self.pan_pose(g2, t, grip_local) for t in tilts], 'carry:mid')
        self.first_feasible('left', T_pan, [self.pan_pose(g_pour, t, grip_local) for t in tilts], 'carry:over')
        self.delay(1, save_freq=-1)
        # pour: roll about the handle until the mouth points down, hold, roll back level
        self.skill('pour')
        for deg in (0, 10, 18, 25, 35, 50, 70):   # slow, stepwise: the balls roll out instead of being thrown
            self.first_feasible('left', T_pan, [self.pan_pose(g_pour, np.deg2rad(x), grip_local) for x in (deg, deg - 5)], f'pour:{deg}')
        self.delay(3, save_freq=-1)
        for deg in (45, 20, 0):   # back level in steps too: one fast roll back spun the pan 50 degrees in the grasp
            self.first_feasible('left', T_pan, [self.pan_pose(g_pour, np.deg2rad(x), grip_local) for x in (deg, deg + 5)], f'pour:back{deg}')
        if not self.plan_success:
            return {'info': {'failed': 'pour'}}
        # put both tools down (brush first, then dustpan, near where they started), open, lift, both arms home
        self.skill('put_down_and_home')
        def down(start, dy, dz):
            m = mat(start).copy(); m[1, 3] += dy; m[2, 3] += dz; return m
        for arm, T, start, dys in (('right', T_brush, self.brush_start, (0.0, -0.03)),
                                   ('left', T_pan, self.pan_start, (-0.04, -0.07, 0.0))):
            for dy in dys:
                if self.plan_ok(arm, T, down(start, dy, 0.04)) and self.plan_ok(arm, T, down(start, dy, 0.004)):
                    break
            self.first_feasible(arm, T, [down(start, dy, 0.04)], f'put_down:{arm}:above')
            self.first_feasible(arm, T, [down(start, dy, 0.004)], f'put_down:{arm}')
            self.log_move(f'open:{arm}', self.move(self.open_gripper(ArmTag(arm))))
            self.log_move(f'lift_off:{arm}', self.move(self.move_by_displacement(ArmTag(arm), z=0.05)))
        self.log_move('home', self.move(self.back_to_origin(L), self.back_to_origin(R)))
        self.delay(2, save_freq=-1)
        return {'info': {} if self.plan_success else {'failed': 'put_down'}}

    PAN_CP, BRUSH_CP = [1, 3], [0, 2]   # top-down yaws that close the fingers across each handle (debug_sweep.py axis)


def visible(env, min_pixels=30):
    """Every object visible in the head camera at the current frame (refreshes the render first)."""
    camera = env.cameras.static_camera_list[env.cameras.head_camera_id]
    env.scene.update_render(); camera.take_picture()
    seg = camera.get_picture('Segmentation')[..., 1]
    ents = [env.pan_e, env.brush_e, env.basket_e, *env.balls.values()]
    H, W = seg.shape
    for e in ents:
        yy, xx = np.where(seg == e.per_scene_id)
        if len(xx) < min_pixels or xx.min() < 2 or xx.max() >= W - 2 or yy.min() < 2 or yy.max() >= H - 2:
            return False, e.name
    return True, None


def config(robotwin, eval_mode=False):
    """RoboTwin demo_clean settings with the aloha-agilex embodiment (same as the official RoboTwin tasks)."""
    import copy
    import yaml
    root = Path(robotwin)
    cfg = yaml.safe_load((root / 'task_config/demo_clean.yml').read_text())
    emb = yaml.safe_load((root / 'task_config/_embodiment_config.yml').read_text())
    robot = emb[cfg['embodiment'][0]]['file_path']
    robot_cfg = yaml.safe_load((Path(robot) / 'config.yml').read_text())
    cfg.update(task_name='sweep', task_config='demo_clean',
               left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True,
               left_embodiment_config=copy.deepcopy(robot_cfg), right_embodiment_config=copy.deepcopy(robot_cfg),
               eval_mode=eval_mode, eval_video_log=False, save_data=False, render_freq=0)
    return cfg
