"""Competition-like RoboTwin scenes for the final-round target-cue experiment.

Task 1: put the named fruit / vegetable on the named plate colour (36 objects, 5 plate colours, 2 plates per colour).
Task 2: put the named cosmetic into the named level of a 3-level transparent organizer (10 cosmetics).
Task 3: brush, dustpan, red and white paper balls, round basket. Detection images only: the task does not score
recognition and its sweeping expert is out of scope.
Green backdrop and green table cloth as at the venue; head camera D435 320x240 as in the official RoboTwin setup.
Objects are toy-sized (see assets/assets.json). Assets: RoboTwin, Objaverse (CC, assets/licenses.json), procedural.
"""
import json
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d
from envs._base_task import Base_Task
from envs.utils import ArmTag
from envs.utils.actor_utils import Actor
from envs.utils.create_actor import create_box, create_table

from catalog import (ASSETS, META, T1_OBJECTS, PLATE_COLORS, COSMETICS, T3_OBJECTS, LEVELS, CLASSES, NAME,
                     TEMPLATES, TASK3_INSTRUCTION, CONFUSE)
TABLE_TOP = 0.74
GREEN_WALL, GREEN_TABLE = (0.16, 0.55, 0.38), (0.20, 0.60, 0.40)
VIEW_X_RIGHT = 0.24   # the head camera's view is not symmetric: right image edge ~ x 0.24 on the table, left ~ -0.31
PLATE_R = 0.060   # 12 cm plates: two fit side by side on one arm's half of the table
ORG_W, ORG_D, ORG_WALL = 0.24, 0.050, 0.035      # organizer width, depth per level, wall height above each floor
ORG_FLOORS = {'bottom': 0.004, 'middle': 0.044, 'top': 0.084}
ACRYLIC = (0.86, 0.93, 1.0, 0.22)


def instruction(task, target, dest, template):
    return TEMPLATES[task][template].format(o=NAME[target], d=dest)  # dest: plate colour or organizer level


def _contact_data(ext, center=(0.0, 0.0, 0.0)):
    """Top-down grasps at the object's grasp centre (vertex centroid, mid-height there; not the box centre, which is
    empty space for a banana or a pair of cherries), 12 orientations 30 degrees apart so one is always near the arm's
    preferred direction whatever the object's yaw; functional point at the bottom centre."""
    h = (np.array(ext) / 2).tolist()
    c = np.array(center) / np.maximum(np.array(h), 1e-6)   # Actor multiplies contact translations by 'scale'
    m0 = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    contacts = []
    for k in range(12):
        t = k * np.pi / 6
        rz = np.array([[np.cos(t), -np.sin(t), 0, 0], [np.sin(t), np.cos(t), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        m = rz @ m0
        m[:3, 3] = c
        contacts.append(m.tolist())
    return {
        'center': [0, 0, 0], 'extents': h, 'scale': h,
        'target_pose': [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
        'contact_points_pose': contacts,
        'transform_matrix': np.eye(4).tolist(),
        'functional_matrix': [[[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, -1], [0.0, 0.0, 0.0, 1.0]]],
        'contact_points_description': [], 'contact_points_group': [list(range(12))], 'contact_points_mask': [True],
        'target_point_description': ['The center point on the bottom of the object.'],
    }


def load_asset(scene, key, xy, yaw, name=None, static=False):
    m = META[key]
    path = str(ASSETS / m['file'])
    b = scene.create_actor_builder()
    b.set_physx_body_type('static' if static else 'dynamic')
    b.add_convex_collision_from_file(filename=path)
    b.add_visual_from_file(filename=path)
    e = b.build(name=name or key)
    z = TABLE_TOP + m['extents'][2] / 2 + 0.002
    e.set_pose(sapien.Pose([xy[0], xy[1], z], t3d.euler.euler2quat(0, 0, yaw)))
    if not static:  # round fruit would roll away from its sampled spot (and fail RoboTwin's stability check)
        body = e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        body.angular_damping, body.linear_damping = 6.0, 1.0
    return Actor(e, _contact_data(m['extents'], m.get('grasp_center', (0, 0, 0))), mass=0.05)


def build_plate(scene, color, xy, name):
    path = str(ASSETS / META[f'plate_{color}']['file'])
    b = scene.create_actor_builder()
    b.set_physx_body_type('static')
    b.add_visual_from_file(filename=path)
    b.add_cylinder_collision(pose=sapien.Pose([0, 0, 0.002], [.70710678, 0, .70710678, 0]), radius=0.041, half_length=0.002)
    for i in range(16):  # sloped rim as a ring of thin boxes
        a = i * 2 * np.pi / 16
        b.add_box_collision(pose=sapien.Pose([0.052 * np.cos(a), 0.052 * np.sin(a), 0.008], t3d.euler.euler2quat(0, -0.6, a)),
                            half_size=[0.010, 0.010, 0.0015])
    e = b.build(name=name)
    e.set_pose(sapien.Pose([xy[0], xy[1], TABLE_TOP], [1, 0, 0, 0]))  # plate mesh spans z 0..0.019
    return e


def build_organizer(scene, x0, y0):
    """3-level stepped organizer, open-top compartments; bottom level nearest the robot (smallest y)."""
    mat = sapien.render.RenderMaterial(base_color=list(ACRYLIC), roughness=0.05, metallic=0.0)
    mat.transmission = 0.9
    t = 0.0025
    levels = {}
    for i, lv in enumerate(LEVELS):
        f = ORG_FLOORS[lv]
        ya, yb = y0 + i * ORG_D, y0 + (i + 1) * ORG_D
        b = scene.create_actor_builder()
        b.set_physx_body_type('static')
        boxes = [  # (centre, half size)
            ([x0, (ya + yb) / 2, TABLE_TOP + f], [ORG_W / 2, ORG_D / 2, t]),                                   # floor
            ([x0, ya + t, TABLE_TOP + (f + ORG_WALL) / 2], [ORG_W / 2, t, (f + ORG_WALL) / 2]),                 # front wall
            ([x0 - ORG_W / 2 + t, (ya + yb) / 2, TABLE_TOP + (f + ORG_WALL) / 2], [t, ORG_D / 2, (f + ORG_WALL) / 2]),
            ([x0 + ORG_W / 2 - t, (ya + yb) / 2, TABLE_TOP + (f + ORG_WALL) / 2], [t, ORG_D / 2, (f + ORG_WALL) / 2]),
        ]
        if lv == 'top':
            boxes.append(([x0, yb - t, TABLE_TOP + (f + ORG_WALL) / 2], [ORG_W / 2, t, (f + ORG_WALL) / 2]))  # back wall
        for c, h in boxes:
            b.add_box_collision(pose=sapien.Pose(c), half_size=h)
            b.add_box_visual(pose=sapien.Pose(c), half_size=h, material=mat)
        e = b.build(name=f'organizer_{lv}')
        levels[lv] = {'entity': e, 'x': (x0 - ORG_W / 2, x0 + ORG_W / 2), 'y': (ya, yb), 'floor': TABLE_TOP + f}
    return levels


def footprint(key, yaw):
    """Half sizes of the axis-aligned box around the object's footprint after rotating it by yaw."""
    ex, ey = META[key]['extents'][0] / 2, META[key]['extents'][1] / 2
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return ex * c + ey * s, ex * s + ey * c


def sample_yaw(key, rng):
    """Elongated objects (brushes, banana, chili, cucumber, ...) lie roughly across the table, as people put them down."""
    ex, ey = META[key]['extents'][:2]
    base = 0.0 if ex >= ey else np.pi / 2
    if max(ex, ey) > 2 * min(ex, ey):
        return base + rng.uniform(-0.45, 0.45) + rng.choice([0, np.pi])
    return rng.uniform(-np.pi, np.pi)


def sample_positions(halves, xlim, ylim, rng, tries=4000, gap=0.012, centers=None):
    """Non-overlapping axis-aligned footprints; halves = [(hx, hy)]. The footprint stays inside xlim/ylim; `centers`
    (one (x0, x1, y0, y1) per item) further bounds each centre. Returns centre points."""
    placed, out = [], []
    for j, (hx, hy) in enumerate(halves):
        cx0, cx1, cy0, cy1 = centers[j] if centers else (-9, 9, -9, 9)
        lo_x, hi_x = max(xlim[0] + hx, cx0), min(xlim[1] - hx, cx1)
        lo_y, hi_y = max(ylim[0] + hy, cy0), min(ylim[1] - hy, cy1)
        if lo_x > hi_x or lo_y > hi_y:
            raise ValueError('layout sampling failed')
        for _ in range(tries):
            p = (rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))
            if all(abs(p[0] - q[0]) > hx + q[2] + gap or abs(p[1] - q[1]) > hy + q[3] + gap for q in placed):
                placed.append((*p, hx, hy)); out.append(p); break
        else:
            raise ValueError('layout sampling failed')
    return out


def pick_distractors(target, pool, n, rng):
    others = [k for k in pool if k != target]
    if rng.uniform() < 1 / 3:
        group = [g for g in CONFUSE if target in g]
        near = sorted({k for g in group for k in g if k != target and k in pool})
        if near:
            first = list(rng.choice(near, size=min(len(near), n), replace=False))
            rest = [k for k in others if k not in first]
            return first + list(rng.choice(rest, size=n - len(first), replace=False))
    return list(rng.choice(others, size=n, replace=False))


class RealTask(Base_Task):
    """spec: {'task': 1|2|3, 'target': key, 'dest': colour|level, 'template': int, 'n_objects': int, 'n_plates': int}"""

    def setup_demo(self, spec=None, **kwargs):
        self.spec = dict(spec)
        self.frames = []
        super()._init_task_env_(**kwargs)
        self.step_lim = {1: 600, 2: 700, 3: 900}[self.spec['task']]
        if self.spec['task'] == 3:
            self.set_instruction(TASK3_INSTRUCTION)
        else:
            self.set_instruction(instruction(self.spec['task'], self.spec['target'], self.spec['dest'], self.spec['template']))

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        self.wall_texture, self.table_texture = None, None
        self.wall = create_box(self.scene, sapien.Pose(p=[0, 1, 1.5]), half_size=[3, 0.6, 1.5], color=GREEN_WALL,
                               name='wall', is_static=True)
        self.table = create_table(self.scene, sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
                                  length=1.2, width=0.7, height=table_height, thickness=0.05, color=GREEN_TABLE,
                                  is_static=True)

    def load_actors(self):
        rng = np.random  # seeded by _init_task_env_
        s = self.spec
        self.candidates, self.candidate_keys, self.plates, self.levels, self.extras, self.extra_keys = [], [], [], {}, [], []
        if s['task'] == 1:
            keys, colors, yaws, pos = self._layout_task1(rng)
            for c, p in zip(colors, pos[:len(colors)]):
                self.plates.append({'color': c, 'xy': p, 'entity': build_plate(self.scene, c, p, f'plate_{c}')})
            for k, p, y in zip(keys, pos[len(colors):], yaws):
                self.candidates.append(load_asset(self.scene, k, p, y))
                self.candidate_keys.append(k)
        elif s['task'] == 2:
            # organizer on one arm's side (placing across the middle is where the expert failed, as in task 1), its
            # back level within RoboTwin's usual reach; the target cosmetic lies on that side too
            side = rng.choice([-1.0, 1.0])
            x0, y0 = (rng.uniform(0.10, 0.115) if side > 0 else -rng.uniform(0.12, 0.14)), rng.uniform(-0.07, -0.04)
            self.levels = build_organizer(self.scene, x0, y0)
            n = s.get('n_objects', 4)
            for attempt in range(600):  # pure sampling, no simulator work: retry here instead of rebuilding the env
                k_n = max(3, n - attempt // 200)
                keys = [s['target']] + pick_distractors(s['target'], COSMETICS, k_n - 1, rng)
                rng.shuffle(keys)
                yaws = [sample_yaw(k, rng) for k in keys]
                side_x = (0.03, 0.26) if side > 0 else (-0.26, -0.03)
                centers = [(*side_x, -0.17, y0 - 0.03) if k == s['target'] else (-0.26, 0.26, -0.17, y0 - 0.03) for k in keys]
                try:
                    pos = sample_positions([footprint(k, y) for k, y in zip(keys, yaws)], (-0.30, VIEW_X_RIGHT), (-0.22, y0 - 0.01), rng,
                                           tries=300, centers=centers)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError('no task-2 layout')
            for k, p, y in zip(keys, pos, yaws):
                self.candidates.append(load_asset(self.scene, k, p, y))
                self.candidate_keys.append(k)
        else:
            self.extra_keys = ['dustpan', 'brush', 'basket', 'paper_ball_red', 'paper_ball_white']
            for attempt in range(600):
                yaws = [sample_yaw(k, rng) for k in self.extra_keys]
                try:
                    pos = sample_positions([footprint(k, y) for k, y in zip(self.extra_keys, yaws)], (-0.30, VIEW_X_RIGHT), (-0.22, 0.14), rng, tries=300)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError('no task-3 layout')
            for k, p, y in zip(self.extra_keys, pos, yaws):
                self.extras.append(load_asset(self.scene, k, p, y, static=(k == 'basket')))
        if s['task'] in (1, 2):
            self.target = self.candidates[self.candidate_keys.index(s['target'])]
            self.target_start = np.array(self.target.get_pose().p)

    def _layout_task1(self, rng):
        """Objects and plates inside the head camera's view, footprints apart. One arm must reach both the target and
        the nearest plate of the named colour (same side, or both in the central band), and that region must hold
        another plate too, so "take the plate on my side" never scores the plate. The venue has 10 plates, two per
        colour: plates are drawn from that set, so a colour can appear twice. Retries are pure sampling."""
        s = self.spec
        n, m = s.get('n_objects', 4), s.get('n_plates', int(rng.choice([2, 3, 3])))
        for attempt in range(1500):
            k_n, k_m = max(3, n - attempt // 400), max(2, m - attempt // 600)
            keys = [s['target']] + pick_distractors(s['target'], T1_OBJECTS, k_n - 1, rng)
            rng.shuffle(keys)
            box10 = [c for c in PLATE_COLORS for _ in range(2)]
            box10.remove(s['dest'])
            colors = [s['dest']] + list(rng.choice(box10, size=k_m - 1, replace=False))
            rng.shuffle(colors)
            yaws = [sample_yaw(k, rng) for k in keys]
            halves = [(PLATE_R, PLATE_R)] * len(colors) + [footprint(k, y) for k, y in zip(keys, yaws)]
            # centre ranges as in RoboTwin's single-arm tasks (objects x within +-0.25, y -0.2..0.05), y kept >= -0.16
            # so the near edge stays in the head camera's view
            centers = [(-0.25, 0.25, -0.16, 0.05)] * len(halves)
            try:
                pos = sample_positions(halves, (-0.32, VIEW_X_RIGHT), (-0.24, 0.12), rng, tries=200, centers=centers)
            except ValueError:
                continue
            ti = len(colors) + keys.index(s['target'])
            xt = pos[ti][0]
            # RoboTwin's single-arm tasks put the container on the arm's own side (same sign, |x| >= 0.02): crossing
            # the middle is where the expert's plans failed
            reach = lambda x: abs(x) >= 0.03 and np.sign(x) == np.sign(xt)
            if abs(xt) < 0.03:
                continue
            dest = [i for i, c in enumerate(colors) if c == s['dest']]
            near = min(dest, key=lambda i: np.hypot(pos[i][0] - pos[ti][0], pos[i][1] - pos[ti][1]))
            others = [i for i, c in enumerate(colors) if c != s['dest'] and reach(pos[i][0])]
            if reach(pos[near][0]) and others:
                return keys, colors, yaws, pos
        raise ValueError('no reachable task-1 layout')

    def play_once(self):
        s = self.spec
        obj = self.target
        arm = ArmTag('left' if obj.get_pose().p[0] < 0 else 'right')
        self.move(self.grasp_actor(obj, arm_tag=arm, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10))
        if s['task'] == 1:
            px, py = self.dest_plate()['xy']
            self.move(self.place_actor(obj, target_pose=[px, py, TABLE_TOP + 0.008, 0, 1, 0, 0], arm_tag=arm,
                                       functional_point_id=0, pre_dis=0.09, dis=0.02, constrain='align'))
        else:
            lv = self.levels[s['dest']]
            cx, cy = sum(lv['x']) / 2, sum(lv['y']) / 2
            self.move(self.move_by_displacement(arm_tag=arm, z=0.06))
            self.move(self.place_actor(obj, target_pose=[cx, cy, lv['floor'] + 0.004, 0, 1, 0, 0], arm_tag=arm,
                                       functional_point_id=0, pre_dis=0.10, dis=0.02, constrain='align'))
        self.move(self.move_by_displacement(arm_tag=arm, z=0.08))
        return {'info': {}}

    def check_success(self):
        s = self.spec
        if s['task'] == 3:
            return False
        p = self.target.get_pose().p
        open_ = self.is_left_gripper_open() and self.is_right_gripper_open()
        if s['task'] == 1:  # any plate of the named colour counts
            on = any(q['color'] == s['dest'] and np.hypot(p[0] - q['xy'][0], p[1] - q['xy'][1]) < PLATE_R - 0.008 for q in self.plates)
            return bool(on and open_ and TABLE_TOP < p[2] < TABLE_TOP + 0.12)
        return self.in_level(p, s['dest']) and open_

    def dest_plate(self):
        """The plate of the named colour nearest the target's start (two plates can share a colour)."""
        p0 = self.target_start if hasattr(self, 'target_start') else self.target.get_pose().p
        same = [q for q in self.plates if q['color'] == self.spec['dest']]
        return min(same, key=lambda q: np.hypot(p0[0] - q['xy'][0], p0[1] - q['xy'][1]))

    def in_level(self, p, level):
        """Target centre inside the level's footprint and resting on its floor (a standing lotion bottle's centre is
        7 cm up, so the height window scales with the object's longest side)."""
        lv = self.levels[level]
        h = max(META[self.spec['target']]['extents'])
        return bool(lv['x'][0] < p[0] < lv['x'][1] and lv['y'][0] < p[1] < lv['y'][1]
                    and lv['floor'] - 0.005 < p[2] < lv['floor'] + h / 2 + 0.02)

    def where(self):
        """Where the target ended up: plate colour / organizer level / None (for error analysis)."""
        p = self.target.get_pose().p
        if self.spec['task'] == 1:
            for q in self.plates:
                if np.hypot(p[0] - q['xy'][0], p[1] - q['xy'][1]) < PLATE_R - 0.008 and p[2] < TABLE_TOP + 0.12:
                    return q['color']
            return None
        for lv in LEVELS:
            if self.in_level(p, lv):
                return lv
        return None

    def _take_picture(self):
        if self.save_data:
            obs = self.get_obs()
            self.frames.append({
                'state': np.asarray(obs['joint_action']['vector'], dtype=np.float32),
                'ee': tcp_positions(self),
                **{k: obs['observation'][c]['rgb'].copy() for k, c in
                   [('cam_high', 'head_camera'), ('cam_left_wrist', 'left_camera'), ('cam_right_wrist', 'right_camera')]},
            })
            self.FRAME_IDX += 1


def labeled_entities(env):
    """(class name, entity) for every detectable object in the scene."""
    out = [(k, a.actor) for k, a in zip(env.candidate_keys, env.candidates)]
    out += [(f"plate_{p['color']}", p['entity']) for p in env.plates]
    out += [(f'organizer_{lv}', v['entity']) for lv, v in env.levels.items()]
    out += [(k, a.actor) for k, a in zip(env.extra_keys, env.extras)]
    return out


def boxes(env, min_pixels=12, margin=2, strict=True):
    """Visible-pixel boxes from the head camera's entity segmentation at the current frame, one per labeled entity.
    strict (cue scenes, as at the venue: everything fully visible): raise if any object is missing or cut by the border.
    not strict (detector images): keep the visible part of cut objects and flag them; drop objects with < min_pixels."""
    camera = env.cameras.static_camera_list[env.cameras.head_camera_id]
    env.scene.update_render()   # push the current poses to the renderer: right after setup_demo (no get_obs yet) the
    camera.take_picture()       # segmentation is stale and shows no object (eval seeds, 11/12); no picture exists yet
                                # either (SAPIEN: _Map_base::at)
    seg = camera.get_picture('Segmentation')[..., 1]
    H, W = seg.shape
    res = []
    for cls, e in labeled_entities(env):
        yy, xx = np.where(seg == e.per_scene_id)
        if len(xx) < min_pixels:
            if strict:
                raise ValueError(f'{cls} not visible in frame zero')
            continue
        cut = bool(xx.min() < margin or xx.max() >= W - margin or yy.min() < margin or yy.max() >= H - margin)
        if cut and strict:
            raise ValueError(f'{cls} truncated at the frame boundary')
        res.append({'cls': cls, 'box': [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
                    'pixels': int(len(xx)), 'truncated': cut})
    return res


def tcp_positions(env):
    return np.array([env.robot.get_left_tcp_pose()[:3], env.robot.get_right_tcp_pose()[:3]])


def config(robotwin, eval_mode=False):
    """RoboTwin demo_clean settings with the aloha-agilex embodiment (same as the official RoboTwin tasks)."""
    import copy
    import yaml
    root = Path(robotwin)
    cfg = yaml.safe_load((root / 'task_config/demo_clean.yml').read_text())
    emb = yaml.safe_load((root / 'task_config/_embodiment_config.yml').read_text())
    robot = emb[cfg['embodiment'][0]]['file_path']
    robot_cfg = yaml.safe_load((Path(robot) / 'config.yml').read_text())
    cfg.update(task_name='real_cue', task_config='demo_clean',
               left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True,
               left_embodiment_config=copy.deepcopy(robot_cfg), right_embodiment_config=copy.deepcopy(robot_cfg),
               eval_mode=eval_mode, eval_video_log=False, save_data=False, render_freq=0)
    return cfg
