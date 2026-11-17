"""Normalize competition-like object assets for the RoboTwin/SAPIEN scenes.

Every output asset is a single GLB in metres, Z-up, with its bounding-box centre at the origin, and node
transforms baked into the vertices, so the SAPIEN loader can use identity pose and scale 1.
Sources: RoboTwin assets (copied from the cluster), Objaverse GLBs (Sketchfab CC licences, listed in licenses.json),
and procedural meshes (plates, cosmetics, mangosteen, paper balls) built here.
Usage: python process_assets.py <robotwin_dir> <objaverse_glb_dir> <out_dir>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

RT, OBJ, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)

# key: (task, english name used in instructions, source, target max dimension in metres, options)
# source: ('rt', category, id) | ('obj', file stem) | ('proc', builder name)
ASSETS = {
    # ---- task 1 fruit (25) ----
    'green_apple':   (1, 'green apple', ('rt', '103_fruit', 0), .075, {}),
    'peach':         (1, 'peach', ('rt', '103_fruit', 5), .070, {}),
    'pomegranate':   (1, 'pomegranate', ('obj', 'pomegranate__1'), .080, {}),
    'tangerine':     (1, 'tangerine', ('obj', 'tangerine__0'), .060, {}),
    'watermelon':    (1, 'watermelon', ('obj', 'watermelon__0'), .100, {'max_width': .066, 'rot': ('x', 90)}),
    'banana':        (1, 'banana', ('obj', 'banana__1'), .150, {'rot': ('x', 90)}),
    'red_apple':     (1, 'red apple', ('rt', '035_apple', 1), .075, {}),
    'grapes':        (1, 'grapes', ('obj', 'grapes__0'), .100, {'max_width': .066}),
    'starfruit':     (1, 'star fruit', ('obj', 'starfruit__0'), .090, {'tint': (.92, .82, .22)}),
    'red_delicious': (1, 'red delicious apple', ('obj', 'red_delicious__0'), .080, {}),
    'strawberry':    (1, 'strawberry', ('obj', 'strawberry__1'), .045, {}),
    'persimmon':     (1, 'persimmon', ('proc', 'persimmon'), .065, {}),
    'lemon':         (1, 'lemon', ('rt', '103_fruit', 3), .070, {}),
    'orange':        (1, 'orange', ('rt', '103_fruit', 2), .075, {}),
    'mangosteen':    (1, 'mangosteen', ('proc', 'mangosteen'), .060, {}),
    'pineapple':     (1, 'pineapple', ('rt', '103_fruit', 4), .140, {'max_width': .066}),
    'mango':         (1, 'mango', ('obj', 'mango__0'), .100, {'max_width': .066}),
    'pear':          (1, 'pear', ('rt', '103_fruit', 1), .090, {}),
    'passion_fruit': (1, 'passion fruit', ('obj', 'passion_fruit__0'), .065, {}),
    'dragon_fruit':  (1, 'dragon fruit', ('obj', 'dragon_fruit__0'), .100, {'max_width': .066}),
    'kiwi':          (1, 'kiwi', ('obj', 'kiwi__0'), .065, {}),
    'avocado':       (1, 'avocado', ('obj', 'avocado__0'), .090, {'max_width': .066}),
    'cherry':        (1, 'cherry', ('obj', 'cherry__1'), .045, {}),
    'wax_apple':     (1, 'wax apple', ('obj', 'wax_apple__0'), .060, {'tint': (.86, .22, .30)}),
    'papaya':        (1, 'papaya', ('obj', 'papaya__0'), .120, {'tint': (.62, .72, .12), 'max_width': .066}),
    # ---- task 1 vegetables and mushroom (11) ----
    'tomato':        (1, 'tomato', ('rt', '069_vagetable', 3), .065, {}),
    'mushroom':      (1, 'mushroom', ('obj', 'mushroom__0'), .080, {}),
    'cucumber':      (1, 'cucumber', ('rt', '069_vagetable', 2), .150, {}),
    'pumpkin':       (1, 'pumpkin', ('obj', 'pumpkin__0'), .075, {}),
    'bell_pepper':   (1, 'bell pepper', ('obj', 'bell_pepper__0'), .075, {}),
    'carrot':        (1, 'carrot', ('rt', '069_vagetable', 6), .140, {}),
    'red_chili':     (1, 'red chili pepper', ('obj', 'red_chili__0'), .110, {'rot': ('y', 90)}),
    'green_chili':   (1, 'green chili pepper', ('obj', 'red_chili__1'), .110, {'tint': (.18, .62, .10), 'rot': ('y', 90)}),
    'eggplant':      (1, 'eggplant', ('rt', '069_vagetable', 4), .140, {}),
    'onion':         (1, 'onion', ('rt', '069_vagetable', 1), .070, {}),
    'potato':        (1, 'potato', ('rt', '069_vagetable', 0), .080, {}),
    # ---- task 2 cosmetics (10, procedural from the official descriptions) ----
    'lotion_bottle':     (2, 'beige lotion pump bottle', ('proc', 'lotion_bottle'), None, {}),
    'lip_gloss':         (2, 'beige lip gloss', ('proc', 'lip_gloss'), None, {}),
    'lipstick':          (2, 'black lipstick', ('proc', 'lipstick'), None, {}),
    'eyeshadow_palette': (2, 'multicolor eyeshadow palette', ('proc', 'eyeshadow_palette'), None, {}),
    'brush_large':       (2, 'large black and gold makeup brush', ('proc', 'brush_large'), None, {}),
    'brush_small':       (2, 'small black and gold makeup brush', ('proc', 'brush_small'), None, {}),
    'highlighter':       (2, 'light round compact', ('proc', 'highlighter'), None, {}),
    'contour':           (2, 'dark round compact', ('proc', 'contour'), None, {}),
    'powder_puff':       (2, 'pink powder puff', ('proc', 'powder_puff'), None, {}),
    'glitter_compact':   (2, 'black round glitter compact', ('proc', 'glitter_compact'), None, {}),
    # ---- task 3 (detection only) ----
    'dustpan':       (3, 'dustpan', ('rt', '082_smallshovel', 3), .200, {}),
    'brush':         (3, 'brush', ('rt', '083_brush', 2), .200, {}),
    'basket':        (3, 'basket', ('rt', '110_basket', 0), .220, {}),
    'paper_ball_red':   (3, 'red paper ball', ('proc', 'paper_ball_red'), None, {}),
    'paper_ball_white': (3, 'white paper ball', ('proc', 'paper_ball_white'), None, {}),
}
GRASP_W = 0.060  # max graspable width for task 1/2 objects (aloha-agilex parallel gripper, conservative)
PLATE_COLORS = {'white': (.94, .94, .93), 'yellow': (.98, .84, .18), 'pink': (.97, .62, .74),
                'purple': (.58, .44, .80), 'blue': (.28, .52, .90)}


def fix_specgloss(src: Path, dst: Path):
    """Rewrite KHR_materials_pbrSpecularGlossiness materials as metallic-roughness (diffuse -> base colour)."""
    b = src.read_bytes()
    n = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + n])
    rest = b[20 + n:]
    changed = False
    for m in j.get('materials', []):
        sg = (m.get('extensions') or {}).pop('KHR_materials_pbrSpecularGlossiness', None)
        if sg is None:
            continue
        changed = True
        pbr = m.setdefault('pbrMetallicRoughness', {})
        if 'diffuseTexture' in sg:
            pbr['baseColorTexture'] = sg['diffuseTexture']
        if 'diffuseFactor' in sg:
            pbr['baseColorFactor'] = sg['diffuseFactor']
        pbr['metallicFactor'] = 0.0
        pbr['roughnessFactor'] = 0.7
        if not m['extensions']:
            m.pop('extensions')
    for k in ('extensionsUsed', 'extensionsRequired'):
        if k in j:
            j[k] = [e for e in j[k] if e != 'KHR_materials_pbrSpecularGlossiness']
            if not j[k]:
                j.pop(k)
    if not changed:
        dst.write_bytes(b)
        return
    js = json.dumps(j, separators=(',', ':')).encode()
    js += b' ' * ((4 - len(js) % 4) % 4)
    body = struct.pack('<I', len(js)) + b'JSON' + js + rest
    dst.write_bytes(b[:8] + struct.pack('<I', 12 + len(body)) + body)


def baked_meshes(path: Path, largest_part_only=False):
    scene = trimesh.load(str(path), force='scene')
    meshes = []
    for node in scene.graph.nodes_geometry:
        T, g = scene.graph[node]
        m = scene.geometry[g].copy()
        m.apply_transform(T)
        meshes.append(m)
    if largest_part_only and len(meshes) > 1:
        vol = [np.prod(m.extents) for m in meshes]
        keep = int(np.argmax(vol))
        c = meshes[keep].bounds.mean(0)
        r = np.linalg.norm(meshes[keep].extents) * 0.6
        # keep the largest part plus small attached parts (calyx, stem) whose centre is close to it
        meshes = [m for i, m in enumerate(meshes) if i == keep or
                  (np.prod(m.extents) < 0.2 * vol[keep] and np.linalg.norm(m.bounds.mean(0) - c) < r)]
    return meshes


def tint(meshes, rgb):
    for m in meshes:
        v = m.visual
        if isinstance(v, trimesh.visual.TextureVisuals) and v.material is not None:
            mat = v.material
            if hasattr(mat, 'baseColorFactor'):
                mat.baseColorFactor = np.array([*(np.array(rgb) * 255), 255], dtype=np.uint8)
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                # keep luminance detail of the texture, recolour it
                g = np.asarray(mat.baseColorTexture.convert('L'), dtype=np.float32) / 255.
                g = 0.65 + 0.35 * (g - g.min()) / max(np.ptp(g), 1e-3)
                img = np.clip(g[..., None] * np.array(rgb)[None, None] * 255, 0, 255).astype(np.uint8)
                mat.baseColorTexture = Image.fromarray(img)
                mat.baseColorFactor = np.array([255, 255, 255, 255], dtype=np.uint8)
        else:
            colored(m, rgb)


def normalize(meshes, size, max_width=None, yup=True, rot=None):
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]) if yup else np.eye(4)
    if rot is not None:  # extra rotation after Z-up, e.g. lay a standing object down: ('x', 90)
        R = trimesh.transformations.rotation_matrix(np.radians(rot[1]), {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[rot[0]]) @ R
    for m in meshes:
        m.apply_transform(R)
    allv = np.vstack([m.vertices for m in meshes])
    lo, hi = allv.min(0), allv.max(0)
    ext = hi - lo
    s = size / ext.max()
    # width cap only for compact objects: a long or curved object (banana, chili) is grasped across its thickness
    lying_thin = ext[2] < 0.5 * min(ext[0], ext[1])   # flat-lying thin object: grasped across its thickness
    if max_width is not None and max(ext[0], ext[1]) <= 1.6 * min(ext[0], ext[1]) and not lying_thin:
        s = min(s, max_width / min(ext[0], ext[1]))
    c = (lo + hi) / 2
    for m in meshes:
        m.apply_translation(-c)
        m.apply_scale(s)
    return ext * s


def colored(mesh, rgb, alpha=1.0, rough=0.6):
    mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[*rgb, alpha], metallicFactor=0.0, roughnessFactor=rough,
                                              alphaMode='BLEND' if alpha < 1 else 'OPAQUE', doubleSided=alpha < 1)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    return mesh


def cyl(r, h, rgb, z0=0.0, sections=48, alpha=1.0):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    m.apply_translation([0, 0, z0 + h / 2])
    return colored(m, rgb, alpha)


def lying(parts):
    """Rotate an upright (Z-axis) object to lie along X."""
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    for p in parts:
        p.apply_transform(R)
    return parts


def crumpled_sphere(r, rgb, seed):
    rng = np.random.default_rng(seed)
    m = trimesh.creation.icosphere(subdivisions=3, radius=r)
    n = m.vertex_normals
    m.vertices += n * rng.normal(0, r * 0.12, (len(m.vertices), 1))
    m = m.subdivide()
    return colored(m, rgb)


def procedural(name):
    gold, black = (.80, .64, .30), (.06, .06, .06)
    if name == 'mangosteen':
        body = trimesh.creation.icosphere(subdivisions=4, radius=.03)
        body.vertices[:, 2] *= .85
        colored(body, (.28, .08, .16))
        calyx = [colored(trimesh.creation.box([.016, .008, .004]), (.25, .42, .18)) for _ in range(4)]
        for i, c in enumerate(calyx):
            c.apply_transform(trimesh.transformations.rotation_matrix(i * np.pi / 2, [0, 0, 1]))
            c.apply_translation([.007 * np.cos(i * np.pi / 2), .007 * np.sin(i * np.pi / 2), .025])
        stem = cyl(.0025, .012, (.35, .45, .2), z0=.025)
        return [body, *calyx, stem]
    if name == 'persimmon':
        body = trimesh.creation.icosphere(subdivisions=4, radius=.033)
        body.vertices[:, 2] *= .72
        colored(body, (.93, .45, .08), rough=.4)
        leaves = [colored(trimesh.creation.box([.020, .011, .003]), (.30, .40, .15)) for _ in range(4)]
        for i, c in enumerate(leaves):
            c.apply_transform(trimesh.transformations.rotation_matrix(i * np.pi / 2 + np.pi / 4, [0, 0, 1]))
            c.apply_translation([.009 * np.cos(i * np.pi / 2 + np.pi / 4), .009 * np.sin(i * np.pi / 2 + np.pi / 4), .023])
        return [body, *leaves, cyl(.003, .008, (.35, .30, .15), z0=.022)]
    if name == 'lotion_bottle':
        beige = (.92, .85, .72)
        return [cyl(.022, .095, beige), cyl(.012, .012, beige, z0=.095), cyl(.007, .018, (.95, .93, .9), z0=.107),
                colored(trimesh.creation.box([.032, .011, .011]).apply_translation([.008, 0, .130]), (.95, .93, .9))]
    if name == 'lip_gloss':
        beige = (.90, .78, .66)
        return lying([cyl(.0085, .055, beige), cyl(.009, .030, (.85, .72, .55), z0=.055)])
    if name == 'lipstick':
        return lying([cyl(.0105, .048, black), cyl(.007, .018, (.70, .08, .14), z0=.048),
                      cyl(.0112, .036, (.85, .90, .95), z0=.046, alpha=.35)])
    if name == 'eyeshadow_palette':
        base = colored(trimesh.creation.box([.095, .058, .011]).apply_translation([0, 0, .0055]), (.10, .09, .09))
        pans = []
        cols = [(.86, .70, .60), (.72, .45, .35), (.55, .30, .25), (.35, .20, .18), (.90, .80, .55), (.75, .55, .30),
                (.60, .40, .55), (.40, .25, .45), (.85, .55, .55), (.65, .30, .30), (.50, .55, .40), (.30, .30, .35)]
        for i, c in enumerate(cols):
            x, y = -.035 + (i % 6) * .014, -.0115 + (i // 6) * .023
            pans.append(colored(trimesh.creation.box([.012, .020, .0012]).apply_translation([x, y, .0114]), c))
        return [base, *pans]
    if name in ('brush_large', 'brush_small'):
        k = 1.0 if name == 'brush_large' else .68
        handle = trimesh.creation.cone(radius=.0062 * k, height=.13 * k, sections=32)
        handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        handle.apply_translation([0, 0, .13 * k])
        colored(handle, black)
        ferrule = cyl(.0068 * k, .030 * k, gold, z0=.128 * k)
        bristle = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        bristle.vertices *= [.012 * k, .012 * k, .026 * k]
        bristle.apply_translation([0, 0, (.158 + .018) * k])
        colored(bristle, (.12, .09, .08))
        return lying([handle, ferrule, bristle])
    if name in ('highlighter', 'contour', 'glitter_compact'):
        body = {'highlighter': (.93, .89, .82), 'contour': (.28, .19, .15), 'glitter_compact': (.05, .05, .05)}[name]
        top = {'highlighter': (.96, .93, .88), 'contour': (.34, .24, .19), 'glitter_compact': (.18, .16, .14)}[name]
        r = .028 if name != 'glitter_compact' else .027
        parts = [cyl(r, .011, body), cyl(r * .96, .0012, top, z0=.011)]
        if name == 'glitter_compact':
            rng = np.random.default_rng(3)
            for _ in range(40):
                a, d = rng.uniform(0, 2 * np.pi), r * .9 * np.sqrt(rng.uniform())
                parts.append(colored(trimesh.creation.box([.0016, .0016, .0004]).apply_translation(
                    [d * np.cos(a), d * np.sin(a), .0124]), (.85, .80, .65)))
        return parts
    if name == 'powder_puff':
        puff = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        puff.vertices *= [.029, .029, .007]
        puff.apply_translation([0, 0, .007])
        strap = trimesh.creation.box([.036, .010, .003]).apply_translation([0, 0, .0145])
        return [colored(puff, (.97, .70, .78)), colored(strap, (.93, .55, .66))]
    if name == 'paper_ball_red':
        return [crumpled_sphere(.024, (.80, .12, .12), 1)]
    if name == 'paper_ball_white':
        return [crumpled_sphere(.024, (.93, .93, .91), 2)]
    raise KeyError(name)


def plate(rgb):
    # revolve a profile: flat bottom (r 0.055), sloped wall to rim at r 0.08, height 0.018
    prof = np.array([[0, 0], [.055, 0], [.075, .014], [.080, .018], [.078, .019], [.073, .016], [.053, .004], [0, .004]])
    prof[:, 0] *= 0.75   # 12 cm diameter (scene_real.PLATE_R = 0.06)
    m = trimesh.creation.revolve(prof, sections=96)
    return [colored(m, rgb)]


def export(key, parts, meta, info):
    sc = trimesh.Scene(parts)
    out = OUT / f'{key}.glb'
    sc.export(str(out))
    allv = np.vstack([p.vertices for p in parts])
    ext = allv.max(0) - allv.min(0)
    # grasp centre: vertex centroid in x/y, mid-height of the vertices near it (bbox centre can be empty space)
    # the body point nearest the box centre in x/y (vertex centroids drift to dense parts, e.g. brush bristles)
    d = np.linalg.norm(allv[:, :2] - (allv[:, :2].min(0) + allv[:, :2].max(0)) / 2, axis=1)
    cxy = allv[d <= d.min() + 0.004, :2].mean(0)
    near = allv[np.linalg.norm(allv[:, :2] - cxy, axis=1) < 0.008]
    if len(near) < 20:
        near = allv[np.argsort(np.linalg.norm(allv[:, :2] - cxy, axis=1))[:50]]
    gc = [float(cxy[0]), float(cxy[1]), float((near[:, 2].min() + near[:, 2].max()) / 2)]
    meta[key] = {**info, 'file': out.name, 'extents': ext.round(4).tolist(),
                 'graspable_width': float(round(min(ext[0], ext[1]), 4)), 'grasp_center': [round(v, 4) for v in gc]}


def main():
    meta, fixed = {}, OUT / '_fixed'
    fixed.mkdir(exist_ok=True)
    for key, (task, name, src, size, opt) in ASSETS.items():
        if src[0] == 'proc':
            parts = procedural(src[1])
            if size:
                normalize(parts, size, yup=False)
            else:
                allv = np.vstack([p.vertices for p in parts])
                for p in parts:
                    p.apply_translation(-(allv.min(0) + allv.max(0)) / 2)
        else:
            path = (RT / src[1] / 'visual' / f'base{src[2]}.glb') if src[0] == 'rt' else (OBJ / f'{src[1]}.glb')
            fx = fixed / f'{key}.glb'
            fix_specgloss(path, fx)
            parts = baked_meshes(fx, opt.get('largest_part_only', False))
            if 'tint' in opt:
                tint(parts, opt['tint'])
            # one-hand grasp: the narrower horizontal side must fit the gripper (toy fruit is smaller than real fruit)
            mw = opt.get('max_width', GRASP_W) if task in (1, 2) else opt.get('max_width')
            normalize(parts, size, min(mw, GRASP_W) if mw and task in (1, 2) else mw, rot=opt.get('rot'))
        export(key, parts, meta, {'task': task, 'name': name, 'source': list(map(str, src))})
        print(f"{key:18s} task{task} {meta[key]['extents']} grasp_w={meta[key]['graspable_width']}", flush=True)
    for color, rgb in PLATE_COLORS.items():
        export(f'plate_{color}', plate(rgb), meta, {'task': 1, 'name': f'{color} plate', 'source': ['proc', 'plate']})
    (OUT / 'assets.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
