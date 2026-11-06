"""Blender 内运行：把「相机视图」投影到 3D 表面，再 bake 成 UV 贴图。

    blender.exe -b -P tools/blender_bake_views.py -- <config.json>

与旧的 texel 反向映射路线的本质区别：
  * 旧：UV 图集每个 texel 反查 3D 点 → 按屏幕坐标采样视图
        → 碎片 UV 岛上相邻 texel 会跳到视图里完全无关的位置（碎玻璃）。
  * 新：Blender 把每个三角面**正向**光栅化到 UV，投影坐标在三角形内部
        连续插值（相机矩阵算出来的 ProjUV），bake 天然不会碎。

另外按 StableGen Wiki 的做法重新 unwrap（Cube Project，适合 TRELLIS.2 稠密网格），
不复用原生链 xatlas 的 106 个碎片 UV 岛。

产出（写到 config 的 work_dir）：
  baked_uv.glb          归一化 + 重新 unwrap 的网格（BakeUV 为 active）
  views/<name>_color.npy   该视图烘出来的颜色 (S,S,3) 线性
  views/<name>_weight.npy  该视图的权重 (S,S) = alpha * cos^k
  meta.json              归一化参数 / 相机 / 面数
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
CFG = json.load(open(argv[0], encoding="utf-8"))

SIZE = int(CFG.get("size", 1024))
ORTHO = float(CFG.get("ortho", 2.8))       # 正交相机视野边长（正方形视图）
EXP = float(CFG.get("exponent", 3.0))
MARGIN = int(CFG.get("margin", 4))
DECIMATE = int(CFG.get("decimate", 0))
ENGINE = CFG.get("engine", "CYCLES")
OCCLUDE = bool(CFG.get("occlude", True))

# ── 以下四个参数全部取自 StableGen v0.5.0 场景属性默认值 ──────────────────
# visibility_vignette_width = 0.15（properties.py:828，clamp 到 0.49）
FEATHER = float(CFG.get("feather", 0.1))
# visibility_vignette_softness = 1.0（:833），羽化曲线的 gamma
FEATHER_GAMMA = float(CFG.get("feather_gamma", 1.0))
# discard_factor = 70.0°（:705）——超过这个入射角的 texel 权重直接归零，
# 否则掠射面会被拉成条纹（StableGen shader 里是 angle_check = deg(acos(facing)) < discard）
DISCARD_DEG = float(CFG.get("discard", 180.0))
# scene_ray_epsilon = bbox 对角线 × 1e-5（projection.py:284-305），随场景尺度缩放
RAY_EPS = float(CFG.get("ray_eps", 0.0))    # 0 = 自动按模型对角线算
# 遮挡判定的三条路：
#   shader（默认）：ShaderNodeRaycast，与 StableGen create_native_raycast_visibility 一致。
#     ⚠ Cycles 会排除「正在着色的那个图元」，所以射线必然穿到表面**之后**才命中：
#     Is Hit 表示「表面背后还有几何」。Length = dist + eps*10 正是为此。
#   bvh：BVHTree 逐顶点 ray_cast，写 POINT attribute（Vertex 级精度，非 texel 级）。
#   depth：渲染 Z pass + 深度比较，4.5 老版本兜底。
OCCLUSION = str(CFG.get("occlusion", "depth")).lower()
BVH = None
DEPTH_NEAR = 2.0
DEPTH_FAR = 5.0
DEPTH_EPS = 0.012
WORK = CFG["work_dir"]
VDIR = os.path.join(WORK, "views")
os.makedirs(VDIR, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=CFG["mesh"])

scene = bpy.context.scene
meshes = [o for o in scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("GLB 里没有 mesh")


# --- 归一化到 [-1,1]（与 tools/blender_preview.py 完全一致，保证相机可复用） ---
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
mn = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
mx = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
center = (mn + mx) / 2
scale = 2.0 / max(mx - mn)
for o in meshes:
    o.matrix_world = Matrix.Scale(scale, 4) @ Matrix.Translation(-center) @ o.matrix_world

bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
mesh = obj.data

if DECIMATE and len(mesh.polygons) > DECIMATE:
    mod = obj.modifiers.new("decimate", "DECIMATE")
    mod.ratio = DECIMATE / len(mesh.polygons)
    bpy.ops.object.modifier_apply(modifier=mod.name)
print(f"BAKE mesh: {len(mesh.vertices)} verts / {len(mesh.polygons)} faces", flush=True)

# StableGen 的 ray 容差是 bbox 对角线 × 1e-5，不是固定值。归一化后对角线约 3.46，
# 所以 eps ≈ 3.5e-5；之前写死的 0.02 大了 500 倍，会把薄处直接判成遮挡。
if RAY_EPS <= 0.0:
    diag = math.sqrt(sum((mx[i] - mn[i]) ** 2 for i in range(3))) * scale
    RAY_EPS = max(diag * 1e-5, 1e-7)
print(f"BAKE ray_eps={RAY_EPS:.3e}", flush=True)

if OCCLUSION == "bvh":
    from mathutils.bvhtree import BVHTree

    BVH = BVHTree.FromPolygons([v.co.copy() for v in mesh.vertices],
                               [list(p.vertices) for p in mesh.polygons],
                               all_triangles=False)
    print(f"BAKE occlusion=bvh verts={len(mesh.vertices)}", flush=True)

# --- UV 策略 ---
# bake 是「逐三角面正向光栅化」，碎片 UV 岛完全无害；真正致命的是 **UV 重叠**
# （Cube Project 会把朝向相近的多个面投到同一区域，互相覆盖 → 表面斑驳）。
# 因此默认保留输入网格已有的不重叠 UV（TRELLIS.2 原生链用 xatlas 展开过）。
# 岛间距：太小吃 bake margin 的跨岛渗色，太大把图集占用吃光（实测 0.0029 →
# 占用从 44.5% 掉到 20%）。取 ~2px 起步；真正防“跨部件串色”的是
# `bake._fill_holes` 的岛内约束，不是这里。
UV_GAP = max(2.0 / max(SIZE, 1), 0.001)

UNWRAP = CFG.get("unwrap", "auto")
have_uv = len(mesh.uv_layers) > 0
if UNWRAP == "auto":
    # 白模（shape_only 导出）根本没有 UV 层，实测 uv_layers == []，
    # 所以必须自己展开。默认走 StableGen 的 'basic'（ANGLE_BASED, margin 0.001）
    # + pack_islands，而不是 smart_project：后者在 10 万面上会切出上千个
    # 小岛、每个都吃 island_margin，实测图集占用只有 21%。
    UNWRAP = "keep" if have_uv else "smart"
if UNWRAP == "keep":
    if not have_uv:
        raise RuntimeError("unwrap=keep 但网格没有 UV；改用 --unwrap basic")
    bakeuv = mesh.uv_layers.active
    bakeuv.name = "BakeUV"
    print("BAKE unwrap: 保留原始 UV -> BakeUV", flush=True)
else:
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if UNWRAP == "cube":
        bpy.ops.uv.cube_project(cube_size=2.0, scale_to_bounds=True,
                                correct_aspect=True)
    elif UNWRAP == "basic":
        # StableGen unwrap(method='basic')：uv.unwrap(method='ANGLE_BASED', margin=0.001)
        bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=UV_GAP)
        _packed = False
        for kwargs in ({"margin": UV_GAP, "rotate": True, "scale": True},
                       {"margin": UV_GAP}, {}):
            try:
                bpy.ops.uv.pack_islands(**kwargs)
                _packed = True
                break
            except (TypeError, RuntimeError):
                continue
        print(f"BAKE pack_islands: {_packed}", flush=True)
    else:
        bpy.ops.uv.smart_project(island_margin=UV_GAP, rotate_method="AXIS_ALIGNED")
        try:
            bpy.ops.uv.pack_islands(margin=UV_GAP, rotate=True, scale=True)
        except (TypeError, RuntimeError):
            pass
    bpy.ops.object.mode_set(mode="OBJECT")
    bakeuv = mesh.uv_layers.active
    bakeuv.name = "BakeUV"
    # 统计图集占用：UV 三角形面积和 / 1.0，直接反映有效 texel 密度
    try:
        me = mesh
        me.calc_loop_triangles()
        _uv = np.empty(len(me.loops) * 2, np.float32)
        bakeuv.data.foreach_get("uv", _uv)
        _uv = _uv.reshape(-1, 2)
        _tri = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("loops", _tri)
        _tri = _tri.reshape(-1, 3)
        _a, _b, _c = _uv[_tri[:, 0]], _uv[_tri[:, 1]], _uv[_tri[:, 2]]
        _area = 0.5 * np.abs((_b[:, 0] - _a[:, 0]) * (_c[:, 1] - _a[:, 1])
                             - (_c[:, 0] - _a[:, 0]) * (_b[:, 1] - _a[:, 1]))
        print(f"BAKE unwrap: {UNWRAP} -> BakeUV，UV 面积占用 {_area.sum():.1%} "
              f"（islands={len(me.polygons)} faces）", flush=True)
    except Exception as _e:  # 统计失败不影响 bake
        print(f"BAKE unwrap: {UNWRAP} -> BakeUV（占用统计失败：{_e}）", flush=True)
mesh.uv_layers.active = bakeuv
bakeuv.active_render = True

# --- BakeUV compact（StableGen rendering.py:2298）---------------------------
# 把「被至少一台相机投影覆盖」的面整体缩放填满图集，完全没被覆盖的面塌缩到
# (0,0)。覆盖 UV 与 baked image 一起消费，所以逐轴 reframe 是无损的；
# guard 保证塌缩面的 bake margin 不会碰到有效岛。
# 投影用的是独立的 ProjUV（相机空间），改 BakeUV 不影响投影内容。
ALL_VIEWS = CFG.get("all_views") or []
if CFG.get("compact") and ALL_VIEWS and UNWRAP != "keep":
    _V = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", _V)
    _V = _V.reshape(-1, 3).astype(np.float64)
    _lvi = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", _lvi)
    _starts = np.empty(len(mesh.polygons), np.int32)
    _totals = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_start", _starts)
    mesh.polygons.foreach_get("loop_total", _totals)

    _covered = np.zeros(len(mesh.polygons), bool)
    for _v in ALL_VIEWS:
        _az = math.radians(float(_v.get("azimuth", 0)))
        _el = math.radians(float(_v.get("elevation", 0)))
        _d = Vector((math.sin(_az) * math.cos(_el),
                     -math.cos(_az) * math.cos(_el),
                     math.sin(_el)))
        _m = (Vector((0, 0, 0)) - _d * 3.5).to_track_quat("-Z", "Y").to_matrix().to_4x4()
        _m.translation = _d * 3.5
        _inv = np.array(_m.inverted(), dtype=np.float64)
        _P = _V @ _inv[:3, :3].T + _inv[:3, 3]
        _u = (0.5 + _P[:, 0] / ORTHO)[_lvi]
        _w = (0.5 + _P[:, 1] / ORTHO)[_lvi]
        _minu = np.minimum.reduceat(_u, _starts)
        _maxu = np.maximum.reduceat(_u, _starts)
        _minv = np.minimum.reduceat(_w, _starts)
        _maxv = np.maximum.reduceat(_w, _starts)
        # 源码 :2165 的保守判据：面 UV bbox 与 [0,1] 相交即算覆盖
        _covered |= ((_maxu >= 0.0) & (_minu <= 1.0)
                     & (_maxv >= 0.0) & (_minv <= 1.0))

    _loopmask = np.zeros(len(mesh.loops), bool)
    for _i in np.flatnonzero(_covered):
        _loopmask[_starts[_i]:_starts[_i] + _totals[_i]] = True

    _uv = np.empty(len(mesh.loops) * 2, np.float32)
    bakeuv.data.foreach_get("uv", _uv)
    _uv = _uv.reshape(-1, 2)
    _guard = min(0.2, max(2.0 / max(SIZE, 1),
                          (float(MARGIN) + 2.0) / max(SIZE, 1)))
    _target = max(0.01, 1.0 - 2.0 * _guard)
    if _loopmask.any():
        _cl = _uv[_loopmask]
        _min_u, _max_u = float(_cl[:, 0].min()), float(_cl[:, 0].max())
        _min_v, _max_v = float(_cl[:, 1].min()), float(_cl[:, 1].max())
        _su = _target / max(_max_u - _min_u, 1e-12)
        _sv = _target / max(_max_v - _min_v, 1e-12)
        _uv[_loopmask, 0] = _guard + (_cl[:, 0] - _min_u) * _su
        _uv[_loopmask, 1] = _guard + (_cl[:, 1] - _min_v) * _sv
    _uv[~_loopmask] = 0.0
    bakeuv.data.foreach_set("uv", _uv.ravel())
    print(f"BAKE compact: covered {_covered.mean():.1%} faces, "
          f"guard={_guard:.4f}, scale=({_su if _loopmask.any() else 1:.3f},"
          f"{_sv if _loopmask.any() else 1:.3f})", flush=True)

# 中间产物：带干净 UV 的白模，供 blender_apply_atlas.py 贴最终图集。
# keep 模式 UV 没变，宿主要走「二进制换贴图」（网格一字节不改），这份中间网格
# 既没用又占 50MB（Blender 按 loop 拆顶点），直接不导。
if UNWRAP == "keep":
    print("BAKE skip baked_uv.glb（keep 模式改由宿主二进制换贴图）", flush=True)
else:
    uv_glb = os.path.join(WORK, "baked_uv.glb")
    bpy.ops.export_scene.gltf(filepath=uv_glb, export_format="GLB",
                              export_texcoords=True, export_normals=True,
                              export_materials="NONE")
    print("BAKE saved", uv_glb, flush=True)

# --- 顶点 / loop 索引一次性读出，后面全部 numpy 向量化 ---
vco = np.empty(len(mesh.vertices) * 3, np.float32)
mesh.vertices.foreach_get("co", vco)
V = vco.reshape(-1, 3).astype(np.float64)
lvi = np.empty(len(mesh.loops), np.int32)
mesh.loops.foreach_get("vertex_index", lvi)

# --- bake 环境 ---
try:
    scene.render.engine = ENGINE
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
if scene.render.engine == "CYCLES":
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
scene.render.bake.use_clear = True
scene.render.bake.margin = MARGIN
scene.render.bake.use_selected_to_active = False
scene.render.resolution_x = scene.render.resolution_y = SIZE

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
mesh.materials.clear()


def new_target(name):
    img = bpy.data.images.new(name, width=SIZE, height=SIZE,
                              alpha=True, float_buffer=True)
    img.generated_color = (0, 0, 0, 1)
    return img


def build_mat(key, mode, target, uvmap_name, cam_dir=None,
              cam_loc=None, zdepth=None, cam_right=None, cam_up=None,
              feather=0.0, vis_attr=None):
    """mode=color: Emission(视图颜色)；mode=weight: Emission(alpha * cos^k * 可见性)。

    另有 debug_vis / debug_facing / debug_edge 三种模式，只为定位「哪一项把权重乘成 0」。
    """
    mat = bpy.data.materials.new(f"{mode}_{key}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])

    if mode.startswith("debug_"):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        if mode == "debug_alpha":
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(CFG["views"][key]["image"])
            tex.extension = "CLIP"
            uvn = nt.nodes.new("ShaderNodeUVMap")
            uvn.uv_map = uvmap_name
            nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
            nt.links.new(tex.outputs["Alpha"], em.inputs["Color"])
            tgt = nt.nodes.new("ShaderNodeTexImage")
            tgt.image = target
            tgt.select = True
            nt.nodes.active = tgt
            return mat
        cp = nt.nodes.new("ShaderNodeCombineXYZ")
        for i in range(3):
            cp.inputs[i].default_value = float(cam_loc[i])
        sub = nt.nodes.new("ShaderNodeVectorMath")
        sub.operation = "SUBTRACT"
        nt.links.new(geo.outputs["Position"], sub.inputs[0])
        nt.links.new(cp.outputs["Vector"], sub.inputs[1])
        if mode.startswith("debug_dcheck_"):
            # 量 (dist - HitDistance) 的量级：eps 乘不同倍数，看判据在哪一档恢复
            mult = float(mode.rsplit("_", 1)[1])
            dist = nt.nodes.new("ShaderNodeVectorMath")
            dist.operation = "LENGTH"
            nt.links.new(sub.outputs["Vector"], dist.inputs[0])
            ndir = nt.nodes.new("ShaderNodeVectorMath")
            ndir.operation = "NORMALIZE"
            nt.links.new(sub.outputs["Vector"], ndir.inputs[0])
            ray = nt.nodes.new("ShaderNodeRaycast")
            nt.links.new(cp.outputs["Vector"], ray.inputs["Position"])
            nt.links.new(ndir.outputs["Vector"], ray.inputs["Direction"])
            ln = nt.nodes.new("ShaderNodeMath")
            ln.operation = "MULTIPLY_ADD"
            ln.inputs[1].default_value = 1.0
            ln.inputs[2].default_value = RAY_EPS * 10.0
            nt.links.new(dist.outputs["Value"], ln.inputs[0])
            nt.links.new(ln.outputs[0], ray.inputs["Length"])
            hp = nt.nodes.new("ShaderNodeMath")
            hp.operation = "ADD"
            hp.inputs[1].default_value = RAY_EPS * mult
            nt.links.new(ray.outputs["Hit Distance"], hp.inputs[0])
            dc = nt.nodes.new("ShaderNodeMath")
            dc.operation = "LESS_THAN"
            nt.links.new(dist.outputs["Value"], dc.inputs[0])
            nt.links.new(hp.outputs[0], dc.inputs[1])
            vm = nt.nodes.new("ShaderNodeMath")
            vm.operation = "MINIMUM"
            nt.links.new(ray.outputs["Is Hit"], vm.inputs[0])
            nt.links.new(dc.outputs[0], vm.inputs[1])
            nt.links.new(vm.outputs[0], em.inputs["Color"])
        elif mode in ("debug_vis", "debug_dcheck", "debug_vis2"):
            dist = nt.nodes.new("ShaderNodeVectorMath")
            dist.operation = "LENGTH"
            nt.links.new(sub.outputs["Vector"], dist.inputs[0])
            ndir = nt.nodes.new("ShaderNodeVectorMath")
            ndir.operation = "NORMALIZE"
            nt.links.new(sub.outputs["Vector"], ndir.inputs[0])
            ray = nt.nodes.new("ShaderNodeRaycast")
            nt.links.new(cp.outputs["Vector"], ray.inputs["Position"])
            nt.links.new(ndir.outputs["Vector"], ray.inputs["Direction"])
            ln = nt.nodes.new("ShaderNodeMath")
            ln.operation = "MULTIPLY_ADD"
            ln.inputs[1].default_value = 1.0
            ln.inputs[2].default_value = RAY_EPS * 10.0
            nt.links.new(dist.outputs["Value"], ln.inputs[0])
            nt.links.new(ln.outputs[0], ray.inputs["Length"])
            hp = nt.nodes.new("ShaderNodeMath")
            hp.operation = "ADD"
            hp.inputs[1].default_value = RAY_EPS
            nt.links.new(ray.outputs["Hit Distance"], hp.inputs[0])
            dc = nt.nodes.new("ShaderNodeMath")
            dc.operation = "LESS_THAN"
            nt.links.new(dist.outputs["Value"], dc.inputs[0])
            nt.links.new(hp.outputs[0], dc.inputs[1])
            if mode == "debug_vis":
                nt.links.new(ray.outputs["Is Hit"], em.inputs["Color"])
            elif mode == "debug_dcheck":
                nt.links.new(dc.outputs[0], em.inputs["Color"])
            else:   # debug_vis2 = MINIMUM(Is Hit, dist_check)
                vm = nt.nodes.new("ShaderNodeMath")
                vm.operation = "MINIMUM"
                nt.links.new(ray.outputs["Is Hit"], vm.inputs[0])
                nt.links.new(dc.outputs[0], vm.inputs[1])
                nt.links.new(vm.outputs[0], em.inputs["Color"])
        elif mode == "debug_facing":
            dot = nt.nodes.new("ShaderNodeVectorMath")
            dot.operation = "DOT_PRODUCT"
            dot.inputs[1].default_value = tuple(float(x) for x in cam_dir)
            nt.links.new(geo.outputs["Normal"], dot.inputs[0])
            mx = nt.nodes.new("ShaderNodeMath")
            mx.operation = "MAXIMUM"
            mx.inputs[1].default_value = 0.0
            nt.links.new(dot.outputs["Value"], mx.inputs[0])
            if mode == "debug_facing":
                nt.links.new(mx.outputs[0], em.inputs["Color"])
        elif mode == "debug_pow":
            dot = nt.nodes.new("ShaderNodeVectorMath")
            dot.operation = "DOT_PRODUCT"
            dot.inputs[1].default_value = tuple(float(x) for x in cam_dir)
            nt.links.new(geo.outputs["Normal"], dot.inputs[0])
            mx = nt.nodes.new("ShaderNodeMath")
            mx.operation = "MAXIMUM"
            mx.inputs[1].default_value = 0.0
            nt.links.new(dot.outputs["Value"], mx.inputs[0])
            mn1 = nt.nodes.new("ShaderNodeMath")
            mn1.operation = "MINIMUM"
            mn1.inputs[1].default_value = 1.0
            nt.links.new(mx.outputs[0], mn1.inputs[0])
            pw = nt.nodes.new("ShaderNodeMath")
            pw.operation = "POWER"
            pw.inputs[1].default_value = EXP
            nt.links.new(mn1.outputs[0], pw.inputs[0])
            if mode == "debug_pow":
                nt.links.new(pw.outputs[0], em.inputs["Color"])
        else:   # debug_edge
            def ax(vec):
                dv = nt.nodes.new("ShaderNodeVectorMath")
                dv.operation = "DOT_PRODUCT"
                dv.inputs[1].default_value = tuple(float(x) for x in vec)
                nt.links.new(sub.outputs["Vector"], dv.inputs[0])
                sc = nt.nodes.new("ShaderNodeMath")
                sc.operation = "MULTIPLY"
                sc.inputs[1].default_value = 2.0 / ORTHO
                nt.links.new(dv.outputs["Value"], sc.inputs[0])
                ab = nt.nodes.new("ShaderNodeMath")
                ab.operation = "ABSOLUTE"
                nt.links.new(sc.outputs[0], ab.inputs[0])
                return ab.outputs[0]
            e = nt.nodes.new("ShaderNodeMath")
            e.operation = "MAXIMUM"
            nt.links.new(ax(cam_right), e.inputs[0])
            nt.links.new(ax(cam_up), e.inputs[1])
            nt.links.new(e.outputs[0], em.inputs["Color"])
        tgt = nt.nodes.new("ShaderNodeTexImage")
        tgt.image = target
        tgt.select = True
        nt.nodes.active = tgt
        return mat

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(CFG["views"][key]["image"])
    tex.extension = "CLIP"          # 视图外不要重复平铺
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = uvmap_name
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])

    if mode == "color":
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    else:
        geo = nt.nodes.new("ShaderNodeNewGeometry")

        # ── facing = clamp(dot(N, cam_dir), 0, 1) ────────────────────────
        # StableGen: dot_d_normal(normalize(cam→P), N)，再取负、clamp 到 [0,1]
        # （camera→surface 与朝外法线点积为负 = 正面朝相机）。这里 cam_dir
        # 本身就指向相机，所以直接 dot 即为正值，等价。
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = tuple(float(x) for x in cam_dir)
        nt.links.new(geo.outputs["Normal"], dot.inputs[0])
        mx = nt.nodes.new("ShaderNodeMath")
        mx.operation = "MAXIMUM"
        mx.inputs[1].default_value = 0.0
        nt.links.new(dot.outputs["Value"], mx.inputs[0])
        mn1 = nt.nodes.new("ShaderNodeMath")
        mn1.operation = "MINIMUM"
        mn1.inputs[1].default_value = 1.0
        nt.links.new(mx.outputs[0], mn1.inputs[0])
        # StableGen 的 clamp_dot2 = MINIMUM(clamp_dot, 1.0)
        pw = nt.nodes.new("ShaderNodeMath")
        pw.operation = "POWER"
        pw.inputs[1].default_value = EXP
        nt.links.new(mn1.outputs[0], pw.inputs[0])
        w_out = pw.outputs[0]

        # ── 角度截断：angle_check = deg(acos(facing)) < discard_factor ──
        # 超过 70° 的掠射面权重归零，否则会被拉成条纹（StableGen shader 同款）。
        cos_cut = math.cos(math.radians(DISCARD_DEG))
        ang_ok = nt.nodes.new("ShaderNodeMath")
        ang_ok.operation = "GREATER_THAN"
        ang_ok.inputs[1].default_value = cos_cut
        nt.links.new(mn1.outputs[0], ang_ok.inputs[0])
        w_ang = nt.nodes.new("ShaderNodeMath")
        w_ang.operation = "MULTIPLY"
        nt.links.new(w_out, w_ang.inputs[0])
        nt.links.new(ang_ok.outputs[0], w_ang.inputs[1])
        w_out = w_ang.outputs[0]

        # ── alpha：生成图带透明时不把黑底投上去 ────────────────────────
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(w_out, mul.inputs[0])
        nt.links.new(tex.outputs["Alpha"], mul.inputs[1])
        w_out = mul.outputs[0]

        if OCCLUSION == "shader" and cam_loc is not None:
            # 与 StableGen create_native_raycast_visibility 逐节点一致
            # （projection.py:1550-1620）：
            #   dir  = normalize(P - camLoc)
            #   ray  = Raycast(Position=camLoc, Direction=dir, Length=dist + eps*10)
            #   vis  = min(Is Hit, dist < HitDistance + eps)
            # Cycles 会排除「正在着色的那个图元」，所以一条打向可见表面的射线会
            # 穿过去、命中的是表面背后的几何 → Is Hit=1；若表面之前有东西挡着，
            # HitDistance < dist → 第二项为 0 → 不可见。Length 加 eps*10 是为了
            # 保证一定越过表面本身。之前我用 SUBTRACT(dist-eps) 是反的。
            to_surf = nt.nodes.new("ShaderNodeVectorMath")
            to_surf.operation = "SUBTRACT"
            to_surf.inputs[1].default_value = tuple(float(x) for x in cam_loc)
            nt.links.new(geo.outputs["Position"], to_surf.inputs[0])

            dist = nt.nodes.new("ShaderNodeVectorMath")
            dist.operation = "LENGTH"
            nt.links.new(to_surf.outputs["Vector"], dist.inputs[0])

            ndir = nt.nodes.new("ShaderNodeVectorMath")
            ndir.operation = "NORMALIZE"
            nt.links.new(to_surf.outputs["Vector"], ndir.inputs[0])

            cam_xyz = nt.nodes.new("ShaderNodeCombineXYZ")
            for i, axis in enumerate(("X", "Y", "Z")):
                cam_xyz.inputs[i].default_value = float(cam_loc[i])

            ray = nt.nodes.new("ShaderNodeRaycast")
            nt.links.new(cam_xyz.outputs["Vector"], ray.inputs["Position"])
            nt.links.new(ndir.outputs["Vector"], ray.inputs["Direction"])
            # Length = dist + eps*10
            ln = nt.nodes.new("ShaderNodeMath")
            ln.operation = "MULTIPLY_ADD"
            ln.inputs[1].default_value = 1.0
            ln.inputs[2].default_value = RAY_EPS * 10.0
            nt.links.new(dist.outputs["Value"], ln.inputs[0])
            nt.links.new(ln.outputs[0], ray.inputs["Length"])

            # hit_dist_plus = HitDistance + eps
            hit_plus = nt.nodes.new("ShaderNodeMath")
            hit_plus.operation = "ADD"
            hit_plus.inputs[1].default_value = RAY_EPS
            nt.links.new(ray.outputs["Hit Distance"], hit_plus.inputs[0])

            # dist_check = LESS_THAN(dist, hit_dist_plus)
            dist_check = nt.nodes.new("ShaderNodeMath")
            dist_check.operation = "LESS_THAN"
            nt.links.new(dist.outputs["Value"], dist_check.inputs[0])
            nt.links.new(hit_plus.outputs[0], dist_check.inputs[1])

            # occlusion_ok = MINIMUM(Is Hit, dist_check)
            vis = nt.nodes.new("ShaderNodeMath")
            vis.operation = "MINIMUM"
            nt.links.new(ray.outputs["Is Hit"], vis.inputs[0])
            nt.links.new(dist_check.outputs[0], vis.inputs[1])

            mul3 = nt.nodes.new("ShaderNodeMath")
            mul3.operation = "MULTIPLY"
            nt.links.new(w_out, mul3.inputs[0])
            nt.links.new(vis.outputs[0], mul3.inputs[1])
            w_out = mul3.outputs[0]

        elif OCCLUSION == "depth" and zdepth is not None:
            # 可见性 = 「该点视线深度 ≤ 深度图在该屏幕位置的前沿深度 + eps」
            # 深度图与 ProjUV 同一坐标系，直接在 shader 里比，宿主侧零改动。
            ztex = nt.nodes.new("ShaderNodeTexImage")
            ztex.image = bpy.data.images.load(zdepth)
            ztex.extension = "CLIP"
            ztex.interpolation = "Closest"   # 深度图不能插值，否则边缘会糊出假遮挡
            ztex.image.colorspace_settings.name = "Non-Color"
            nt.links.new(uvn.outputs["UV"], ztex.inputs["Vector"])

            sub = nt.nodes.new("ShaderNodeVectorMath")
            sub.operation = "SUBTRACT"
            sub.inputs[1].default_value = tuple(float(x) for x in cam_loc)
            nt.links.new(geo.outputs["Position"], sub.inputs[0])
            pdot = nt.nodes.new("ShaderNodeVectorMath")
            pdot.operation = "DOT_PRODUCT"
            pdot.inputs[1].default_value = tuple(-float(x) for x in cam_dir)
            nt.links.new(sub.outputs["Vector"], pdot.inputs[0])

            span = DEPTH_FAR - DEPTH_NEAR
            nrm = nt.nodes.new("ShaderNodeMath")
            nrm.operation = "MULTIPLY_ADD"
            nrm.inputs[1].default_value = 1.0 / span
            nrm.inputs[2].default_value = -DEPTH_NEAR / span
            nt.links.new(pdot.outputs["Value"], nrm.inputs[0])

            # d = own - front；d<=0 可见 →1，d>=eps 遮挡 →0
            diff = nt.nodes.new("ShaderNodeMath")
            diff.operation = "SUBTRACT"
            nt.links.new(nrm.outputs[0], diff.inputs[0])
            nt.links.new(ztex.outputs["Color"], diff.inputs[1])
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = "MULTIPLY"
            inv.inputs[1].default_value = -1.0 / DEPTH_EPS
            nt.links.new(diff.outputs[0], inv.inputs[0])
            vis = nt.nodes.new("ShaderNodeMath")
            vis.operation = "MINIMUM"
            vis.inputs[1].default_value = 1.0
            nt.links.new(inv.outputs[0], vis.inputs[0])
            vis2 = nt.nodes.new("ShaderNodeMath")
            vis2.operation = "MAXIMUM"
            vis2.inputs[1].default_value = 0.0
            nt.links.new(vis.outputs[0], vis2.inputs[0])

            mul2 = nt.nodes.new("ShaderNodeMath")
            mul2.operation = "MULTIPLY"
            nt.links.new(mul.outputs[0], mul2.inputs[0])
            nt.links.new(vis2.outputs[0], mul2.inputs[1])
            w_out = mul2.outputs[0]

        if vis_attr:
            # BVH 逐顶点可见性，POINT 域 attribute，bake 时按 texel 插值。
            at = nt.nodes.new("ShaderNodeAttribute")
            at.attribute_name = vis_attr
            vm = nt.nodes.new("ShaderNodeMath")
            vm.operation = "MULTIPLY"
            nt.links.new(w_out, vm.inputs[0])
            nt.links.new(at.outputs["Fac"], vm.inputs[1])
            w_out = vm.outputs[0]

        if cam_right is not None and cam_up is not None:
            # 视锥坐标 + 边缘羽化，照 StableGen create_native_feather（:1779）：
            #   x = dot(P-camLoc, right) / 半画幅，y 同理，edge = max(|x|,|y|)
            #   frustum_ok = edge < 1
            #   frame      = 1 - smoothstep(1-feather, 1, edge)，再取 gamma
            # StableGen 用透视 filmw；本脚本是正交相机，除以半边长 ORTHO/2 即可。
            rel = nt.nodes.new("ShaderNodeVectorMath")
            rel.operation = "SUBTRACT"
            rel.inputs[1].default_value = tuple(float(x) for x in cam_loc)
            nt.links.new(geo.outputs["Position"], rel.inputs[0])

            def axis(vec):
                dotv = nt.nodes.new("ShaderNodeVectorMath")
                dotv.operation = "DOT_PRODUCT"
                dotv.inputs[1].default_value = tuple(float(x) for x in vec)
                nt.links.new(rel.outputs["Vector"], dotv.inputs[0])
                sc = nt.nodes.new("ShaderNodeMath")
                sc.operation = "MULTIPLY"
                sc.inputs[1].default_value = 2.0 / ORTHO     # 半边长归一化
                nt.links.new(dotv.outputs["Value"], sc.inputs[0])
                ab = nt.nodes.new("ShaderNodeMath")
                ab.operation = "ABSOLUTE"
                nt.links.new(sc.outputs[0], ab.inputs[0])
                return ab.outputs[0]

            ax, ay = axis(cam_right), axis(cam_up)
            edge = nt.nodes.new("ShaderNodeMath")
            edge.operation = "MAXIMUM"
            nt.links.new(ax, edge.inputs[0])
            nt.links.new(ay, edge.inputs[1])

            # 视锥外直接 0（StableGen frustum_ok）
            inside = nt.nodes.new("ShaderNodeMath")
            inside.operation = "LESS_THAN"
            inside.inputs[1].default_value = 1.0
            nt.links.new(edge.outputs[0], inside.inputs[0])
            w_in = nt.nodes.new("ShaderNodeMath")
            w_in.operation = "MULTIPLY"
            nt.links.new(w_out, w_in.inputs[0])
            nt.links.new(inside.outputs[0], w_in.inputs[1])
            w_out = w_in.outputs[0]

            if feather > 0.0:
                mr = nt.nodes.new("ShaderNodeMapRange")
                mr.interpolation_type = "SMOOTHSTEP"
                mr.inputs["From Min"].default_value = 1.0 - float(feather)
                mr.inputs["From Max"].default_value = 1.0
                mr.inputs["To Min"].default_value = 1.0
                mr.inputs["To Max"].default_value = 0.0
                mr.clamp = True
                nt.links.new(edge.outputs[0], mr.inputs["Value"])
                frame = mr.outputs["Result"]

                if abs(FEATHER_GAMMA - 1.0) > 1e-6:
                    gm = nt.nodes.new("ShaderNodeMath")
                    gm.operation = "POWER"
                    gm.inputs[1].default_value = FEATHER_GAMMA
                    nt.links.new(frame, gm.inputs[0])
                    frame = gm.outputs[0]

                fe = nt.nodes.new("ShaderNodeMath")
                fe.operation = "MULTIPLY"
                nt.links.new(w_out, fe.inputs[0])
                nt.links.new(frame, fe.inputs[1])
                w_out = fe.outputs[0]

        # float -> Color 会广播成 (w,w,w,1)，不必依赖各版本 CombineColor 的输入名
        nt.links.new(w_out, em.inputs["Color"])

    tgt = nt.nodes.new("ShaderNodeTexImage")
    tgt.image = target
    tgt.select = True
    nt.nodes.active = tgt

    if os.environ.get("SG_DUMP_GRAPH") and mode == "weight":
        def dump(node, depth=0, seen=None):
            seen = seen or set()
            if node.name in seen or depth > 14:
                return
            seen.add(node.name)
            op = getattr(node, "operation", "")
            extra = ""
            if node.bl_idname == "ShaderNodeMath":
                extra = " b=" + ",".join(
                    f"{i:.4g}" for i in (node.inputs[1].default_value,
                                         node.inputs[2].default_value
                                         if len(node.inputs) > 2 else 0))
            if node.bl_idname == "ShaderNodeMapRange":
                extra = f" [{node.inputs['From Min'].default_value:.3f}..{node.inputs['From Max'].default_value:.3f}]"
            print("GRAPH " + "  " * depth + f"{node.bl_idname}/{op}{extra}", flush=True)
            for inp in node.inputs:
                for link in inp.links:
                    print("GRAPH " + "  " * (depth + 1) + f"<- {link.from_socket.name}", flush=True)
                    dump(link.from_node, depth + 2, seen)

        for link in em.inputs["Color"].links:
            dump(link.from_node)
    return mat


def compute_vertex_visibility(cam_dir, cam_loc, bvh):
    """逐顶点遮挡：**从相机打向顶点，长度 dist - eps，命中即被挡**。

    这是 StableGen `_bake_visibility_weights` 的判据
    （projection.py:1180-1195）：
        hit, _, _, hit_dist = scene_tree.ray_cast(cam_loc, dir, dist + eps*20)
        occluded = hit is not None and dist > hit_dist + eps
    等价写法：把射线长度截到 dist - eps，只要命中就说明前面有东西。

    ⚠️ 之前的实现是「从顶点往相机退 dist+eps 再打回来」，起点落在模型外面，
    必然打空 → 可见性 93~98%（等于没做遮挡）。这版修正了起点与方向。

    ⚠️ TRELLIS.2 白模是**薄壳**（实测平均厚度 0.005），内壁顶点会被外壁挡住，
    因此内壁 texel 权重为 0 —— 与 StableGen 的 `dist_check` 行为一致，
    不是 bug；那些区域由其它相机补。
    """
    from mathutils import Vector

    verts = mesh.vertices
    n = len(verts)
    vis = np.ones(n, dtype=np.float32)
    loc = Vector(tuple(float(x) for x in cam_loc))
    margin = max(RAY_EPS * 10.0, 1e-6)

    for i, v in enumerate(verts):
        p = v.co
        d = p - loc
        dist = d.length
        if dist < 1e-6:
            vis[i] = 0.0
            continue
        nd = d / dist
        hit = bvh.ray_cast(loc, nd, max(dist - margin, 1e-6))
        if hit[0] is not None:
            vis[i] = 0.0
    return vis


def write_vertex_attr(name, values):
    attrs = mesh.attributes
    a = attrs.get(name) or attrs.new(name, "FLOAT", "POINT")
    a.data.foreach_set("value", values)
    return name


def render_zdepth(path):
    """用同一台正交相机渲一张归一化深度图（0=近 1=远），16 位 PNG。

    只当「前沿深度」用，不需要抗锯齿或色彩管理；Cycles 的 Z pass 是解析值，
    1 sample 就够。
    """
    scene.camera = cam
    vl = bpy.context.view_layer
    vl.use_pass_z = True

    # Blender 5.x 把合成器从 scene.node_tree 移到了 compositing_node_group，
    # 且移除了 CompositorNodeMapRange / CompositorNodeComposite（改用
    # ShaderNodeMapRange + NodeGroupOutput）。4.5 走旧接口。
    if hasattr(scene, "compositing_node_group"):
        nt = scene.compositing_node_group
        if nt is None:
            nt = bpy.data.node_groups.new("SGCompositing", "CompositorNodeTree")
            scene.compositing_node_group = nt
        nt.nodes.clear()
        if not any(s.name == "Image" for s in nt.interface.items_tree):
            nt.interface.new_socket(name="Image", in_out="OUTPUT",
                                    socket_type="NodeSocketColor")
        rl = nt.nodes.new("CompositorNodeRLayers")
        rl.scene = scene
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.clamp = True
        mr.inputs["From Min"].default_value = DEPTH_NEAR
        mr.inputs["From Max"].default_value = DEPTH_FAR
        nt.links.new(rl.outputs["Depth"], mr.inputs["Value"])
        out = nt.nodes.new("NodeGroupOutput")
        nt.links.new(mr.outputs["Result"], out.inputs["Image"])
    else:
        scene.use_nodes = True
        nt = scene.node_tree
        nt.nodes.clear()
        rl = nt.nodes.new("CompositorNodeRLayers")
        rl.scene = scene
        mr = nt.nodes.new("CompositorNodeMapRange")
        mr.use_clamp = True
        mr.inputs["From Min"].default_value = DEPTH_NEAR
        mr.inputs["From Max"].default_value = DEPTH_FAR
        nt.links.new(rl.outputs["Depth"], mr.inputs["Value"])
        comp = nt.nodes.new("CompositorNodeComposite")
        nt.links.new(mr.outputs["Value"], comp.inputs["Image"])

    img_set = scene.render.image_settings
    old_fmt, old_depth = img_set.file_format, img_set.color_depth
    img_set.file_format = "PNG"
    img_set.color_depth = "16"
    img_set.color_mode = "BW"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img_set.file_format, img_set.color_depth = old_fmt, old_depth
    img_set.color_mode = "RGBA"
    print("BAKE zdepth", path, flush=True)


def bake_once(mat, target):
    mesh.materials.clear()
    mesh.materials.append(mat)
    bpy.ops.object.bake(type="EMIT")
    buf = np.empty(SIZE * SIZE * 4, np.float32)
    target.pixels.foreach_get(buf)
    return buf.reshape(SIZE, SIZE, 4)


# Blender 单个 mesh 最多 8 个 UV layer。每视图 new 一个 ProjUV 会撞上限：
# 第 8 个返回 None，foreach_set 抛 AttributeError，而 Blender 仍以 0 退出，
# 宿主侧会「成功」合并成少一个视图的贴图（实测 8 机位只烘出 7 个，84.6%）。
# 所有视图复用同一个 ProjUV，逐轮覆盖坐标即可，机位数不再受限制。
proj = mesh.uv_layers.get("ProjUV") or mesh.uv_layers.new(name="ProjUV")
if proj is None:
    raise RuntimeError("创建 ProjUV 失败（UV layer 已达 8 个上限）")
mesh.uv_layers.active = bakeuv
bakeuv.active_render = True

for key, v in CFG["views"].items():
    az = math.radians(float(v.get("azimuth", 0)))
    el = math.radians(float(v.get("elevation", 0)))
    d = Vector((math.sin(az) * math.cos(el),
                -math.cos(az) * math.cos(el),
                math.sin(el)))
    cam_data = bpy.data.cameras.new("cam_" + key)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ORTHO
    cam = bpy.data.objects.new("cam_" + key, cam_data)
    scene.collection.objects.link(cam)
    cam.location = d * 3.5
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    # 相机投影 UV：逐顶点算，再按 loop 展开；三角形内部由 bake 光栅化插值
    inv = np.array(cam.matrix_world.inverted(), dtype=np.float64)
    P = V @ inv[:3, :3].T + inv[:3, 3]
    u = 0.5 + P[:, 0] / ORTHO
    w = 0.5 + P[:, 1] / ORTHO
    flat = np.stack([u[lvi], w[lvi]], axis=1).astype(np.float32).ravel()
    if proj is None:
        raise RuntimeError(f"{key}: ProjUV 不可用")
    proj.data.foreach_set("uv", flat)
    print(f"BAKE view {key}: az={v.get('azimuth', 0)} el={v.get('elevation', 0)} "
          f"u=[{u.min():.3f},{u.max():.3f}] v=[{w.min():.3f},{w.max():.3f}]", flush=True)

    color_img = new_target("color_" + key)
    cmat = build_mat(key, "color", color_img, proj.name)
    color = bake_once(cmat, color_img)[..., :3].copy()

    zdepth_path = os.path.join(VDIR, key + "_zdepth.png")
    if OCCLUDE:
        render_zdepth(zdepth_path)

    weight_img = new_target("weight_" + key)
    rot = cam.matrix_world.to_3x3()
    vis_name = None
    if OCCLUSION == "bvh" and BVH is not None:
        vis = compute_vertex_visibility(d, cam.location, BVH)
        vis_name = write_vertex_attr("SG_VIS_" + key, vis)
        print(f"BAKE view {key}: bvh visible {(vis > 0.5).mean():.1%}", flush=True)
    wmat = build_mat(key, "weight", weight_img, proj.name, cam_dir=d,
                     cam_loc=cam.location,
                     zdepth=zdepth_path if OCCLUDE else None,
                     cam_right=rot @ Vector((1.0, 0.0, 0.0)),
                     cam_up=rot @ Vector((0.0, 1.0, 0.0)),
                     feather=FEATHER, vis_attr=vis_name)
    weight = bake_once(wmat, weight_img)[..., 0].copy()

    print(f"BAKE wstats {key}: mean={weight.mean():.4f} max={weight.max():.4f} "
          f">0.01={(weight > 0.01).mean():.1%}", flush=True)
    if os.environ.get("SG_DEBUG_VIS") and key == next(iter(CFG["views"])):
        for m in ("debug_dcheck_1", "debug_dcheck_100", "debug_dcheck_10000", "debug_dcheck_1000000"):
            dimg = new_target(m + key)
            dm = build_mat(key, m, dimg, proj.name, cam_dir=d,
                           cam_loc=cam.location,
                           cam_right=rot @ Vector((1.0, 0.0, 0.0)),
                           cam_up=rot @ Vector((0.0, 1.0, 0.0)))
            arr = bake_once(dm, dimg)[..., 0]
            print(f"BAKE DEBUG {m}: mean={arr.mean():.4f} max={arr.max():.4f} "
                  f">0.01={(arr > 0.01).mean():.1%}", flush=True)
            bpy.data.images.remove(dimg)

    np.save(os.path.join(VDIR, key + "_color.npy"), color)
    np.save(os.path.join(VDIR, key + "_weight.npy"), weight)
    print(f"BAKE view {key}: weight>0.01 覆盖 {(weight > 0.01).mean():.1%}", flush=True)

    bpy.data.images.remove(color_img)
    bpy.data.images.remove(weight_img)
    bpy.data.objects.remove(cam, do_unlink=True)

json.dump({"size": SIZE, "ortho": ORTHO, "exponent": EXP,
           "faces": len(mesh.polygons), "verts": len(mesh.vertices),
           "center": list(center), "scale": scale},
          open(os.path.join(WORK, "meta.json"), "w"))
print("BAKE ALL DONE", flush=True)
