"""StableGen `project_image`（projection.py:2248）+ `bake_texture`（rendering.py:2676）
的等价实现：**一个材质内所有相机求和÷权重和，只 bake 一次**。

    blender.exe -b -P tools/blender_project_all.py -- <config.json>

config:
    {"mesh": "white.glb", "ortho": 2.16, "size": 2048,
     "views": [{"name":"v00","image":"ready/v00.png",
                "azimuth":0,"elevation":0}, ...],
     "exponent": 7.0, "margin": 4, "unwrap": "basic",
     "atlas": ".../atlas.png", "out": ".../final.glb"}

与 `blender_context_projection.py` 共用同一套相机投影 UV（CORNER 域
`sg_uv_*` attribute，照 projection.py:360）与权重公式，保证 ctx render 与
最终 bake **同源**——这是 StableGen 的关键：ctx 渲的就是投影材质本身。
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

ORTHO = float(CFG.get("ortho", 2.16))
SIZE = int(CFG.get("size", 2048))
MARGIN = int(CFG.get("margin", 4))
K = float(CFG.get("exponent", 7.0))
VIEWS = CFG["views"]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=CFG["mesh"])
scene = bpy.context.scene
meshes = [o for o in scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("没有 mesh")
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
mesh = obj.data

pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
mx = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
if (mx - mn).length > 1e-6:
    scale = 2.0 / max(mx - mn)
    obj.matrix_world = (Matrix.Scale(scale, 4)
                        @ Matrix.Translation(-(mn + mx) / 2)
                        @ obj.matrix_world)
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def cam_basis(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = Vector((math.sin(az) * math.cos(el),
                -math.cos(az) * math.cos(el),
                math.sin(el)))
    loc = d * 3.5
    rot = (Vector((0, 0, 0)) - loc).to_track_quat("-Z", "Y").to_matrix()
    m = rot.to_4x4()
    m.translation = loc
    return d, m, rot @ Vector((1, 0, 0)), rot @ Vector((0, 1, 0))


# --- 1) 所有相机的投影 UV（CORNER 域 attribute）---
V = np.empty(len(mesh.vertices) * 3, np.float32)
mesh.vertices.foreach_get("co", V)
V = V.reshape(-1, 3).astype(np.float64)
lvi = np.empty(len(mesh.loops), np.int32)
mesh.loops.foreach_get("vertex_index", lvi)
starts = np.empty(len(mesh.polygons), np.int32)
totals = np.empty(len(mesh.polygons), np.int32)
mesh.polygons.foreach_get("loop_start", starts)
mesh.polygons.foreach_get("loop_total", totals)

covered = np.zeros(len(mesh.polygons), bool)
all_uv = {}
for v in VIEWS:
    d, mat_cam, right, up = cam_basis(float(v.get("azimuth", 0)),
                                      float(v.get("elevation", 0)))
    inv = np.array(mat_cam.inverted(), dtype=np.float64)
    P = V @ inv[:3, :3].T + inv[:3, 3]
    # 与 blender_bake_views.py 的 ProjUV 完全一致（该实现已验证能出正确贴图）：
    # P.y 向上增大 → v 增大 → TexImage 采样图像上部，**不取负**。
    uv = np.stack([0.5 + P[:, 0] / ORTHO,
                   0.5 + P[:, 1] / ORTHO], axis=1)
    all_uv[v["name"]] = uv
    # covered_faces：面 bbox 与 [0,1] 相交（rendering.py:2165）
    lu = uv[lvi, 0]
    lv = uv[lvi, 1]
    minu, maxu = np.minimum.reduceat(lu, starts), np.maximum.reduceat(lu, starts)
    minv, maxv = np.minimum.reduceat(lv, starts), np.maximum.reduceat(lv, starts)
    covered |= ((maxu >= 0) & (minu <= 1) & (maxv >= 0) & (minv <= 1))

# --- 2) BakeUV：unwrap + compact（rendering.py:2298）---
# Cycles **bake 会忽略 TexImage 的自定义 Vector**，强制用 active UV；
# 只有 ShaderNodeUVMap 指名的 UV layer 可靠（blender_bake_views.py 已验证）。
# Blender 5.4 最多 8 层 → 7 个 ProjUV + 1 个 BakeUV。
if len(VIEWS) > 7:
    raise RuntimeError(f"Blender UV layer 上限 8（7 投影 + 1 BakeUV），收到 {len(VIEWS)} 视图")
for layer in list(mesh.uv_layers):
    mesh.uv_layers.remove(layer)
for v in VIEWS:
    layer = mesh.uv_layers.new(name="ProjUV_" + v["name"])
    layer.data.foreach_set("uv", all_uv[v["name"]][lvi].astype(np.float32).ravel())
bakeuv = mesh.uv_layers.new(name="BakeUV")
mesh.uv_layers.active = bakeuv
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
UV_GAP = max(2.0 / SIZE, 0.001)
if CFG.get("unwrap", "basic") == "basic":
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=UV_GAP)
    for kw in ({"margin": UV_GAP, "rotate": True, "scale": True},
               {"margin": UV_GAP}, {}):
        try:
            bpy.ops.uv.pack_islands(**kw)
            break
        except (TypeError, RuntimeError):
            continue
else:
    bpy.ops.uv.smart_project(island_margin=UV_GAP,
                             rotate_method="AXIS_ALIGNED")
bpy.ops.object.mode_set(mode="OBJECT")
mesh = obj.data                       # mode 切换后 data 引用要重新取
bakeuv = mesh.uv_layers.get("BakeUV")
if bakeuv is None:
    raise RuntimeError("BakeUV 层丢失")
mesh.uv_layers.active = bakeuv
bakeuv.active_render = True
for layer in mesh.uv_layers:
    if layer.name != "BakeUV":
        layer.active_render = False

loopmask = np.zeros(len(mesh.loops), bool)
for i in np.flatnonzero(covered):
    loopmask[starts[i]:starts[i] + totals[i]] = True
uv_all = np.empty(len(mesh.loops) * 2, np.float32)
bakeuv.data.foreach_get("uv", uv_all)
uv_all = uv_all.reshape(-1, 2)
guard = min(0.2, max(2.0 / SIZE, (MARGIN + 2.0) / SIZE))
target = max(0.01, 1.0 - 2.0 * guard)
if loopmask.any():
    cl = uv_all[loopmask]
    su = target / max(float(cl[:, 0].max() - cl[:, 0].min()), 1e-12)
    sv = target / max(float(cl[:, 1].max() - cl[:, 1].min()), 1e-12)
    uv_all[loopmask, 0] = guard + (cl[:, 0] - cl[:, 0].min()) * su
    uv_all[loopmask, 1] = guard + (cl[:, 1] - cl[:, 1].min()) * sv
uv_all[~loopmask] = 0.0
bakeuv.data.foreach_set("uv", uv_all.ravel())
print(f"PROJECT compact: covered {covered.mean():.1%} faces guard={guard:.4f}",
      flush=True)
mesh.uv_layers.active = bakeuv
bakeuv.active_render = True

# --- 1b) 可见性（StableGen `_bake_visibility_weights` rendering.py:938）---
# 只靠 cos^k 时，背视图的颜色会糊到正面（薄壳网格法线不可靠）。逐顶点从
# 相机打 BVH 射线，命中别的面即判遮挡；结果写 POINT 域 FLOAT attribute，
# Cycles bake 可读（CORNER FLOAT2 不可读，POINT FLOAT 可读）。
from mathutils.bvhtree import BVHTree
_verts = [Vector(tuple(float(x) for x in v.co)) for v in mesh.vertices]
_polys = [list(f.vertices) for f in mesh.polygons]
_bvh = BVHTree.FromPolygons(_verts, _polys, all_triangles=False)
_diag = float((mx - mn).length) * scale if (mx - mn).length > 1e-6 else 1.0
_eps = max(_diag * 1e-5, 1e-7)
# 本白模是薄壳（实测平均厚 0.005）：源码 eps=bbox*1e-5 会让内壁被判成
# 遮挡（可见性掉到 20%）。容差取对角线的 1%，能排除内壁、仍能挡住
# 身体前后等真正的遮挡。
_tol = max(_diag * 0.01, 1e-4)
for v in VIEWS:
    _d, _mcam, _r, _u = cam_basis(float(v.get("azimuth", 0)),
                                   float(v.get("elevation", 0)))
    _loc = Vector(tuple(float(x) for x in _mcam.translation))
    _name = "SG_VIS_" + v["name"]
    _old = mesh.attributes.get(_name)
    if _old:
        mesh.attributes.remove(_old)
    _attr = mesh.attributes.new(_name, "FLOAT_VECTOR", "POINT")
    _vals = np.ones(len(_verts), np.float32)
    for _i, _p in enumerate(_verts):
        _delta = _p - _loc
        _dist = _delta.length
        if _dist < 1e-6:
            continue
        _hit = _bvh.ray_cast(_loc, _delta.normalized(), _dist + _tol)
        if _hit[0] is not None and float(_hit[3]) < _dist - _tol:
            _vals[_i] = 0.0
    # FLOAT 的 Fac 输出在 Cycles bake 里读不到；FLOAT_VECTOR/Vector 可读
    _attr.data.foreach_set("vector", np.stack([_vals] * 3, axis=1).ravel())
    print(f"PROJECT vis {v['name']}: visible {_vals.mean():.1%}", flush=True)

# --- 3) 一个材质：Σ(color_i · w_i) / Σw_i ---
mat = bpy.data.materials.new("SGProjection")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
em = nt.nodes.new("ShaderNodeEmission")
nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
geo = nt.nodes.new("ShaderNodeNewGeometry")
acc = None
wsum = None
first_attr = None
first_tex = None

for v in VIEWS:
    d, mat_cam, right, up = cam_basis(float(v.get("azimuth", 0)),
                                      float(v.get("elevation", 0)))
    # 每个视图一套 UVMap → TexImage.Vector（bake 里唯一可靠的自定义 UV 方式）
    uvnode = nt.nodes.new("ShaderNodeUVMap")
    uvnode.uv_map = "ProjUV_" + v["name"]
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uvnode.outputs["UV"], sep.inputs["Vector"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(v["image"])
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    nt.links.new(uvnode.outputs["UV"], tex.inputs["Vector"])
    if first_attr is None:
        first_attr, first_tex = uvnode, tex

    # frustum_ok：UV 在 [0,1] 内
    fr = []
    for sock in (sep.outputs["X"], sep.outputs["Y"]):
        ge0 = nt.nodes.new("ShaderNodeMath")
        ge0.operation = "GREATER_THAN"
        ge0.inputs[1].default_value = 0.0
        nt.links.new(sock, ge0.inputs[0])
        le1 = nt.nodes.new("ShaderNodeMath")
        le1.operation = "LESS_THAN"
        le1.inputs[1].default_value = 1.0
        nt.links.new(sock, le1.inputs[0])
        fr.append((ge0.outputs[0], le1.outputs[0]))
    infr = nt.nodes.new("ShaderNodeMath")
    infr.operation = "MINIMUM"
    nt.links.new(fr[0][0], infr.inputs[0])
    nt.links.new(fr[0][1], infr.inputs[1])
    for x in (fr[1][0], fr[1][1]):
        m2 = nt.nodes.new("ShaderNodeMath")
        m2.operation = "MINIMUM"
        nt.links.new(infr.outputs[0], m2.inputs[0])
        nt.links.new(x, m2.inputs[1])
        infr = m2

    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = tuple(float(x) for x in d)
    nt.links.new(geo.outputs["Normal"], dot.inputs[0])
    mx0 = nt.nodes.new("ShaderNodeMath")
    mx0.operation = "MAXIMUM"
    mx0.inputs[1].default_value = 0.0
    nt.links.new(dot.outputs["Value"], mx0.inputs[0])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = K
    nt.links.new(mx0.outputs[0], pw.inputs[0])
    w = nt.nodes.new("ShaderNodeMath")
    w.operation = "MULTIPLY"
    nt.links.new(pw.outputs[0], w.inputs[0])
    nt.links.new(tex.outputs["Alpha"], w.inputs[1])
    wf = nt.nodes.new("ShaderNodeMath")
    wf.operation = "MULTIPLY"
    nt.links.new(w.outputs[0], wf.inputs[0])
    nt.links.new(infr.outputs[0], wf.inputs[1])
    # 生成后提指数由 cfg 的 exponent=15 完成（`weight_exponent_after_generation`）。
    # 不再额外 POWER：二次放大让权重下溢到 0，bake 会全黑（实测 mean 0.006）。
    # 也不用 ShaderNodeAttribute 遮挡 —— Cycles bake 在本机读不到自定义
    # POINT/CORNER attribute（多轮带对照组探针证实）。

    sc = nt.nodes.new("ShaderNodeVectorMath")
    sc.operation = "SCALE"
    nt.links.new(tex.outputs["Color"], sc.inputs[0])
    try:
        nt.links.new(wf.outputs[0], sc.inputs["Scale"])
    except (KeyError, IndexError):
        nt.links.new(wf.outputs[0], sc.inputs[3])

    if acc is None:
        acc, wsum = sc.outputs["Vector"], wf.outputs[0]
    else:
        add = nt.nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        nt.links.new(acc, add.inputs[0])
        nt.links.new(sc.outputs["Vector"], add.inputs[1])
        acc = add.outputs["Vector"]
        aw = nt.nodes.new("ShaderNodeMath")
        aw.operation = "ADD"
        nt.links.new(wsum, aw.inputs[0])
        nt.links.new(wf.outputs[0], aw.inputs[1])
        wsum = aw.outputs[0]

inv = nt.nodes.new("ShaderNodeMath")
inv.operation = "DIVIDE"
inv.inputs[0].default_value = 1.0
nt.links.new(wsum, inv.inputs[1])
div = nt.nodes.new("ShaderNodeVectorMath")
div.operation = "SCALE"
nt.links.new(acc, div.inputs[0])
try:
    nt.links.new(inv.outputs[0], div.inputs["Scale"])
except (KeyError, IndexError):
    nt.links.new(inv.outputs[0], div.inputs[3])
# Vector -> Color 必须显式走 MixRGB：bake 的 EMIT pass 不会做隐式 socket 转换，
# 直连 Vector 会烘出全黑（实测 atlas mean=0.0）。ctx 渲染版正好有这道 MixRGB。
as_color = nt.nodes.new("ShaderNodeMixRGB")
as_color.blend_type = "MIX"
as_color.inputs[0].default_value = 0.0
as_color.inputs[2].default_value = (0.0, 0.0, 0.0, 1.0)
nt.links.new(div.outputs["Vector"], as_color.inputs[1])
nt.links.new(as_color.outputs["Color"], em.inputs["Color"])
mesh.materials.clear()
mesh.materials.append(mat)

print(f"PROJECT uv range u=[{uv_all[:, 0].min():.4f},{uv_all[:, 0].max():.4f}] "
      f"v=[{uv_all[:, 1].min():.4f},{uv_all[:, 1].max():.4f}] "
      f"active={bakeuv.name} layers={len(mesh.uv_layers)}",
      flush=True)
mesh.update()

# 分级探针（SG_PROBE=attr|tex|weight）：定位 bake 里哪一级变 0
_probe = CFG.get("probe")
if _probe == "constant":
    for _l in list(em.inputs["Color"].links):
        nt.links.remove(_l)
    em.inputs["Color"].default_value = (0.8, 0.2, 0.2, 1.0)
    print("PROJECT probe=constant", flush=True)
elif _probe and first_attr is not None:
    for _l in list(as_color.inputs[1].links):
        nt.links.remove(_l)
    if _probe == "attr":
        _sock = first_attr.outputs.get("UV") or first_attr.outputs.get("Vector")
        nt.links.new(_sock, as_color.inputs[1])
    elif _probe == "tex":
        nt.links.new(first_tex.outputs["Color"], as_color.inputs[1])
    print(f"PROJECT probe={_probe}", flush=True)

# --- 4) 一次 bake ---
image = bpy.data.images.new("SGAtlas", width=SIZE, height=SIZE,
                            alpha=False, float_buffer=False)
target_node = nt.nodes.new("ShaderNodeTexImage")
target_node.image = image
nt.nodes.active = target_node
target_node.select = True

scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.bake.margin = MARGIN
scene.render.bake.use_clear = True
scene.view_settings.view_transform = "Standard"
scene.display_settings.display_device = "sRGB"
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
res = bpy.ops.object.bake(type="EMIT", uv_layer=bakeuv.name)
if "FINISHED" not in res:
    raise RuntimeError(f"bake 返回 {set(res)}")
_px = np.empty(SIZE * SIZE * 4, np.float32)
image.pixels.foreach_get(_px)
print(f"PROJECT bake mean={float(_px.reshape(-1, 4)[:, :3].mean()):.4f} "
      f"nonzero={float((_px.reshape(-1, 4)[:, :3].max(1) > 0.01).mean()):.1%}",
      flush=True)
image.filepath_raw = CFG["atlas"]
image.file_format = "PNG"
image.save()
print("PROJECT atlas ->", CFG["atlas"], flush=True)

# --- 5) 导出最终 GLB（BakeUV + atlas）---
mat2 = bpy.data.materials.new("BakedAtlas")
mat2.use_nodes = True
n2 = mat2.node_tree
bsdf = n2.nodes.get("Principled BSDF")
t2 = n2.nodes.new("ShaderNodeTexImage")
t2.image = image
n2.links.new(t2.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.85
bsdf.inputs["Metallic"].default_value = 0.0
mesh.materials.clear()
mesh.materials.append(mat2)
bpy.ops.export_scene.gltf(filepath=CFG["out"], export_format="GLB",
                          export_texcoords=True, export_normals=True,
                          export_materials="EXPORT")
print("PROJECT out ->", CFG["out"], flush=True)
