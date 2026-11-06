"""StableGen 式 ctx render：**多相机投影材质**直接渲，不经 UV 图集。

    blender.exe -b -P tools/blender_context_projection.py -- <config.json>

config:
    {"mesh": "bake/baked_uv.glb",          # 已归一化的网格
     "ortho": 2.16, "size": 1024,
     "views": [{"name":"v00","image":"ready/v00.png","azimuth":0,"elevation":0}, ...],
     "camera": {"name":"v03","azimuth":135,"elevation":0},
     "out_dir": ".../ctx"}

为什么不用图集采样（`blender_context_views.py`）：
    StableGen `_export_emit_image_impl`（rendering.py:940-1010）把
    `project_image` 建的投影材质 color 输出直接接 Emission 渲出来。那个材质是
    「每台相机一段投影 UV + TexImage + 权重，全部求和÷权重和」。它渲的是
    **各相机原始生成图的实时 3D 表面混合**，没有经过
    「烘进 UV 图集 → 再从图集采样」这一步，所以不存在二次重采样损失。
    从图集渲 ctx 再投回图集，每轮都会累积误差（实测头发花白、腿部白块）。

投影 UV 照 `projection.py:360 _write_projection_uvw`：为**正交**相机把每个
corner 的 (u,v) 存成 CORNER 域 FLOAT2 attribute，材质里用 ShaderNodeAttribute
读取。权重与 `create_native_raycast_visibility` 同构：`pow(max(dot(N,camDir),0), k)`
× 生成图 alpha；所有相机求和后除以权重和；权重≈0 的着色点输出品红。
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
CFG = json.load(open(argv[0], encoding="utf-8"))

ORTHO = float(CFG.get("ortho", 2.16))
SIZE = int(CFG.get("size", 1024))
K = float(CFG.get("exponent", 3.0))
OUT = CFG["out_dir"]
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=CFG["mesh"])
scene = bpy.context.scene

meshes = [o for o in scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("网格里没有 mesh")
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
mesh = obj.data

# 与 blender_bake_views.py 一致的归一化（baked_uv.glb 已归一化，这一步幂等）
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


def cam_basis(az_deg: float, el_deg: float):
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = Vector((math.sin(az) * math.cos(el),
                -math.cos(az) * math.cos(el),
                math.sin(el)))
    loc = d * 3.5
    rot = (Vector((0, 0, 0)) - loc).to_track_quat("-Z", "Y").to_matrix()
    return loc, d, rot @ Vector((1, 0, 0)), rot @ Vector((0, 1, 0))


# --- 为每台相机写 CORNER 域投影 UV（源码 :360 的正交分支）---
VIEWS = CFG["views"]
for v in VIEWS:
    loc, d, right, up = cam_basis(float(v.get("azimuth", 0)),
                                  float(v.get("elevation", 0)))
    attr_name = "sg_uv_" + v["name"]
    old = mesh.attributes.get(attr_name)
    if old:
        mesh.attributes.remove(old)
    # POINT 域（不是 CORNER）：正交投影 UV 是顶点坐标的仿射函数，逐顶点存 +
    # 线性插值与逐 corner 等价；且 Cycles **bake** 读 CORNER FLOAT2 attribute
    # 会返回 0，POINT 才可用。与 blender_project_all.py 保持同源。
    attr = mesh.attributes.new(attr_name, "FLOAT_VECTOR", "POINT")
    import numpy as np
    _V = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", _V)
    _V = _V.reshape(-1, 3).astype(np.float64) - np.array(loc, np.float64)
    _r = np.array(right, np.float64)
    _u = np.array(up, np.float64)
    # 同 blender_bake_views.py 的 ProjUV：**不取负**（取负会让图上下翻转，
    # 实测脸消失、手臂被腿的颜色覆盖）。
    _uv3 = np.zeros((len(_V), 3), np.float32)
    _uv3[:, 0] = 0.5 + _V @ _r / ORTHO
    _uv3[:, 1] = 0.5 + _V @ _u / ORTHO
    attr.data.foreach_set("vector", _uv3.ravel())

# --- 材质：Σ(color_i · w_i) / Σw_i，w≈0 时输出品红 ---
mat = bpy.data.materials.new("sg_ctx_projection")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
em = nt.nodes.new("ShaderNodeEmission")
nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
geo = nt.nodes.new("ShaderNodeNewGeometry")

acc_node = None
wsum_node = None

for v in VIEWS:
    loc, d, right, up = cam_basis(float(v.get("azimuth", 0)),
                                  float(v.get("elevation", 0)))
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = "sg_uv_" + v["name"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(v["image"])
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    nt.links.new(a.outputs["Vector"], tex.inputs["Vector"])

    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = tuple(float(x) for x in d)
    nt.links.new(geo.outputs["Normal"], dot.inputs[0])
    mx = nt.nodes.new("ShaderNodeMath")
    mx.operation = "MAXIMUM"
    mx.inputs[1].default_value = 0.0
    nt.links.new(dot.outputs["Value"], mx.inputs[0])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = K
    nt.links.new(mx.outputs[0], pw.inputs[0])
    w = nt.nodes.new("ShaderNodeMath")
    w.operation = "MULTIPLY"
    nt.links.new(pw.outputs[0], w.inputs[0])
    nt.links.new(tex.outputs["Alpha"], w.inputs[1])

    scale = nt.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    nt.links.new(tex.outputs["Color"], scale.inputs[0])
    try:
        nt.links.new(w.outputs[0], scale.inputs["Scale"])
    except (KeyError, IndexError):
        nt.links.new(w.outputs[0], scale.inputs[3])

    if acc_node is None:
        acc_node = scale.outputs["Vector"]
        wsum_node = w.outputs[0]
    else:
        add = nt.nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        nt.links.new(acc_node, add.inputs[0])
        nt.links.new(scale.outputs["Vector"], add.inputs[1])
        acc_node = add.outputs["Vector"]
        addw = nt.nodes.new("ShaderNodeMath")
        addw.operation = "ADD"
        nt.links.new(wsum_node, addw.inputs[0])
        nt.links.new(w.outputs[0], addw.inputs[1])
        wsum_node = addw.outputs[0]

# 除以权重和（用 VectorMath SCALE + Math 的倒数）
inv = nt.nodes.new("ShaderNodeMath")
inv.operation = "DIVIDE"
inv.inputs[0].default_value = 1.0
nt.links.new(wsum_node, inv.inputs[1])
div = nt.nodes.new("ShaderNodeVectorMath")
div.operation = "SCALE"
nt.links.new(acc_node, div.inputs[0])
try:
    nt.links.new(inv.outputs[0], div.inputs["Scale"])
except (KeyError, IndexError):
    nt.links.new(inv.outputs[0], div.inputs[3])

# 无覆盖 → 品红；fac = (wsum < eps)
cmp_node = nt.nodes.new("ShaderNodeMath")
cmp_node.operation = "LESS_THAN"
nt.links.new(wsum_node, cmp_node.inputs[0])
cmp_node.inputs[1].default_value = 1e-4
mix = nt.nodes.new("ShaderNodeMixRGB")
mix.blend_type = "MIX"
mix.inputs[2].default_value = (1.0, 0.0, 1.0, 1.0)
nt.links.new(cmp_node.outputs[0], mix.inputs[0])
nt.links.new(div.outputs["Vector"], mix.inputs[1])
nt.links.new(mix.outputs["Color"], em.inputs["Color"])

mesh.materials.clear()
mesh.materials.append(mat)

# 剪影 alpha 用的纯白材质
mat_mask = bpy.data.materials.new("sg_ctx_mask")
mat_mask.use_nodes = True
_m = mat_mask.node_tree
_m.nodes.clear()
_mo = _m.nodes.new("ShaderNodeOutputMaterial")
_me = _m.nodes.new("ShaderNodeEmission")
_me.inputs["Color"].default_value = (1, 1, 1, 1)
_m.links.new(_me.outputs["Emission"], _mo.inputs["Surface"])

world = scene.world or bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
    bg.inputs[1].default_value = 1.0

scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.resolution_x = scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.use_compositing = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

cam = CFG["camera"]
loc, d, right, up = cam_basis(float(cam.get("azimuth", 0)),
                              float(cam.get("elevation", 0)))
cd = bpy.data.cameras.new("ctx_cam")
cd.type = "ORTHO"
cd.ortho_scale = ORTHO
cd.clip_start = 0.01
cd.clip_end = 100.0
co = bpy.data.objects.new("ctx_cam", cd)
scene.collection.objects.link(co)
co.location = loc
co.rotation_euler = (Vector((0, 0, 0)) - loc).to_track_quat("-Z", "Y").to_euler()
scene.camera = co
bpy.context.view_layer.update()

name = cam["name"]
beauty = os.path.join(OUT, f"{name}_beauty.png")
alpha = os.path.join(OUT, f"{name}_alpha.png")
scene.render.film_transparent = False
scene.render.filepath = beauty
bpy.ops.render.render(write_still=True)
print(f"CTXPROJ {name} -> {beauty}", flush=True)

mesh.materials[0] = mat_mask
scene.render.film_transparent = True
scene.render.image_settings.color_mode = "RGBA"
scene.render.filepath = alpha
bpy.ops.render.render(write_still=True)
print(f"CTXPROJ alpha -> {alpha}", flush=True)
print("CTXPROJ DONE", flush=True)
