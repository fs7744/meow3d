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
# 遮挡剔除：不剔除的话，被挡住的背面会把颜色投到正面（蓝羽饰糊在银盔上）。
# 实测四相机时最明显，必须开。
OCCLUDE = bool(CFG.get("occlude", True))
DEPTH_NEAR = 2.0            # 归一化相机在 3.5 处、模型在 [-1,1] → 深度落在 [2.1,4.9]
DEPTH_FAR = 5.0
DEPTH_EPS = 0.012           # 归一化深度容差（×3.0m ≈ 3.6cm），压厚度可调小
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

# --- UV 策略 ---
# bake 是「逐三角面正向光栅化」，碎片 UV 岛完全无害；真正致命的是 **UV 重叠**
# （Cube Project 会把朝向相近的多个面投到同一区域，互相覆盖 → 表面斑驳）。
# 因此默认保留输入网格已有的不重叠 UV（TRELLIS.2 原生链用 xatlas 展开过）。
UNWRAP = CFG.get("unwrap", "keep")
have_uv = len(mesh.uv_layers) > 0
if UNWRAP == "auto":
    UNWRAP = "keep" if have_uv else "smart"
if UNWRAP == "keep":
    if not have_uv:
        raise RuntimeError("unwrap=keep 但网格没有 UV；改用 --unwrap smart")
    bakeuv = mesh.uv_layers.active
    bakeuv.name = "BakeUV"
    print("BAKE unwrap: 保留原始不重叠 UV -> BakeUV", flush=True)
else:
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if UNWRAP == "cube":
        bpy.ops.uv.cube_project(cube_size=2.0, scale_to_bounds=True,
                                correct_aspect=True)
    else:
        bpy.ops.uv.smart_project(island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")
    bakeuv = mesh.uv_layers.active
    bakeuv.name = "BakeUV"
    print(f"BAKE unwrap: {UNWRAP} -> BakeUV", flush=True)
mesh.uv_layers.active = bakeuv
bakeuv.active_render = True

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
              cam_loc=None, zdepth=None):
    """mode=color: Emission(视图颜色)；mode=weight: Emission(alpha * cos^k * 可见性)。"""
    mat = bpy.data.materials.new(f"{mode}_{key}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])

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
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = tuple(float(x) for x in cam_dir)
        nt.links.new(geo.outputs["Normal"], dot.inputs[0])
        mx = nt.nodes.new("ShaderNodeMath")
        mx.operation = "MAXIMUM"
        mx.inputs[1].default_value = 0.0
        nt.links.new(dot.outputs["Value"], mx.inputs[0])
        pw = nt.nodes.new("ShaderNodeMath")
        pw.operation = "POWER"
        pw.inputs[1].default_value = EXP
        nt.links.new(mx.outputs[0], pw.inputs[0])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(pw.outputs[0], mul.inputs[0])
        nt.links.new(tex.outputs["Alpha"], mul.inputs[1])
        w_out = mul.outputs[0]

        if zdepth is not None:
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

        # float -> Color 会广播成 (w,w,w,1)，不必依赖各版本 CombineColor 的输入名
        nt.links.new(w_out, em.inputs["Color"])

    tgt = nt.nodes.new("ShaderNodeTexImage")
    tgt.image = target
    tgt.select = True
    nt.nodes.active = tgt
    return mat


def render_zdepth(path):
    """用同一台正交相机渲一张归一化深度图（0=近 1=远），16 位 PNG。

    只当「前沿深度」用，不需要抗锯齿或色彩管理；Cycles 的 Z pass 是解析值，
    1 sample 就够。
    """
    scene.camera = cam
    vl = bpy.context.view_layer
    vl.use_pass_z = True
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
    wmat = build_mat(key, "weight", weight_img, proj.name, cam_dir=d,
                     cam_loc=cam.location,
                     zdepth=zdepth_path if OCCLUDE else None)
    weight = bake_once(wmat, weight_img)[..., 0].copy()

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
