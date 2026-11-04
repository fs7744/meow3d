"""Blender 内运行：从指定机位渲出「几何指引图」，供 Qwen 生成严格对齐的新视角。

    blender.exe -b -P tools/blender_guidance.py -- <config.json>

为什么要有这一步：投影管线的多相机版本坏在「Qwen 生成的侧视图几何不对」——
`view_02` 号称 90°，实际还偏正面，投影到窄侧视 bbox 就被横向拉伸成脏斑。
给 Qwen 一张**同一台相机渲出来的结构图**，它就不用猜姿态和构图了。

每个机位输出 4 张（PNG，尺寸 = size × size，与 bake 的 ORTHO 完全同框）：

  <key>_depth.png   归一化深度，**近=白 远=黑**，背景黑（ControlNet depth 的惯例）
  <key>_normal.png  世界空间法线编码成 RGB（n*0.5+0.5）
  <key>_clay.png    素模渲染（EEVEE，studio 光 + cavity），做 lineart/canny 的源
  <key>_beauty.png  带原贴图的渲染（EEVEE），当「Image Edit 的底图」用
  <key>_flat.png    baseColor 直出（Emission，无光照），当颜色锚定的**可信基准**用

⚠️ beauty 与 flat 不能混用：beauty 带灯光和高光，把它当颜色基准写进贴图，
渲染时又被打一次光 → 整体发灰发亮。StableGen 的 `export_emit_image` 也是
用 Emission 渲上下文，不是带灯光的 beauty。

同框的含义：相机固定为 `ortho`（由 cfg 传入，默认 2.8；`--fill-frame` 会收紧）、
距离 3.5、看向原点，模型归一化到 [-1,1]，
于是渲染图里模型占的像素范围与 `bake.projection_bbox()` 算出的 [0,1]² 框**逐像素
一致**。宿主侧把它当指引图交给 Qwen，Qwen 按同框输出，再走 `make_aligned_view`
的裁剪缩放就是近似恒等变换，不会二次变形。

深度/法线用 CYCLES 的 pass（EEVEE 没有 Normal pass），素模/彩图用 EEVEE Next
（光栅化，1 采样就干净）。
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
CFG = json.load(open(argv[0], encoding="utf-8"))

SIZE = int(CFG.get("size", 1024))
ORTHO = float(CFG.get("ortho", 2.8))
CAM_DIST = 3.5
DEPTH_NEAR = 2.0                    # 与 blender_bake_views.py 同一套归一化区间
DEPTH_FAR = 5.0
WORK = os.path.abspath(CFG["work_dir"])
PASSES = list(CFG.get("passes",
                      ["depth", "normal", "clay", "beauty", "flat"]))
os.makedirs(WORK, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.abspath(CFG["mesh"]))

scene = bpy.context.scene
meshes = [o for o in scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("GLB 里没有 mesh")

# --- 归一化到 [-1,1]：与 blender_bake_views.py / blender_preview.py 逐字一致 ---
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
mn = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
mx = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
center = (mn + mx) / 2
scale = 2.0 / max(mx - mn)
for o in meshes:
    o.matrix_world = Matrix.Scale(scale, 4) @ Matrix.Translation(-center) @ o.matrix_world

scene.render.resolution_x = scene.render.resolution_y = SIZE
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
try:
    scene.view_settings.view_transform = "Standard"
except TypeError:
    pass

# --- 相机（全部正交、看向原点；elevation 由方位角公式带出，与 bake 一致）---
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = ORTHO
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# --- 灯光（只给 beauty 用；能量压住，白铠甲不至于过曝成一片白）---
LIGHTS = [((3, -4, 5), 520), ((-4, -2, 3), 330), ((0, 4, 2), 260), ((0, -5, 1.5), 430)]
for i, (pos, energy) in enumerate(LIGHTS):
    ld = bpy.data.lights.new(f"light{i}", "AREA")
    ld.energy, ld.size = energy, 4.0
    lo = bpy.data.objects.new(f"light{i}", ld)
    scene.collection.objects.link(lo)
    lo.location = pos
    lo.rotation_euler = (Vector((0, 0, 0)) - lo.location).to_track_quat("-Z", "Y").to_euler()

world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
bg.inputs[0].default_value = (0.30, 0.31, 0.35, 1)
bg.inputs[1].default_value = 1.0


def set_engine(name):
    try:
        scene.render.engine = name
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"


def cycles_passes():
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    vl = bpy.context.view_layer
    vl.use_pass_z = True
    vl.use_pass_normal = True


def comp_begin():
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    comp = nt.nodes.new("CompositorNodeComposite")
    return nt, rl, comp


def render_depth(path):
    """近=白、远=黑。背景（Z 无交点）在 Cycles 里是极大值，MapRange 截到 0。"""
    nt, rl, comp = comp_begin()
    mr = nt.nodes.new("CompositorNodeMapRange")
    mr.use_clamp = True
    mr.inputs["From Min"].default_value = DEPTH_NEAR
    mr.inputs["From Max"].default_value = DEPTH_FAR
    mr.inputs["To Min"].default_value = 1.0        # 近 → 白
    mr.inputs["To Max"].default_value = 0.0        # 远 → 黑
    nt.links.new(rl.outputs["Depth"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Value"], comp.inputs["Image"])
    img_set = scene.render.image_settings
    img_set.color_mode = "BW"
    img_set.color_depth = "8"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img_set.color_mode = "RGBA"
    print("GUIDE depth", path, flush=True)


def render_normal(path):
    nt, rl, comp = comp_begin()
    # Cycles 的 Normal pass 已经是 (n*0.5+0.5)，直接落盘即可
    nt.links.new(rl.outputs["Normal"], comp.inputs["Image"])
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("GUIDE normal", path, flush=True)


def render_clay(path):
    """素模：workbench 单一灰色 + cavity，让沟槽/边缘显出来。"""
    # **必须关掉合成器**：前面渲 depth/normal 时挂的 RLayers 节点树还在，
    # workbench 下它没有 Normal/Z 输出 → 整张渲成黑的（实测踩过）。
    scene.use_nodes = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.78, 0.78, 0.80)
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
    except TypeError:
        pass
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("GUIDE clay", path, flush=True)


def flat_material(src, name):
    """用原材质的 baseColorTexture 建一个纯 Emission 材质（无灯光、无高光）。"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    img = None
    if src is not None and src.use_nodes:
        # GLB 导入的 Principled BSDF 只有一张 baseColorTexture 接到 Base Color
        for n in src.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None:
                img = n.image
                break
    if img is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        uvn = nt.nodes.new("ShaderNodeUVMap")
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    return m


def render_flat(path):
    """baseColor 直出：StableGen `export_emit_image` 的等价物。

    颜色锚定必须拿它当基准，而不是带灯光的 beauty。渲完立刻还原材质，
    否则下一个机位的 beauty 也会变成无光照的。
    """
    scene.use_nodes = False
    set_engine("BLENDER_EEVEE_NEXT")
    saved, cache = [], {}
    for obj in meshes:
        for slot in obj.material_slots:
            src = slot.material
            saved.append((slot, src))
            key = src.name if src else "__none__"
            if key not in cache:
                cache[key] = flat_material(src, f"flat_{key}")
            slot.material = cache[key]
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = path
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        for slot, src in saved:
            slot.material = src
    print("GUIDE flat", path, flush=True)


for c in CFG["cameras"]:
    key = c["name"]
    az = math.radians(float(c.get("azimuth", 0)))
    el = math.radians(float(c.get("elevation", 0)))
    d = Vector((math.sin(az) * math.cos(el),
                -math.cos(az) * math.cos(el),
                math.sin(el)))
    cam.location = d * CAM_DIST
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    if "depth" in PASSES or "normal" in PASSES:
        cycles_passes()
        if "depth" in PASSES:
            render_depth(os.path.join(WORK, f"{key}_depth.png"))
        if "normal" in PASSES:
            render_normal(os.path.join(WORK, f"{key}_normal.png"))

    # 素模用 workbench（很快），彩图用 EEVEE
    if "clay" in PASSES:
        render_clay(os.path.join(WORK, f"{key}_clay.png"))
    if "beauty" in PASSES:
        scene.use_nodes = False        # 同上：别让合成器残留把彩图换成法线图
        set_engine("BLENDER_EEVEE_NEXT")
        scene.render.image_settings.color_mode = "RGB"
        scene.render.filepath = os.path.join(WORK, f"{key}_beauty.png")
        bpy.ops.render.render(write_still=True)
        print("GUIDE beauty", scene.render.filepath, flush=True)
    if "flat" in PASSES:
        render_flat(os.path.join(WORK, f"{key}_flat.png"))

print("GUIDE ALL DONE", flush=True)
