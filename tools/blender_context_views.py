"""渲「当前贴图 + 未覆盖区品红」的 context render（StableGen 顺序生成的输入）。

    blender.exe -b -P tools/blender_context_views.py -- <config.json>

config:
    {"mesh": "bake/baked_uv.glb",      # 已归一化、UV=BakeUV 的网格
     "atlas": "atlas_magenta.png",     # 未覆盖 texel 已涂品红的图集
     "ortho": 2.16, "size": 1024,
     "views": [{"name":"v00","azimuth":0,"elevation":0}, ...],
     "out_dir": ".../ctx"}

产出 `out_dir/v{name}_beauty.png`：与 bake 同一套相机，可直接喂给
`qwen_guided_views.py --guide <out_dir> --guide-pass beauty`。

对应 StableGen `rendering.py:1017` 的 `ctx_render{N}`：把当前投影结果按
Emission 渲出来，未投影面的 fallback 色设为品红，让扩散模型只补品红区域。
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

# 与 blender_bake_views.py 完全一致的归一化（baked_uv.glb 已是归一化后的，
# 这里再走一遍是幂等的，只为保证相机约定一字不差）
pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
mx = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
if (mx - mn).length > 1e-6:
    center = (mn + mx) / 2
    scale = 2.0 / max(mx - mn)
    obj.matrix_world = (Matrix.Scale(scale, 4)
                        @ Matrix.Translation(-center)
                        @ obj.matrix_world)
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# UV：baked_uv.glb 的 uv0 就是 BakeUV
uv_layer = mesh.uv_layers.active
if uv_layer is None:
    raise RuntimeError("网格没有 UV；context render 需要 bake 产出的 baked_uv.glb")

# Emission 材质：直接显示图集原色（不要灯光，避免二次打光）
mat = bpy.data.materials.new("ctx")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
em = nt.nodes.new("ShaderNodeEmission")
nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(CFG["atlas"])
tex.extension = "CLIP"
tex.interpolation = "Closest"
uvn = nt.nodes.new("ShaderNodeUVMap")
uvn.uv_map = uv_layer.name
nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
nt.links.new(tex.outputs["Color"], em.inputs["Color"])
mesh.materials.clear()
mesh.materials.append(mat)

# 世界背景：中灰（StableGen 的提示词要求「Replace the background with solid gray」）
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
scene.render.film_transparent = False
scene.render.use_compositing = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
# 必须 Standard：图集 PNG 是 sRGB 编码的，贴进 Emission 再经默认 AgX 输出会
# 整体降饱和/偏灰 → 模型看到的是失真的上下文，投回来的颜色也跟着偏。
# Standard + sRGB 贴图 + PNG 输出 = 逐像素恒等往返。
scene.view_settings.view_transform = "Standard"
try:
    scene.view_settings.look = "None"
except TypeError:      # 5.x 把 look 换成枚举字符串，值名可能不同
    pass
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

# 轮廓 alpha：同一相机、同一网格、film_transparent 渲一遍纯白 → 像素级等于
# 网格剪影。顺序生成用它当投影响应函数，生成图与网格因此天然逐像素对齐
# （用 birefnet 抠图的话，Qwen 画得比网格窄/宽就会露出空洞或背景）。
mat_mask = bpy.data.materials.new("ctx_mask")
mat_mask.use_nodes = True
_nt = mat_mask.node_tree
_nt.nodes.clear()
_out = _nt.nodes.new("ShaderNodeOutputMaterial")
_em = _nt.nodes.new("ShaderNodeEmission")
_em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
_nt.links.new(_em.outputs["Emission"], _out.inputs["Surface"])

for v in CFG["views"]:
    az = math.radians(float(v.get("azimuth", 0.0)))
    el = math.radians(float(v.get("elevation", 0.0)))
    d = Vector((math.sin(az) * math.cos(el),
                -math.cos(az) * math.cos(el),
                math.sin(el)))
    cam_data = bpy.data.cameras.new("c_" + v["name"])
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ORTHO
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam = bpy.data.objects.new("c_" + v["name"], cam_data)
    scene.collection.objects.link(cam)
    cam.location = d * 3.5
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    bpy.context.view_layer.update()

    path = os.path.join(OUT, f"v{v['name'][1:]}_beauty.png" if v["name"].startswith("v")
                        else f"{v['name']}_beauty.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"CTX {v['name']} -> {path}", flush=True)

    # 剪影 alpha（纯白 Emission + 透明胶片）
    mesh.materials[0] = mat_mask
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    apath = path[:-len("_beauty.png")] + "_alpha.png"
    scene.render.filepath = apath
    bpy.ops.render.render(write_still=True)
    print(f"ALPHA {v['name']} -> {apath}", flush=True)
    mesh.materials[0] = mat
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGB"

    bpy.data.objects.remove(cam, do_unlink=True)

print("CTX DONE", flush=True)
