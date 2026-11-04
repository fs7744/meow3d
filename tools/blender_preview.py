"""给 Blender 用的 GLB 多视角渲染脚本（由 tools/preview_glb.py 调用）。

    blender.exe -b -P tools/blender_preview.py -- <glb> <输出目录>
"""
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
# Blender -b 的 cwd 不是调用者的 cwd，相对路径会跑到 C:\，必须转绝对路径
glb, outdir = os.path.abspath(argv[0]), os.path.abspath(argv[1])
FLAT = "flat" in argv[2:]      # 只看贴图本色（绕过灯光），诊断颜色用
os.makedirs(outdir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)

scene = bpy.context.scene
meshes = [o for o in scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("GLB 里没有 mesh")

pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
mn = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
mx = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
center = (mn + mx) / 2
scale = 2.0 / max(mx - mn)
for o in meshes:
    o.matrix_world = Matrix.Scale(scale, 4) @ Matrix.Translation(-center) @ o.matrix_world

world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
bg.inputs[0].default_value = (0.30, 0.31, 0.35, 1)
bg.inputs[1].default_value = 1.0

# AgX 会把高光压灰、让皮肤发闷；展示贴图真实颜色用 Standard
try:
    scene.view_settings.view_transform = "Standard"
except TypeError:
    pass

if FLAT:
    # 贴图本色直出：把每个材质的 Base Color 接到 Emission，并且不打灯。
    # 判断「颜色对不对」必须这么看 —— 灯光下白色铠甲会把脸照成一片白。
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nt = mat.node_tree
        out_node = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out_node is None:
            continue
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        em = nt.nodes.new("ShaderNodeEmission")
        if bsdf is not None:
            src = bsdf.inputs["Base Color"]
            if src.links:
                nt.links.new(src.links[0].from_socket, em.inputs["Color"])
            else:
                em.inputs["Color"].default_value = src.default_value
        nt.links.new(em.outputs["Emission"], out_node.inputs["Surface"])

for i, (pos, energy) in enumerate([] if FLAT else [
    # 总能量别堆太高：白铠甲 + 金属度高，灯一强就整片死白，看起来「脸发白、
    # 细节没了」。这组值让白色铠甲落在 0.85 左右而不是 1.0。
    ((3, -4, 5), 520), ((-4, -2, 3), 330), ((0, 4, 2), 260),
    ((0, -5, 1.5), 430),    # 正面主光
]):
    ld = bpy.data.lights.new(f"light{i}", "AREA")
    ld.energy, ld.size = energy, 4.0
    lo = bpy.data.objects.new(f"light{i}", ld)
    scene.collection.objects.link(lo)
    lo.location = pos
    lo.rotation_euler = (Vector((0, 0, 0)) - lo.location).to_track_quat("-Z", "Y").to_euler()

cd = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cd)
scene.collection.objects.link(cam)
scene.camera = cam
cd.type = "ORTHO"
cd.ortho_scale = 2.5

try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 600, 800
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False

for name, az in [("front", 0), ("three_quarter", 40), ("side", 90), ("back", 180)]:
    a = math.radians(az)
    cam.location = (3.5 * math.sin(a), -3.5 * math.cos(a), 1.0)
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = os.path.join(outdir, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", scene.render.filepath)

# 头部特写：全身图里脸只有几十像素，判断「脸糊不糊」必须单独放大看。
# 归一化后身高占 [-1,1]，头在 z≈0.75~0.95，取 0.62 视野框住头盔。
cd.ortho_scale = 0.62
scene.render.resolution_x, scene.render.resolution_y = 700, 700
cam.location = (0, -3.5, 0.80)
cam.rotation_euler = (Vector((0, 0, 0.80)) - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.render.filepath = os.path.join(outdir, "head.png")
bpy.ops.render.render(write_still=True)
print("RENDERED", scene.render.filepath)
