"""给 Blender 用的 UV 展开脚本（由 tools/uv_unwrap.py 调用）。

    blender.exe -b -P tools/blender_uv_unwrap.py -- <输入.glb> <输出.glb> [降面目标]
"""
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
decimate = int(argv[2]) if len(argv) > 2 and int(argv[2]) > 0 else 0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not objs:
    raise RuntimeError("GLB 里没有 mesh")

bpy.ops.object.select_all(action="DESELECT")
for o in objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]
if len(objs) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

if decimate and len(obj.data.polygons) > decimate:
    mod = obj.modifiers.new("decimate", "DECIMATE")
    mod.ratio = decimate / len(obj.data.polygons)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"UV_UNWRAP decimated to {len(obj.data.polygons)} faces")

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(island_margin=0.004)
bpy.ops.object.mode_set(mode="OBJECT")
print(f"UV_UNWRAP done: {len(obj.data.polygons)} faces, "
      f"{len(obj.data.uv_layers.active.data) if obj.data.uv_layers.active else 0} loops")

bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB",
                          export_texcoords=True, export_normals=True,
                          export_materials="NONE")
print("UV_UNWRAP saved", dst)
