"""Blender 内运行：把合并好的 atlas 贴到 BakeUV 网格并导出 GLB。

    blender.exe -b -P tools/blender_apply_atlas.py -- <config.json>
"""
import json
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
cfg = json.load(open(argv[0], encoding="utf-8"))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=cfg["mesh"])

objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not objs:
    raise RuntimeError("中间 GLB 里没有 mesh")
bpy.ops.object.select_all(action="DESELECT")
for o in objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]
if len(objs) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

for uv in obj.data.uv_layers:
    uv.active_render = False
obj.data.uv_layers.active.active_render = True

img = bpy.data.images.load(cfg["atlas"])
mat = bpy.data.materials.new("BakedAtlas")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = img
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
if cfg.get("roughness", 0.85) is not None:
    bsdf.inputs["Roughness"].default_value = float(cfg.get("roughness", 0.85))
bsdf.inputs["Metallic"].default_value = 0.0

obj.data.materials.clear()
obj.data.materials.append(mat)

print("APPLY uv layers:", [u.name for u in obj.data.uv_layers], flush=True)
bpy.ops.export_scene.gltf(filepath=cfg["out"], export_format="GLB",
                          export_texcoords=True, export_normals=True,
                          export_materials="EXPORT")
print("APPLY saved", cfg["out"], flush=True)
