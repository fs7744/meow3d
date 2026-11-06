"""用 Blender 自带的 select_overlap 权威判定 UV 是否重叠。

    blender.exe -b -P tools/_uv_overlap2.py -- <mesh.glb>
"""
import sys

import bpy

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]
if len(objs) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
me = obj.data

print(f"UV layers: {[l.name for l in me.uv_layers]}  faces={len(me.polygons)}", flush=True)

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.select_mode(type="FACE")
bpy.ops.uv.select_overlap()
bpy.ops.object.mode_set(mode="OBJECT")

sel = sum(1 for p in me.polygons if p.select)
print(f"OVERLAP faces={sel} / {len(me.polygons)} = {sel / len(me.polygons):.2%}", flush=True)
