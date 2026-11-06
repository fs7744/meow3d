"""在真实白模上直接测 Raycast：Is Hit / Hit Distance / dist 三张图。

    blender.exe -b -P tools/_probe_real.py -- <mesh.glb>
"""
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
path = argv[0]
S = 256

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.render.bake.margin = 0

meshes = [o for o in scene.objects if o.type == "MESH"]
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# 与 blender_bake_views.py 相同的归一化
pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
mx = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
center = (mn + mx) / 2
scale = 2.0 / max(mx - mn)
obj.matrix_world = Matrix.Scale(scale, 4) @ Matrix.Translation(-center) @ obj.matrix_world
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = obj.data

# 展开 UV
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
bpy.ops.object.mode_set(mode="OBJECT")

# 相机 az=0：从 -Y 看向原点
az = 0.0
d = Vector((math.sin(az), -math.cos(az), 0.0))
cam_data = bpy.data.cameras.new("c")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.8
cam = bpy.data.objects.new("c", cam_data)
scene.collection.objects.link(cam)
cam.location = d * 3.5
cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
scene.camera = cam
loc = cam.location.copy()
print(f"PROBE cam_loc={tuple(round(v,3) for v in loc)} verts={len(me.vertices)} "
      f"faces={len(me.polygons)}", flush=True)


def make_mat(channel):
    m = bpy.data.materials.new("m_" + channel)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    cxyz = nt.nodes.new("ShaderNodeCombineXYZ")
    for i in range(3):
        cxyz.inputs[i].default_value = float(loc[i])
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    nt.links.new(geo.outputs["Position"], sub.inputs[0])
    nt.links.new(cxyz.outputs["Vector"], sub.inputs[1])
    dist = nt.nodes.new("ShaderNodeVectorMath")
    dist.operation = "LENGTH"
    nt.links.new(sub.outputs["Vector"], dist.inputs[0])
    ndir = nt.nodes.new("ShaderNodeVectorMath")
    ndir.operation = "NORMALIZE"
    nt.links.new(sub.outputs["Vector"], ndir.inputs[0])

    ray = nt.nodes.new("ShaderNodeRaycast")
    nt.links.new(cxyz.outputs["Vector"], ray.inputs["Position"])
    nt.links.new(ndir.outputs["Vector"], ray.inputs["Direction"])
    ln = nt.nodes.new("ShaderNodeMath")
    ln.operation = "MULTIPLY_ADD"
    ln.inputs[1].default_value = 1.0
    ln.inputs[2].default_value = 2.6e-4
    nt.links.new(dist.outputs["Value"], ln.inputs[0])
    nt.links.new(ln.outputs[0], ray.inputs["Length"])

    if channel == "hit":
        nt.links.new(ray.outputs["Is Hit"], em.inputs["Color"])
    elif channel == "hitdist":
        nt.links.new(ray.outputs["Hit Distance"], em.inputs["Color"])
    elif channel == "dist":
        sc = nt.nodes.new("ShaderNodeMath")
        sc.operation = "MULTIPLY"
        sc.inputs[1].default_value = 0.2   # 3.5 距离 → 0.7，落在 0~1
        nt.links.new(dist.outputs["Value"], sc.inputs[0])
        nt.links.new(sc.outputs[0], em.inputs["Color"])

    t = nt.nodes.new("ShaderNodeTexImage")
    t.name = "BakeTarget"
    return m, nt, t


for channel in ("hit", "hitdist", "dist"):
    m, nt, tgt = make_mat(channel)
    me.materials.clear()
    me.materials.append(m)
    img = bpy.data.images.new(f"b_{channel}", S, S, alpha=True, float_buffer=True)
    tgt.image = img
    nt.nodes.active = tgt
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    res = bpy.ops.object.bake(type="EMIT", width=S, height=S)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)[:, 0]
    if channel == "hit":
        print(f"PROBE hit   mean={px.mean():.3f}  >0.5 {(px > 0.5).mean():.3f}", flush=True)
    elif channel == "hitdist":
        hitmask = px > 0.005
        print(f"PROBE hitdist mean={px.mean():.3f} max={px.max():.3f} "
              f"frac>len*0.99 {(px > 3.0).mean():.3f}", flush=True)
    else:
        m2 = px[px > 0.01]
        print(f"PROBE dist  mean={px.mean():.3f} min={m2.min() if m2.size else 0:.3f} "
              f"max={px.max():.3f}  (÷0.2)", flush=True)
print("PROBE DONE", flush=True)
