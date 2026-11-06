"""分阶段探针：在真实白模 + 真实 v00 视图上，逐级烘出权重的每个因子。

    blender.exe -b -P tools/_probe_stages.py -- <mesh.glb> <view.png> [ortho]

输出每级的 mean / >0.01 占比，用来定位「哪一项把权重乘成 0」。
"""
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
mesh_path, view_path = argv[0], argv[1]
ORTHO = float(argv[2]) if len(argv) > 2 else 2.8
S = 256
EXP = 7.0
COS_CUT = math.cos(math.radians(70.0))
EPS = 2.617e-05
FEATHER = 0.15

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=mesh_path)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.render.bake.margin = 4

meshes = [o for o in scene.objects if o.type == "MESH"]
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
mn = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
mx = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
center = (mn + mx) / 2
scale = 2.0 / max(mx - mn)
obj.matrix_world = Matrix.Scale(scale, 4) @ Matrix.Translation(-center) @ obj.matrix_world
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = obj.data

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
bpy.ops.object.mode_set(mode="OBJECT")
bakeuv = me.uv_layers.active
bakeuv.name = "BakeUV"
bakeuv.active_render = True
proj = me.uv_layers.new(name="ProjUV")

az, el = 0.0, 0.0
d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
cam_data = bpy.data.cameras.new("c")
cam_data.type = "ORTHO"
cam_data.ortho_scale = ORTHO
cam = bpy.data.objects.new("c", cam_data)
scene.collection.objects.link(cam)
cam.location = d * 3.5
cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
loc = cam.location.copy()
rot = cam.matrix_world.to_3x3()

# ProjUV
inv = np.array(cam.matrix_world.inverted(), dtype=np.float64)
vco = np.empty(len(me.vertices) * 3, np.float32)
me.vertices.foreach_get("co", vco)
V = vco.reshape(-1, 3).astype(np.float64)
P = V @ inv[:3, :3].T + inv[:3, 3]
u = 0.5 + P[:, 0] / ORTHO
w = 0.5 + P[:, 1] / ORTHO
lvi = np.empty(len(me.loops), np.int32)
me.loops.foreach_get("vertex_index", lvi)
proj.data.foreach_set("uv", np.stack([u[lvi], w[lvi]], axis=1).astype(np.float32).ravel())
me.uv_layers.active = bakeuv


def build(stage):
    m = bpy.data.materials.new("m_" + stage)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    geo = nt.nodes.new("ShaderNodeNewGeometry")

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(view_path)
    tex.extension = "CLIP"
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = proj.name
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])

    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = tuple(float(x) for x in d)
    nt.links.new(geo.outputs["Normal"], dot.inputs[0])
    mx_ = nt.nodes.new("ShaderNodeMath")
    mx_.operation = "MAXIMUM"
    mx_.inputs[1].default_value = 0.0
    nt.links.new(dot.outputs["Value"], mx_.inputs[0])
    mn_ = nt.nodes.new("ShaderNodeMath")
    mn_.operation = "MINIMUM"
    mn_.inputs[1].default_value = 1.0
    nt.links.new(mx_.outputs[0], mn_.inputs[0])

    if stage == "s1_facing":
        nt.links.new(mx_.outputs[0], em.inputs["Color"])
    elif stage == "s2_pow":
        pw = nt.nodes.new("ShaderNodeMath")
        pw.operation = "POWER"
        pw.inputs[1].default_value = EXP
        nt.links.new(mn_.outputs[0], pw.inputs[0])
        nt.links.new(pw.outputs[0], em.inputs["Color"])
    else:
        pw = nt.nodes.new("ShaderNodeMath")
        pw.operation = "POWER"
        pw.inputs[1].default_value = EXP
        nt.links.new(mn_.outputs[0], pw.inputs[0])
        cur = pw.outputs[0]
        if stage in ("s3_angle", "s4_alpha", "s5_frustum", "s6_feather", "s7_vis"):
            ang = nt.nodes.new("ShaderNodeMath")
            ang.operation = "GREATER_THAN"
            ang.inputs[1].default_value = COS_CUT
            nt.links.new(mn_.outputs[0], ang.inputs[0])
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            nt.links.new(cur, mul.inputs[0])
            nt.links.new(ang.outputs[0], mul.inputs[1])
            cur = mul.outputs[0]
        if stage in ("s4_alpha", "s5_frustum", "s6_feather", "s7_vis"):
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            nt.links.new(cur, mul.inputs[0])
            nt.links.new(tex.outputs["Alpha"], mul.inputs[1])
            cur = mul.outputs[0]
        if stage in ("s5_frustum", "s6_feather", "s7_vis"):
            cp = nt.nodes.new("ShaderNodeCombineXYZ")
            for i in range(3):
                cp.inputs[i].default_value = float(loc[i])
            sub = nt.nodes.new("ShaderNodeVectorMath")
            sub.operation = "SUBTRACT"
            nt.links.new(geo.outputs["Position"], sub.inputs[0])
            nt.links.new(cp.outputs["Vector"], sub.inputs[1])

            def ax(vec):
                dv = nt.nodes.new("ShaderNodeVectorMath")
                dv.operation = "DOT_PRODUCT"
                dv.inputs[1].default_value = tuple(float(x) for x in vec)
                nt.links.new(sub.outputs["Vector"], dv.inputs[0])
                sc = nt.nodes.new("ShaderNodeMath")
                sc.operation = "MULTIPLY"
                sc.inputs[1].default_value = 2.0 / ORTHO
                nt.links.new(dv.outputs["Value"], sc.inputs[0])
                ab = nt.nodes.new("ShaderNodeMath")
                ab.operation = "ABSOLUTE"
                nt.links.new(sc.outputs[0], ab.inputs[0])
                return ab.outputs[0]

            e = nt.nodes.new("ShaderNodeMath")
            e.operation = "MAXIMUM"
            nt.links.new(ax(rot @ Vector((1.0, 0.0, 0.0))), e.inputs[0])
            nt.links.new(ax(rot @ Vector((0.0, 1.0, 0.0))), e.inputs[1])
            inside = nt.nodes.new("ShaderNodeMath")
            inside.operation = "LESS_THAN"
            inside.inputs[1].default_value = 1.0
            nt.links.new(e.outputs[0], inside.inputs[0])
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            nt.links.new(cur, mul.inputs[0])
            nt.links.new(inside.outputs[0], mul.inputs[1])
            cur = mul.outputs[0]
        if stage in ("s6_feather", "s7_vis"):
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.interpolation_type = "SMOOTHSTEP"
            mr.inputs["From Min"].default_value = 1.0 - FEATHER
            mr.inputs["From Max"].default_value = 1.0
            mr.inputs["To Min"].default_value = 1.0
            mr.inputs["To Max"].default_value = 0.0
            mr.clamp = True
            nt.links.new(e.outputs[0], mr.inputs["Value"])
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            nt.links.new(cur, mul.inputs[0])
            nt.links.new(mr.outputs["Result"], mul.inputs[1])
            cur = mul.outputs[0]
        if stage == "s7_vis":
            dist = nt.nodes.new("ShaderNodeVectorMath")
            dist.operation = "LENGTH"
            nt.links.new(sub.outputs["Vector"], dist.inputs[0])
            ndir = nt.nodes.new("ShaderNodeVectorMath")
            ndir.operation = "NORMALIZE"
            nt.links.new(sub.outputs["Vector"], ndir.inputs[0])
            ray = nt.nodes.new("ShaderNodeRaycast")
            nt.links.new(cp.outputs["Vector"], ray.inputs["Position"])
            nt.links.new(ndir.outputs["Vector"], ray.inputs["Direction"])
            ln = nt.nodes.new("ShaderNodeMath")
            ln.operation = "MULTIPLY_ADD"
            ln.inputs[1].default_value = 1.0
            ln.inputs[2].default_value = EPS * 10.0
            nt.links.new(dist.outputs["Value"], ln.inputs[0])
            nt.links.new(ln.outputs[0], ray.inputs["Length"])
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            nt.links.new(cur, mul.inputs[0])
            nt.links.new(ray.outputs["Is Hit"], mul.inputs[1])
            cur = mul.outputs[0]
        nt.links.new(cur, em.inputs["Color"])

    t = nt.nodes.new("ShaderNodeTexImage")
    t.name = "BakeTarget"
    return m, nt, t


for stage in ("s1_facing", "s2_pow", "s3_angle", "s4_alpha", "s5_frustum",
              "s6_feather", "s7_vis"):
    m, nt, tgt = build(stage)
    me.materials.clear()
    me.materials.append(m)
    img = bpy.data.images.new("b_" + stage, S, S, alpha=True, float_buffer=True)
    tgt.image = img
    nt.nodes.active = tgt
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", width=S, height=S)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)[:, 0]
    print(f"STAGE {stage:12} mean={px.mean():.4f} max={px.max():.4f} "
          f">0.01={(px > 0.01).mean():.1%}", flush=True)
print("STAGE DONE", flush=True)
