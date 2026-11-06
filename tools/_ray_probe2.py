"""对照探针：区分「ShaderNodeRaycast 返回 0」与「bake 静默失败」。

    blender.exe -b -P tools/_ray_probe2.py

三组材质，同一场景、同一次 bake 流程：
  A 常量 Emission=1        → 证明 bake 本身能出非零（对照组）
  B Raycast 的 Is Hit      → 该点是否被遮挡
  C Raycast 的 Hit Distance→ 命中距离本身

场景：平面 z=0（2x2），正交相机在 z=3 朝下。射线从相机位置出发向下。
Length=2.9 不该命中（平面在 3.0 处）；Length=3.5 必须命中。
"""
import bpy
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
plane = bpy.context.object
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
bpy.ops.object.mode_set(mode="OBJECT")

cam_data = bpy.data.cameras.new("c")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.0
cam = bpy.data.objects.new("c", cam_data)
scene.collection.objects.link(cam)
cam.location = (0, 0, 3)
scene.camera = cam

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.render.bake.margin = 0
scene.render.bake.use_clear = True

R = 64


def bake(mat, tag):
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    img = bpy.data.images.new(f"b_{tag}", R, R, alpha=True, float_buffer=True)
    node = mat.node_tree.nodes.get("BakeTarget")
    node.image = img
    mat.node_tree.nodes.active = node
    for o in bpy.context.selected_objects:
        o.select_set(False)
    plane.select_set(True)
    bpy.context.view_layer.objects.active = plane
    res = bpy.ops.object.bake(type="EMIT", width=R, height=R)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)
    print(f"PROBE {tag:24} bake={res} channel0 mean={px[:, 0].mean():.4f} "
          f"max={px[:, 0].max():.4f}", flush=True)
    return px[:, 0]


def base_mat(tag):
    m = bpy.data.materials.new(f"m_{tag}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    target = nt.nodes.new("ShaderNodeTexImage")
    target.name = "BakeTarget"
    target.label = "BakeTarget"
    return m, nt, em, target


# ---------- A 对照组：常量 1 ----------
m, nt, em, tgt = base_mat("const")
em.inputs["Color"].default_value = (1.0, 0.13, 0.0, 1.0)
a = bake(m, "A_const")

# ---------- B/C Raycast ----------
for length in (2.9, 3.5):
    m, nt, em, tgt = base_mat(f"ray{length}")
    pos = nt.nodes.new("ShaderNodeCombineXYZ")
    pos.inputs[0].default_value = 0.0
    pos.inputs[1].default_value = 0.0
    pos.inputs[2].default_value = 3.0
    ray = nt.nodes.new("ShaderNodeRaycast")
    print(f"PROBE node inputs  = {[i.name for i in ray.inputs]}", flush=True)
    print(f"PROBE node outputs = {[o.name for o in ray.outputs]}", flush=True)
    nt.links.new(pos.outputs["Vector"], ray.inputs["Position"])
    ray.inputs["Direction"].default_value = (0.0, 0.0, -1.0)
    ray.inputs["Length"].default_value = float(length)
    print(f"PROBE len readback = {ray.inputs['Length'].default_value}", flush=True)
    nt.links.new(ray.outputs["Is Hit"], em.inputs["Color"])
    bake(m, f"B_IsHit_L{length}")
    nt.links.new(ray.outputs["Hit Distance"], em.inputs["Color"])
    bake(m, f"C_HitDist_L{length}")

print("PROBE DONE", flush=True)
