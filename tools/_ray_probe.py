"""最小验证：ShaderNodeRaycast 在 Cycles bake 里的 Is Hit / Hit Distance 行为。

    blender.exe -b -P tools/_ray_probe.py

平面在 z=0，相机在 z=3，从相机向下打 2.9（本该打不到，因为距离是 3.0）。
再打 3.5（会穿过平面）。比较两种的可见性。
"""
import bpy
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
plane = bpy.context.object
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode="OBJECT")

cam_data = bpy.data.cameras.new("c")
cam = bpy.data.objects.new("c", cam_data)
bpy.context.scene.collection.objects.link(cam)
cam.location = (0, 0, 3)
bpy.context.scene.camera = cam


def probe(length):
    m = bpy.data.materials.new(f"m{length}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    # 照 StableGen：Position 必须**连线**（CombineXYZ），default_value 不生效。
    cam_xyz = nt.nodes.new("ShaderNodeCombineXYZ")
    cam_xyz.inputs[0].default_value = 0.0
    cam_xyz.inputs[1].default_value = 0.0
    cam_xyz.inputs[2].default_value = 3.0
    ray = nt.nodes.new("ShaderNodeRaycast")
    nt.links.new(cam_xyz.outputs["Vector"], ray.inputs["Position"])
    ray.inputs["Direction"].default_value = (0.0, 0.0, -1.0)
    ray.inputs["Length"].default_value = float(length)
    nt.links.new(ray.outputs["Is Hit"], em.inputs["Color"])   # 直接输出 Is Hit

    plane.data.materials.clear()
    plane.data.materials.append(m)
    img = bpy.data.images.new(f"b{length}", 32, 32)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.nodes.active = tex
    bpy.ops.object.bake(type="EMIT")
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)
    print(f"PROBE length={length:.2f}  IsHit mean={px[:,0].mean():.3f} "
          f"min={px[:,0].min():.3f} max={px[:,0].max():.3f}", flush=True)


bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.device = "CPU"
bpy.context.scene.cycles.samples = 1
for L in (2.9, 3.5):
    probe(L)
print("PROBE DONE")
