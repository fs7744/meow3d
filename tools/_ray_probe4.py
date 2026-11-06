"""探针 4：确认 Cycles Raycast 是否排除「正在着色的那个图元」。

A 只有一张平面（射线必然穿过它自身）        → IsHit?
B 平面 + 背后 z=-1 的第二张平面             → IsHit?（若 A=0、B=1，则自图元被排除）
C 平面 + 前方 z=1 的独立方块（独立物体）     → IsHit? HitDistance?

    blender.exe -b -P tools/_ray_probe4.py
"""
import bpy
import numpy as np

R = 32


def run(tag, behind, front):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.bake.margin = 0

    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    bpy.ops.object.mode_set(mode="OBJECT")

    extra = []
    if behind:
        bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, -1))
        extra.append(bpy.context.object)
    if front:
        bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0, 0, 1))
        extra.append(bpy.context.object)

    cam_data = bpy.data.cameras.new("c")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.0
    cam = bpy.data.objects.new("c", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (0, 0, 3)
    scene.camera = cam

    def mat_for(channel):
        m = bpy.data.materials.new("m" + channel)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        pos = nt.nodes.new("ShaderNodeCombineXYZ")
        pos.inputs[0].default_value = 0.0
        pos.inputs[1].default_value = 0.0
        pos.inputs[2].default_value = 3.0
        ray = nt.nodes.new("ShaderNodeRaycast")
        nt.links.new(pos.outputs["Vector"], ray.inputs["Position"])
        ray.inputs["Direction"].default_value = (0.0, 0.0, -1.0)
        ray.inputs["Length"].default_value = 5.0
        nt.links.new(ray.outputs[channel], em.inputs["Color"])
        t = nt.nodes.new("ShaderNodeTexImage")
        t.name = "BakeTarget"
        return m, nt, t

    m, nt, tgt = mat_for("Is Hit")
    m2, nt2, tgt2 = mat_for("Hit Distance")

    obj.data.materials.clear()
    obj.data.materials.append(m)
    img = bpy.data.images.new(f"hit_{tag}", R, R, alpha=True, float_buffer=True)
    tgt.image = img
    nt.nodes.active = tgt
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", width=R, height=R)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)

    obj.data.materials.clear()
    obj.data.materials.append(m2)
    img2 = bpy.data.images.new(f"dist_{tag}", R, R, alpha=True, float_buffer=True)
    tgt2.image = img2
    nt2.nodes.active = tgt2
    bpy.ops.object.bake(type="EMIT", width=R, height=R)
    px2 = np.array(img2.pixels[:], dtype=np.float32).reshape(-1, 4)

    print(f"PROBE {tag:16} IsHit={px[:, 0].mean():.3f} "
          f"HitDist mean={px2[:, 0].mean():.3f} max={px2[:, 0].max():.3f}", flush=True)


run("A_plane_only", False, False)
run("B_behind", True, False)
run("C_front_block", False, True)
print("PROBE DONE", flush=True)
