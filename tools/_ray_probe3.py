"""探针 3：Raycast 节点在 bake 里到底能不能命中，逐个排除。

变量：
  1. 遮挡物是「独立物体」还是「被烘的同一个物体」（self hit）
  2. only_local = True / False
  3. Cycles shading_system (OSL) = True / False

    blender.exe -b -P tools/_ray_probe3.py
"""
import bpy
import numpy as np

R = 32


def build(tag, occluder_shared, only_local, osl):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.bake.margin = 0
    if osl:
        scene.cycles.shading_system = True

    # 被烘的平面
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    target_obj = bpy.context.object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    bpy.ops.object.mode_set(mode="OBJECT")

    # 遮挡物：z=1 的方块（在相机 3.0 与平面 0 之间）
    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0, 0, 1))
    occ = bpy.context.object

    if occluder_shared:
        # 把方块并进被烘物体 → 遮挡成为 self hit
        for o in bpy.context.selected_objects:
            o.select_set(False)
        target_obj.select_set(True)
        occ.select_set(True)
        bpy.context.view_layer.objects.active = target_obj
        bpy.ops.object.join()
        target_obj = bpy.context.view_layer.objects.active
        # join 后几何在同一物体上，UV 可能退化，但 bake 目标图仍在，够用
    else:
        occ.hide_render = True

    cam_data = bpy.data.cameras.new("c")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.0
    cam = bpy.data.objects.new("c", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (0, 0, 3)
    scene.camera = cam

    m = bpy.data.materials.new("m")
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
    ray.only_local = only_local
    nt.links.new(pos.outputs["Vector"], ray.inputs["Position"])
    ray.inputs["Direction"].default_value = (0.0, 0.0, -1.0)
    ray.inputs["Length"].default_value = 4.0
    nt.links.new(ray.outputs["Is Hit"], em.inputs["Color"])

    target = nt.nodes.new("ShaderNodeTexImage")
    target.name = "BakeTarget"
    target_obj.data.materials.clear()
    target_obj.data.materials.append(m)
    img = bpy.data.images.new(f"b_{tag}", R, R, alpha=True, float_buffer=True)
    target.image = img
    nt.nodes.active = target

    for o in bpy.context.selected_objects:
        o.select_set(False)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    res = bpy.ops.object.bake(type="EMIT", width=R, height=R)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)
    print(f"PROBE {tag:42} bake={'FINISHED' if 'FINISHED' in res else res} "
          f"IsHit mean={px[:, 0].mean():.3f}", flush=True)


build("shared_occ only_local=F osl=F", True, False, False)
build("shared_occ only_local=T osl=F", True, True, False)
build("shared_occ only_local=F osl=T", True, False, True)
build("separate_occ only_local=F osl=F", False, False, False)
build("separate_occ only_local=F osl=T", False, False, True)
print("PROBE DONE", flush=True)
