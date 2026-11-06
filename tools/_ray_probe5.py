"""探针 5：封闭球 + 与真实 bake 相同的设置，判定 raycast 能否看到「正在着色的物体」。

球半径 1，相机正交在 x=3.5，射线从相机指向球面点，Length = dist + eps*10。
球是封闭的：射线穿过前表面后应命中背面 → Is Hit 应为 1（可见）。

    blender.exe -b -P tools/_ray_probe5.py
"""
import bpy
import numpy as np

R = 48


def run(tag, osl, only_local, length_extra):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.bake.margin = 0
    if osl:
        scene.cycles.shading_system = True

    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    cam_data = bpy.data.cameras.new("c")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.8
    cam = bpy.data.objects.new("c", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (3.5, 0, 0)
    cam.rotation_euler = (0.0, np.pi / 2, 0.0)   # -Z 指向 -X，即朝向原点
    scene.camera = cam

    import mathutils
    m = bpy.data.materials.new("m")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    loc = nt.nodes.new("ShaderNodeCombineXYZ")
    for i, v in enumerate(cam.location):
        loc.inputs[i].default_value = v
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    nt.links.new(geo.outputs["Position"], sub.inputs[0])
    nt.links.new(loc.outputs["Vector"], sub.inputs[1])
    dist = nt.nodes.new("ShaderNodeVectorMath")
    dist.operation = "LENGTH"
    nt.links.new(sub.outputs["Vector"], dist.inputs[0])
    ndir = nt.nodes.new("ShaderNodeVectorMath")
    ndir.operation = "NORMALIZE"
    nt.links.new(sub.outputs["Vector"], ndir.inputs[0])

    ray = nt.nodes.new("ShaderNodeRaycast")
    ray.only_local = only_local
    nt.links.new(loc.outputs["Vector"], ray.inputs["Position"])
    nt.links.new(ndir.outputs["Vector"], ray.inputs["Direction"])
    ln = nt.nodes.new("ShaderNodeMath")
    ln.operation = "MULTIPLY_ADD"
    ln.inputs[1].default_value = 1.0
    ln.inputs[2].default_value = length_extra
    nt.links.new(dist.outputs["Value"], ln.inputs[0])
    nt.links.new(ln.outputs[0], ray.inputs["Length"])

    nt.links.new(ray.outputs["Is Hit"], em.inputs["Color"])
    tgt = nt.nodes.new("ShaderNodeTexImage")
    tgt.name = "BakeTarget"
    obj.data.materials.clear()
    obj.data.materials.append(m)
    img = bpy.data.images.new(f"b_{tag}", R, R, alpha=True, float_buffer=True)
    tgt.image = img
    nt.nodes.active = tgt

    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    res = bpy.ops.object.bake(type="EMIT", width=R, height=R)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)
    print(f"PROBE {tag:40} bake={'FIN' if 'FINISHED' in res else res} "
          f"IsHit mean={px[:, 0].mean():.3f} max={px[:, 0].max():.3f}", flush=True)


run("sphere osl=F only_local=F extra=2.6e-4", False, False, 2.6e-4)
run("sphere osl=T only_local=F extra=2.6e-4", True, False, 2.6e-4)
run("sphere osl=F only_local=T extra=2.6e-4", False, True, 2.6e-4)
run("sphere osl=F only_local=F extra=1.0(GOOD)", False, False, 1.0)
print("PROBE DONE", flush=True)
