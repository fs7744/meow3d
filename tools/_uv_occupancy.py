"""测 UV 岛在 0..1 图集里的实际占用（三角形面积和 + 栅格化占用）。

    blender.exe -b -P tools/_uv_occupancy.py -- <mesh.glb> [size]
"""
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
path = argv[0]
size = int(argv[1]) if len(argv) > 1 else 1024

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
me = obj.data

print(f"UV layers: {[l.name for l in me.uv_layers]}", flush=True)
for layer in me.uv_layers:
    uv = np.empty(len(me.loops) * 2, np.float32)
    layer.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    loops = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loops)
    tri = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.calc_loop_triangles()
    me.loop_triangles.foreach_get("loops", tri)
    tri = tri.reshape(-1, 3)
    a = uv[tri[:, 0]]
    b = uv[tri[:, 1]]
    c = uv[tri[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    # 栅格化占用（忽略重叠，只问「有没有 texel 落在这里」）
    grid = np.zeros((size, size), np.uint8)
    p = (uv * size).astype(np.int32).clip(0, size - 1)
    grid[p[:, 0], p[:, 1]] = 1
    print(f"  {layer.name}: area_sum={area.sum():.3f}  "
          f"vertex_texels={grid.mean():.3f}  size={size}", flush=True)
