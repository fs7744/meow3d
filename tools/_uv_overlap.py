"""测 UV 岛是否重叠：把每个三角形 id 栅格化进 UV 网格，统计 >1 个 id 的格子占比。

    blender.exe -b -P tools/_uv_overlap.py -- <mesh.glb> [size]
"""
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
path = argv[0]
S = int(argv[1]) if len(argv) > 1 else 1024

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
me = bpy.context.view_layer.objects.active.data
me.calc_loop_triangles()

print(f"UV layers: {[l.name for l in me.uv_layers]}", flush=True)
for layer in me.uv_layers:
    uv = np.empty(len(me.loops) * 2, np.float32)
    layer.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    tri = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tri)
    tri = tri.reshape(-1, 3)
    # 每个三角形取若干重心采样点，落格子；随后看每格被几个不同三角形占用
    owner = np.full((S, S), -1, np.int32)
    hits = np.zeros((S, S), np.int32)
    rng = np.random.default_rng(0)
    bary = np.array([[1/3, 1/3, 1/3],
                     [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
                     [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    for t in range(len(tri)):
        p = uv[tri[t]]
        pts = bary @ p
        idx = np.clip((pts * S).astype(np.int32), 0, S - 1)
        gx, gy = idx[:, 0], idx[:, 1]
        prev = owner[gx, gy]
        hits[gx, gy] += 1
        overlap = (prev != -1) & (prev != t)
        if overlap.any():
            owner[gx[overlap], gy[overlap]] = -2      # -2 标记为冲突
        owner[gx[~overlap & (prev == -1)], gy[~overlap & (prev == -1)]] = t
    touched = hits > 0
    conflict = owner == -2
    print(f"  {layer.name}: 覆盖格 {touched.mean():.1%}  "
          f"冲突格(多岛共用) {conflict.sum()} / {touched.sum()} = "
          f"{conflict.sum() / max(touched.sum(), 1):.2%}", flush=True)
