"""测白模是不是双层壳：从表面点沿「朝内」方向打射线，看最近的另一面有多远。

    blender.exe -b -P tools/_probe_shell.py -- <mesh.glb>
"""
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
obj.matrix_world = Matrix.Identity(4)
me = obj.data

BVH = BVHTree.FromPolygons([v.co.copy() for v in me.vertices],
                           [list(p.vertices) for p in me.polygons],
                           all_triangles=False)

rng = np.random.default_rng(0)
idx = rng.choice(len(me.vertices), size=min(400, len(me.vertices)), replace=False)
thick = []
for i in idx:
    v = me.vertices[int(i)]
    n = v.normal.normalized()
    if n.length < 0.5:
        continue
    loc, nrm, _, dist = BVH.ray_cast(v.co - n * 1e-6, -n, 5.0)
    if loc is not None and dist > 1e-6:
        thick.append(float(dist))
thick = np.array(thick)
if thick.size:
    print(f"SHELL samples={thick.size} mean={thick.mean():.4f} "
          f"p10={np.percentile(thick,10):.4f} p50={np.percentile(thick,50):.4f} "
          f"p90={np.percentile(thick,90):.4f} max={thick.max():.4f}", flush=True)
    print(f"SHELL frac<0.02={np.mean(thick<0.02):.2f} frac<0.1={np.mean(thick<0.1):.2f} "
          f"frac<0.3={np.mean(thick<0.3):.2f}", flush=True)
dims = obj.dimensions
print(f"SHELL bbox dims={tuple(round(d,3) for d in dims)} verts={len(me.vertices)} "
      f"faces={len(me.polygons)}", flush=True)
