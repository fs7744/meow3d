import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="D:/code/meow3d/out/native_q21_white_00001_.glb")
o=[x for x in bpy.context.scene.objects if x.type=='MESH'][0]
m=o.data
for i in range(12):
    u=m.uv_layers.new(name=f"P{i:02d}")
    print("UVLIMIT",i,"->",None if u is None else u.name,flush=True)
    if u is None: break
print("UVLIMIT total",len(m.uv_layers),flush=True)
