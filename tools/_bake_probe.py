import bpy, numpy as np
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="D:/code/meow3d/out/native_q21_white_00001_.glb")
o=[x for x in bpy.context.scene.objects if x.type=='MESH'][0]
bpy.ops.object.select_all(action='DESELECT'); o.select_set(True)
bpy.context.view_layer.objects.active=o
mesh=o.data
for l in list(mesh.uv_layers): mesh.uv_layers.remove(l)
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(island_margin=0.003); bpy.ops.object.mode_set(mode='OBJECT')
uv=mesh.uv_layers.active; uv.name='BakeUV'; uv.active_render=True
mat=bpy.data.materials.new('p'); mat.use_nodes=True; nt=mat.node_tree; nt.nodes.clear()
out=nt.nodes.new('ShaderNodeOutputMaterial'); em=nt.nodes.new('ShaderNodeEmission')
em.inputs['Color'].default_value=(0.8,0.2,0.2,1.0)
nt.links.new(em.outputs['Emission'],out.inputs['Surface'])
img=bpy.data.images.new('t',width=256,height=256,alpha=False)
t=nt.nodes.new('ShaderNodeTexImage'); t.image=img; nt.nodes.active=t; t.select=True
mesh.materials.clear(); mesh.materials.append(mat)
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.samples=1
sc.render.bake.margin=4
print('PROBE uv',uv.name,'faces',len(mesh.polygons),flush=True)
r=bpy.ops.object.bake(type='EMIT',uv_layer=uv.name)
print('PROBE result',r,flush=True)
px=np.empty(256*256*4,np.float32); img.pixels.foreach_get(px)
print('PROBE mean',float(px.reshape(-1,4)[:,:3].mean()),'max',float(px.max()),flush=True)
