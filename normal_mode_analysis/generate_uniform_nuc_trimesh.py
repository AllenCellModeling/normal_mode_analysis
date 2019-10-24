# install Blender 2.8:
# https://www.blender.org/download/
#
# start Blender with python interpreter:
# /Applications/Blender.app/Contents/MacOS/Blender --python-console
#
# run script:
# >>> exec(open([full path to this script]).read())
# >>> quit()

import logging
import bpy
import os
import numpy as np

# parameters
mesh_type = 'nucleus' # set to either 'nucleus' or 'icosphere'
mesh_density = 5 # 1-10 (but 10 creates like 5 million vertices)
object_name = 'Mean Nuc Mesh' # blender sets mesh object name from input file name, so this should match

# replace "local_path" with your own repo's local path
local_path = '/Users/juliec/projects'
repo_path = '/normal_mode_analysis/normal_mode_analysis/nucleus_mesh_data/sample_trimeshes_from_blair/'
input_mesh_path = local_path + repo_path + 'mean_nuc_mesh.stl'
if mesh_type == 'nucleus':
	output_path = local_path + repo_path + 'mean_nuc_mesh_uniform_{}'.format(mesh_density)
elif mesh_type == 'icosphere':
	output_path = local_path + repo_path + 'icosphere_{}'.format(mesh_density)
else:
	print('Please select "nucleus" or "icosphere" for your mesh_type. Defaulting to icosphere.')
	output_path = local_path + repo_path + 'icosphere_{}'.format(mesh_density)
	
# delete default blender cube
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()
    
# import input mesh STL and center it 
# (centering should probably be done earlier in pipeline, but doing it here for now)
if mesh_type == 'nucleus':
	bpy.ops.import_mesh.stl(filepath=input_mesh_path)
	input_mesh = bpy.data.objects[object_name]
	input_mesh.location.x = -100.
	input_mesh.location.y = -100.
	input_mesh.location.z = -100.
	input_mesh.name = 'Input Mesh'

# create a sphere with icosahedral triangle mesh
bpy.ops.mesh.primitive_ico_sphere_add(
    subdivisions=mesh_density, 
    radius=100.0, 
    align='WORLD', 
    location=(0.0, 0.0, 0.0), 
    rotation=(0.0, 0.0, 0.0)
)

# shrinkwrap sphere onto input mesh
output_mesh = bpy.data.objects['Icosphere']
if mesh_type == 'nucleus':
	output_mesh.name = 'Output Mesh'
	modifier = output_mesh.modifiers.new('shrinkwrap', 'SHRINKWRAP')
	modifier.target = input_mesh
	output_mesh.select_set(True)
	bpy.ops.object.modifier_apply(apply_as='DATA', modifier='shrinkwrap')

# export STL file
if mesh_type == 'nucleus'
	input_mesh.select_set(False)
output_mesh.select_set(True)
mesh_output_path = '{}.stl'.format(output_path)
if os.path.exists(mesh_output_path):
    os.remove(mesh_output_path)
bpy.ops.export_mesh.stl(
    filepath=mesh_output_path, 
    check_existing=False, 
    filter_glob='*.stl', 
    use_selection=True, 
    global_scale=1.0, 
    use_scene_unit=False, 
    ascii=False, 
    use_mesh_modifiers=True, 
    batch_mode='OFF', 
    axis_forward='Y', 
    axis_up='Z'
)

# export vertices
vertices_output_path = '{}_vertices.npy'.format(output_path)
vertices = []
for vertex in bpy.data.meshes['Icosphere'].vertices:
    vertices.append([vertex.co[0], vertex.co[1], vertex.co[2]])
np.save(vertices_output_path, vertices)
    
# export faces
faces_output_path = '{}_faces.npy'.format(output_path)
faces = []
for face in bpy.data.meshes['Icosphere'].polygons:
    faces.append([face.vertices[0], face.vertices[1], face.vertices[2]])
np.save(faces_output_path, faces)

# save blender file for viewing overlaid input and output meshes
blender_output_path = '{}.blend'.format(output_path)
if os.path.exists(blender_output_path):
    os.remove(blender_output_path)
bpy.ops.wm.save_as_mainfile(filepath=blender_output_path)
