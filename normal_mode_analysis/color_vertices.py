# start blender:
# /Applications/Blender.app/Contents/MacOS/Blender --python-console
#
# run script:
# >>> exec(open("/Users/blairl/Dropbox/AICS/VertexColoredMesh/color_vertices.py").read())
# >>> quit()
# 
# view result in "Vertex Paint" mode in Blender

import logging
import bpy
import os
import numpy as np
import pandas as pd

mesh = 4
mode = 0
file = 'mesh_'+str(mesh)+'_mode_'+str(mode)

# setup
local = '/Users/juliec/projects/'   # replace "local_path" with your own repo's local path
repo = local+'normal_mode_analysis/normal_mode_analysis/' 

# input mesh file path
mesh_input_file_path = repo+'nucleus_mesh_data/sample_trimeshes_from_blair/mean_nuc_mesh_uniform_'+str(mesh)'.stl'

# results file path
output_file_path = repo+'nucleus_nma_heatmaps/'+file+'_colored.blend'

# input eigvec coloring values
eigvec_data = pd.read_pickle(repo+'nucleus_nma/mode_table_nuc_mesh_'+str(mesh)'.pickle')
colors_input_file_path = repo+'nucleus_nma/'+file+'_mags.npy'
vecs = np.save(colors_input_file_path, eigvec_data.iloc[mode]['mags'])

# delete default blender cube
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# import STL
bpy.ops.import_mesh.stl(filepath=mesh_input_file_path)

# edit mode
mode = bpy.context.active_object.mode
bpy.ops.object.mode_set(mode = 'OBJECT')

# get mesh and setup vertex colors
mesh = bpy.context.active_object.data
if not mesh.vertex_colors:
    mesh.vertex_colors.new()
color_layer = mesh.vertex_colors.active

# load color values
values = np.load(colors_input_file_path)
colors = []
for value in values:
    colors.append([1-value, 0., value, 1.])

# set colors
vertex_indices = []
for poly in mesh.polygons:
    for v in poly.vertices:
        vertex_indices.append(v)           
for i in range(len(color_layer.data)):
    color_layer.data[i].color = colors[vertex_indices[i]]

# object mode
bpy.ops.object.mode_set(mode=mode)

# save blend file
if os.path.exists(output_file_path):
    os.remove(output_file_path) 
bpy.ops.wm.save_as_mainfile(filepath=output_file_path)