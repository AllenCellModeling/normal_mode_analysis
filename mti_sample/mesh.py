import numpy as np
import math
import itertools
from skimage import measure


class Mesh():

    def __init__(self, verts, faces):
        """Creates a mesh object in any # of spatial dimensions, constructed with input vertices and faces.
        :param verts: Each item in the list is one vertex, and each vertex is a list of positions in each spatial dimension
        :param faces: Each item in list is a face, and each face is a list of vertices connected to make this face
        """
        self.verts = verts
        self.faces = faces
        self.npts = int(verts.shape[0])
        self.ndims = int(verts[0].shape[0])


def mesh_from_models(model):
    """Creates a mesh object from preset models, by selecting that model's vertices and faces from the respective dictionaries."""
    
    # set shorthand param for icosphere generation
    ico = (1 + np.sqrt(5)) / 2

    # test model definitions: mass positions or mesh vertices
    model_verts = {
        
        # 1D 2 mass line
        '1D_2m': np.array([[0.], [1.] ]),
        
        # 1D 3 mass line
        '1D_3m': np.array([[0.,], [1.,], [2., ] ]),
        
        # 3D 2 mass line
        '3D_2m': np.array([[0., 0., 0.], [1., 0., 0.] ]),
        
        # 2D square
        '2D_sq': np.array([[0.,0.], [0., 1.], [1.,1.], [1.,0.] ]),
        
        # 2D rectangle
        '2D_rt': np.array([[0.,0.], [0., 1.], [0., 2.], [1.,2.], [1.,1.], [1.,0.] ]),
        
        # 3D cube
        '3D_cb': np.array([[0.,0.,0.], [0., 1.,0,], [1.,1.,0.], [1.,0.,0.],[0.,0.,1.], [0., 1.,1,], [1.,1.,1.], [1.,0.,1.] ]),
        
        # 2D triangle
        '2D_tr': np.array([[1.,0.], [0.5, np.sqrt(3)/2.], [0.,0.], ]),
        
        # 3D tetrahedron
        '3D_th': np.array([[1.,0.,0.], [0.5, np.sqrt(3)/2.,0.], [0.,0.,0.], [np.sqrt(3)/2., np.sqrt(3)/2., 1.] ]),
        
        # 3D icosahedron
        '3D_ico': np.array([[-1, ico, 0], [1, ico, 0], [-1, -ico, 0], [1, -ico, 0], [0, -1, ico], [0, 1, ico], [0, -1, -ico], [0, 1, -ico], 
                            [ico, 0, -1], [ico, 0, 1], [-ico, 0, -1], [-ico, 0, 1]])

    }

    # test model definitions: mesh connectivities (for each face, list contains a list of all vertex indeces for vertices each face connects)
    model_faces = {
        '1D_2m': [[0,1],],
        '1D_3m': [[0,1],[1,2]],
        '3D_2m': [[0,1],],
        '2D_sq': [[0,1], [1,2], [2,3], [3,0]],
        '2D_rt': [[0,1], [1,2], [2,3], [3,4], [4,5], [5, 0]],
        '3D_cb': [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,0], [0,4], [1,5], [2,6], [3,7]],
        '2D_tr': [[0,1], [1,2], [2,0]],
        '3D_th': [[0,1], [1,2], [2,0], [0,3], [1,3], [2,3]],
        '3D_ico': [ [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    }

    return Mesh(model_verts[model], model_faces[model])


def polygon_mesh(r=5, N=5, fully_connect=False):
    """Creates a mesh of an N-sided polygon with radius r. It can be fully connected if this paramter is set to true.
    :param r: radius of polygon
    :param N: number of sides of polygon
    :param fully_connect: flag to take model or input vertices and generate fully connected mesh
    :return: Mesh object describing N-sided polygon with radius r.
    """

    verts = [np.array((math.cos(2*np.pi/N*x)*r, math.sin(2*np.pi/N*x)*r)) for x in range(0,N)]
    if not fully_connect:
        faces = []
        for i in range(N-1):
            faces.append([i, i+1])
        faces.append([N-1,0])
    else:
        faces = fully_connect_mesh(verts)
    return Mesh(np.array(verts), np.array(faces))


def volume_trimesh(r=5, ss=1, fully_connect=False):
    """Creates a 3D surface mesh that approaches a sphere as vertices are added.
    :param r: sphere radius
    :param ss: step size for marching cubes meshing
    :param fully_connect: flag to take model or input vertices and generate fully connected mesh
    :return: Mesh object describing 3D surface with approximate radius r.
    """
    
    # Create spherical mask
    size=2*r+3
    center = np.array([(size-1)/2, (size-1)/2, (size-1)/2])
    mask = np.zeros((size,size,size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if np.linalg.norm(np.array([i,j,k])-center)<=r:
                    mask[i,j,k] = 1
                    
    # mesh the mask into verts and faces
    verts, faces, n, v = measure.marching_cubes_lewiner(mask, step_size=ss)
    if fully_connect:
        faces = fully_connect_mesh(verts)
    return Mesh(verts, faces)
    
	
def fully_connect_mesh(verts):
    """Connects all vertices in input mesh to all other vertices.
    :param verts: mesh vertices
    :return: faces of mesh, connecting all vertices to all others
    """
    return list(itertools.combinations(range(len(verts)),2))
