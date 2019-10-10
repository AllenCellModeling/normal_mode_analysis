from get_mesh_modes import *
import numpy as np
import math
import itertools
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")

sb.set_palette(sb.color_palette("Set2"))
color_list = sb.color_palette("Set2")*255


# test model definitions: mass positions/ mesh vertices
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
    '2D_rt2': np.array([[0.,0.], [0., 1.], [0., 2.], [1.,2.], [1.,1.], [1.,0.] ]),
    
    # 3D cube
    '3D_cb': np.array([[0.,0.,0.], [0., 1.,0,], [1.,1.,0.], [1.,0.,0.],[0.,0.,1.], [0., 1.,1,], [1.,1.,1.], [1.,0.,1.] ]),
    
    # 2D triangle
    '2D_tr': np.array([[1.,0.], [0.5, np.sqrt(3)/2.], [0.,0.], ]),
    
    # 3D tetrahedron
    '3D_th': np.array([[1.,0.,0.], [0.5, np.sqrt(3)/2.,0.], [0.,0.,0.], [np.sqrt(3)/2., np.sqrt(3)/2., 1.] ])
}

# test model definitions: mass/mesh connectivities (same naming convention as written out above)
model_faces = {
    '1D_2m': [[0,1],],
    '1D_3m': [[0,1],[1,2]],
    '3D_2m': [[0,1],],
    '2D_sq': [[0,1], [1,2], [2,3], [3,0],],
    '2D_rt': [[0,1], [1,2], [2,3], [3,0],],
    '2D_rt2': [[0,1], [1,2], [2,3], [3,4], [4,5], [5, 0]],
    '3D_cb': [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,0], [0,4], [1,5], [2,6], [3,7]],
    '2D_tr': [[0,1], [1,2], [2,0]],
    '3D_th': [[0,1], [1,2], [2,0], [0,3], [1,3], [2,3]],
}

    
def fully_connect_mesh(verts):
    """Connects all vertices in input mesh to all other vertices. Returns faces.
    :param verts: mesh vertices
    :return: faces of mesh, connecting all vertices to all others
    """
    return list(itertools.combinations(range(len(verts)),2))


def check_diagonalization(v, hess):
    """Returns diagonalized version of Hessian (eigenvalues are along diagonal).
    :param v: matrix of all eigenvectors
    :param hess: hessian matrix of mesh
    :return: diagonalized hessian with eigenvalues along diagonal
    """
    
    d = np.matmul(np.linalg.inv(v),np.matmul(hess,v))
    d[d<10**-5] = 0
    return np.real(d)


def check_orthogonality(v):
    """Check whether all eigenevectors are orthogonal to all others.
    :param v: matrix of all eigenvectors
    :return: True if all eigenvectors (columns of v) are normal to one another, false otherwise.
    """
    
    for indpair in list(itertools.combinations(range(v.shape[0]), 2)):
        a = indpair[0]
        b = indpair[1]
        c = np.dot(v[:,a], v[:,b])
        if np.abs(c)<10**-5: c = 0
    return np.sum(c) == 0


def nma_test_model(model, verts=None, faces=None, fully_connect=False):
    """Uses mesh to find hessian, then calculate its eigenvectors/values to define normal modes.
    :param model: string indicating a model mass system
    :param verts: mesh vertices
    :param faces: mesh faces
    :param fully_connect: flag to take model or input vertices and generate fully connected mesh
    :return: mesh vertices and faces, hessian of mesh, and hessian eigenvalues (w) and vectors (v)
    """
    
    if model is not None:
        verts = model_verts[model]
        faces = model_faces[model]
        
    if fully_connect:
        faces = fully_connect_mesh(verts)
            
    if np.linalg.norm(verts==1):
        verts *= 10
    
    hess = get_hessian_from_mesh(verts, faces)  
    w, v = np.linalg.eigh(hess)
    
    return verts, faces, hess, w, v



def nma_polygon(r, N=100, fully_connect=False, draw=False):
    """Generates N-sided polygon (which may or may not be fully-connected) and runs nma_test_model on it.
    :param r: radius of polygon
    :param N: number of sides of polygon
    :param fully_connect: flag to take model or input vertices and generate fully connected mesh
    :param draw: flag to draw all normal modes and histogram of their frequencies
    :return: eigenvalues and eigenvectors
    """

    def generate_circle_mesh(r, N):
        verts = [np.array((math.cos(2*np.pi/N*x)*r, math.sin(2*np.pi/N*x)*r)) for x in range(0,N)]
        if not fully_connect:
            faces = []
            for i in range(N-1):
                faces.append([i, i+1])
            faces.append([N-1,0])
        else:
            faces = fully_connect_mesh(verts)
        return verts, faces
    
    verts, faces = generate_circle_mesh(r, N)
    
    verts, faces, hess, w, v = nma_test_model(None, np.array(verts), faces, False)
    
    if draw:
        sb.distplot(w, kde=False, bins=N*5)
        plt.figure()
        draw_init_modes(verts, faces, v, w)
        
    return w, v


def draw_shape(verts, faces, c, axis=None):
    """Draws mesh.
    :param verts: mesh vertices
    :param faces: mesh faces
    :param c: color to draw shape lines
    :param axis: axis to draw shape on
    :return: axis shape is drawn on
    """
    if axis is None:
        plt.figure()
        axis = plt.gca()
        
    ndim = verts[0].shape[0]
        
    # get x (and y if ndim=2) coordinates for initial mass positions and draw initial shape
    for pair in faces:
        x = [verts[pair[0]][0], verts[pair[1]][0]]
        if ndim==2:
            y = [verts[pair[0]][1], verts[pair[1]][1]]
        else:
            y = np.zeros(len(x))
        axis.plot(x,y, color=c)
    return axis


def draw_mode(verts, faces, v2, axis=None):
    """Draws inital mesh shape, normal mode, and eigenvectors that take you from one to the other.
    :param verts: mesh vertices
    :param faces: mesh faces
    :param v2: eigenvector for this mode
    :param axis: axis to draw shape on
    :return: axis shape is drawn on
    """
    
    if axis is None:
        fig = plt.figure(figsize = [3,3])
        axis = plt.gca()
    
    nv = verts.shape[0]    
    ndim = verts[0].shape[0]
    axis = draw_shape(verts, faces, color_list[0], axis)

    verts_new = []
    # get x (and y if ndim=2) displacements from eigenvectors and draw deformed shape
    for i in range(nv):
        if ndim == 2:
            verts_new.append([verts[i][0] + v2[i], verts[i][1] + v2[i + nv]])
        else:
            verts_new.append([verts[i][0] + v2[i]])
    verts_new = np.asarray(verts_new)
    axis = draw_shape(verts_new, faces, color_list[1], axis)

    # draw eigenvectors as arrows from initial to new positions
    for i in range(nv):

        x = verts[i][0]
        dx = v2[i]
        if ndim==2:
            y = verts[i][1]
            dy = v2[i+nv]
        else:
            y = 0
            dy = 0
        if dx!=0 or dy!=0:
            axis.arrow(x,y,dx,dy, head_width=0.2, fc='k')    
    axis.set_aspect('equal')
    
    return axis

def draw_init_modes(verts, faces, v, w):
    """Draws the raw eigenvectors generate by noraml mode analysis (before projecting things out).
    :param verts: mesh vertices
    :param faces: mesh faces
    :param v: eigenvectors representing normal modes of mesh
    :param w: eigenvalues representing frequencies of normal modes
    """
    
    # calculate number of spatial dimensions and number of points
    ndim = verts[0].shape[0]
    npts = verts.shape[0]

    # set up axes to plot all normal modes
    psize = 3
    fig, ax = plt.subplots(figsize=[psize*ndim,psize*npts], nrows=npts, ncols=ndim)

    # cycle through all normal modes (N = npts*ndim)
    for j in range(npts*ndim):

        # get plotting axis
        if ndim==2:
            axis = ax[int(j%npts),int(np.floor(j/npts))]
        else:
            axis = ax[int(j%npts)]

        # set plot title to mode frequency
        axis.title.set_text('w = '+str(np.round(w[j],2)))

        axis = draw_mode(verts, faces, v[:, j], axis)

