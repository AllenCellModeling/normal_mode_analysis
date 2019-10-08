from get_mesh_modes import *
import numpy as np
import math
import itertools
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb


# test model definitions: mass positions
model_verts = {
    '1D_2m': np.array([[0.], [1.] ]),
    '1D_3m': np.array([[0.,], [1.,], [2., ] ]),
    '3D_2m': np.array([[0., 0., 0.], [1., 0., 0.] ]),
    '2D_sq': np.array([[0.,0.], [0., 1.], [1.,1.], [1.,0.] ]),
    '2D_rt': np.array([[0.,0.], [0., 2.], [1.,2.], [1.,0.] ]),
    '2D_rt2': np.array([[0.,0.], [0., 1.], [0., 2.], [1.,2.], [1.,1.], [1.,0.] ]),
    '3D_cb': np.array([[0.,0.,0.], [0., 1.,0,], [1.,1.,0.], [1.,0.,0.],[0.,0.,1.], [0., 1.,1,], [1.,1.,1.], [1.,0.,1.] ]),
    '2D_tr': np.array([[1.,0.], [0.5, np.sqrt(3)/2.], [0.,0.], ]),
    '3D_th': np.array([[1.,0.,0.], [0.5, np.sqrt(3)/2.,0.], [0.,0.,0.], [np.sqrt(3)/2., np.sqrt(3)/2., 1.] ])
}

# test model definitions: mass connectivities
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


def nma_test_model(model, verts=None, faces=None):
    
    if model is not None:
        verts = 10*model_verts[model]
        faces = model_faces[model]
    
    hess = get_hessian_from_mesh(verts, faces)  
    w, v = np.linalg.eigh(hess)
    
    d = np.matmul(np.linalg.inv(v),np.matmul(hess,v))
    d[d<10**-5] = 0
    np.real(d)
    
    for indpair in list(itertools.combinations(range(v.shape[0]), 2)):
        a = indpair[0]
        b = indpair[1]
        c = np.dot(v[:,a], v[:,b])
        if np.abs(c)<10**-5: c = 0 
    # print the number of eigenvector pairs which are not normal        
    print('Number of eigenvectors which fail normality check: '+str(np.sum(c)))
    
    return verts, hess, w, v



def nma_polygon(r, N=100):

    def generate_circle_mesh(r, N):
        verts = [np.array((math.cos(2*np.pi/N*x)*r, math.sin(2*np.pi/N*x)*r)) for x in range(0,N)]
        faces = []
        for i in range(N-1):
            faces.append([i, i+1])
        faces.append([N-1,0])
        return verts, faces
    
    verts, faces = generate_circle_mesh(r, N)
    
    x = [verts[i][0] for i in range(len(verts))]
    y = [verts[i][1] for i in range(len(verts))]
    plt.scatter(x,y)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.tight_layout()
    
    verts, hess, w, v = nma_test_model(None, np.array(verts), faces)
    plt.figure()
    sb.distplot(w, kde=False, bins=20)
    draw_init_modes(verts, v, w)


def draw_mode(verts, v2, axis=None):
    
    if axis is None:
        plt.figure()
        axis = plt.gca()
        
    ndim = verts[0].shape[0]
        
    # get x (and y if ndim=2) coordinates for initial mass positions and draw initial shape
    x = []
    y = []
    for i in range(verts.shape[0]):
        x.extend([verts[i][0]])
        if ndim==2:
            y.extend([verts[i][1]])
    x.extend([verts[0][0]])
    if ndim==2:
        y.extend([verts[0][1]])
    else:
        y = np.zeros(len(x))    
    axis.plot(x,y)

    # get x (and y if ndim=2) displacements from eigenvectors and draw deformed shape
    dx = []
    dy = []
    for i in range(verts.shape[0]):
        dx.extend([v2[i]])
        if ndim==2:
            dy.extend([v2[i+verts.shape[0]]])
    dx.extend([v2[0]])
    if ndim==2:
        dy.extend([v2[0+verts.shape[0]]])
    else:
        dy= np.zeros(len(dx))

    x2 = [sum(xx) for xx in zip(x, dx)]
    y2 = [sum(yy) for yy in zip(y, dy)]
    axis.plot(x2,y2)
    axis.set_xlim(min(x2)-1,max(x2)+1)
    axis.set_ylim(min(y2)-1,max(y2)+1)
    axis.set_aspect('equal')

    # draw eigenvectors as arrows from initial to new positions
    for i in range(verts.shape[0]):

        x = verts[i][0]
        dx = v2[i]
        if ndim==2:
            y = verts[i][1]
            dy = v2[i+verts.shape[0]]
        else:
            y = 0
            dy = 0
        if dx!=0 or dy!=0:
            axis.arrow(x,y,dx,dy, head_width=0.2)


def draw_init_modes(verts, v, w):
    # calculate number of spatial dimensions and number of points
    ndim = verts[0].shape[0]
    npts = verts.shape[0]

    # set up axes to plot all normal modes
    psize = 5
    fig, ax = plt.subplots(figsize=[psize*ndim,psize*npts], nrows=npts, ncols=ndim)
    sb.set_palette(sb.color_palette("Set2"))

    # cycle through all normal modes (N = npts*ndim)
    for j in range(npts*ndim):

        # get plotting axis
        if ndim==2:
            axis = ax[int(j%npts),int(np.floor(j/npts))]
        else:
            axis = ax[int(j%npts)]

        # set plot title to mode frequency
        axis.title.set_text('w = '+str(np.round(w[j],2)))

        draw_mode(verts, v[:, j], axis)
