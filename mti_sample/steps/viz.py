import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
import os

def draw_mesh(mesh, save_flag=True):
    """Draws mesh if mesh is in 1D or 2D. If 3D, just plots vertices.
    :param mesh: mesh object (see mesh.py)
    """

    if mesh.ndims == 1:
        for pair in mesh.faces:
            x = [mesh.verts[pair[0]][0], mesh.verts[pair[1]][0]]
            y = np.zeros(len(x))
            plt.plot(x, y, marker='o', color='k')

    if mesh.ndims == 2:
        for pair in mesh.faces:
            x = [mesh.verts[pair[0]][0], mesh.verts[pair[1]][0]]
            y = [mesh.verts[pair[0]][1], mesh.verts[pair[1]][1]]
            plt.plot(x, y, marker='o', color='k')
            axis = plt.gca()
            axis.set_aspect('equal')

    if mesh.ndims == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for vert in mesh.verts:
            ax.scatter(vert[0], vert[1], vert[2], color='k')

    if save_flag:
        save_dir = 'data/viz/'+str(mesh.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir+'/mesh.pdf')
        plt.savefig(save_dir+'/mesh.png')


def draw_whist(nma, save_flag=True):
    """Draw histogram of eigenvalues (w2*m/k)
    :param nma: nma object containing eigenvalue parameter w
    """
    bins = np.linspace(-0.5,max(nma.w)+0.5, max(nma.w)+2)
    sb.distplot(nma.w, kde=False, bins=bins)
    plt.xlabel('Eigenvalues (w2*m/k)')
    plt.ylabel('Counts')

    if save_flag:
        save_dir = 'data/viz/'+str(nma.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir+'/whist.pdf')
        plt.savefig(save_dir+'/whist.png')


def draw_whist_multi(nma_list, save_flag=True):
    """Draw histogram of eigenvalues (w2*m/k)
    :param nma_list: list of nma objects each containing eigenvalue parameter w
    """

    for nma in nma_list:
        bins = np.linspace(-0.5,max(nma.w)+0.5, max(nma.w)+2)
        sb.distplot(nma.w, kde=False, bins=bins, label = nma.name)
    plt.xlabel('Eigenvalues (w2*m/k)')
    plt.ylabel('Counts')
    plt.legend()

    if save_flag:
        save_dir = 'data/viz/whist_comparisons/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fname = ''
        for nma in nma_list:
            fname+=nma.name+'-'

        plt.savefig(save_dir+fname[:-1]+'.pdf')
        plt.savefig(save_dir+fname[:-1]+'.png')