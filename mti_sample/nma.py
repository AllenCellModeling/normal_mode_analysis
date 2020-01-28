import numpy as np
import itertools
import os
from scipy.sparse.linalg import eigsh


class NMA():

    def __init__(self, mesh, name=None, save_flag=False,):
        """Runs normal mode analysis on input mesh by calculating Hessian and its eigenvalues/vectors.
        :param mesh: Mesh object (from raw.py)
        :param save_flag: boolean flag setting whether to save NMA results
        :param model_dir: string giving name of directory to save NMA results in; None if save_flag is False
        """
        self.mesh = mesh
        self.hess = self.get_hessian_from_mesh(self.mesh)
        self.w, self.v = self.get_eigs_from_mesh(self.hess, save_flag, name)
        self.name = name


    def get_hessian_from_mesh(self, mesh):
        """Find Hessian for mesh defined by input vertices and faces.
        :param mesh: mesh object containing vertices and faces
        :return: Hessian matrix describing connectivity of mesh
        """

        # create hessian matrix of size 3N, allowing each pair of points to have x,y,z components
        hess = np.zeros([mesh.ndims*mesh.npts, mesh.ndims*mesh.npts])
        
        # get all unique pairs of points that are connected in the spring network
        edges = []
        for face in mesh.faces:
            for pair in list(itertools.combinations(face, 2)):
                edges.append(pair)
        
        # cycle through pairs of x,y,z coordinates
        ind_pairs = list(itertools.combinations_with_replacement(range(mesh.ndims), 2))
        for ind_pair in ind_pairs:
            ind1, ind2 = ind_pair
            
            # cycle through pairs of connected points in mesh
            for edge in edges:
                i, j = edge
                
                # fill in off-diagonal hessian elements
                if (i != j):
                    xyz1 = mesh.verts[i]
                    xyz2 = mesh.verts[j]
                    R = np.linalg.norm(xyz1-xyz2)
                    val = -(xyz2[ind2] - xyz1[ind2])*(xyz2[ind1] - xyz1[ind1])/(R**2)

                    hess[mesh.npts*ind1 + i, mesh.npts*ind2 + j] = val
                    hess[mesh.npts*ind2 + j, mesh.npts*ind1 + i] = val
                    hess[mesh.npts*ind1 + j, mesh.npts*ind2 + i] = val
                    hess[mesh.npts*ind2 + i, mesh.npts*ind1 + j] = val

        # fill in diagonal and sub-block diagonal elements of hessian
        for ind_pair in ind_pairs:
            ind1, ind2 = ind_pair
            for pt in range(mesh.npts):
                hess[ind1*mesh.npts+pt][ind2*mesh.npts+pt] = -np.sum(hess[ind1*mesh.npts+pt][ind2*mesh.npts:(ind2+1)*mesh.npts])
                if ind1!=ind2:
                    hess[ind2*mesh.npts+pt][ind1*mesh.npts+pt] = hess[ind1*mesh.npts+pt][ind2*mesh.npts+pt]

        return hess


    def get_eigs_from_mesh(self, hess, save_flag = False, model_dir = None):
        """Get eigenvalues and eigenvectors of hessian.
        :param hess: hessian for mesh
        :param save_flag: flag to save hessian, eigenvalues and eigenvectors to file
        :return: hessian eigenvalues (w) and eigenvectors (v) (v[:,i] correcsponds to w[i])
        """

        # use solver to get eigenvalues (w) and eigenvectors (v)
        w, v = np.linalg.eigh(hess)

        if save_flag:
            save_dir = 'nma/'+model_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(save_dir+'/hessian', hess)
            np.save(save_dir+'/eigvals', w)
            np.save(save_dir+'/eigvecs', v)

        return w, v
