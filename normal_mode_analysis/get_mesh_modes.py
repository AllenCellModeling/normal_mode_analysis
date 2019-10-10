import numpy as np
import itertools
from scipy.sparse.linalg import eigsh


def get_hessian_from_mesh(verts, faces):
    """Find Hessian for mesh defined by input vertices and faces."""
    
    # create hessian matrix of size 3N, allowing each pair of points to have x,y,z components
    ndim = int(verts[0].shape[0])
    npts = int(verts.shape[0])
    hess = np.zeros([ndim*verts.shape[0], ndim*verts.shape[0]])
    
    # get all unique pairs of points that are connected in the spring network
    all_pairs = []
    for face in faces:
        for pair in list(itertools.combinations(face, 2)):
            all_pairs.append(pair)
    
    # cycle through pairs of x,y,z coordinates
    for indpair in list(itertools.combinations_with_replacement(range(ndim), 2)):
        ind1 = indpair[0]
        ind2 = indpair[1]
        
        # cycle through pairs of connected points in mesh
        for ptpair in all_pairs:
            i = ptpair[0]
            j = ptpair[1]
            
            # fill in off-diagonal hessian elements
            if (i != j):
                xyz1 = verts[i]
                xyz2 = verts[j]
                R = np.linalg.norm(xyz1-xyz2)
                val = -(xyz2[ind2] - xyz1[ind2])*(xyz2[ind1] - xyz1[ind1])/(R**2)
                hess[npts*ind1 + i, npts*ind2 + j] = val
                hess[npts*ind2 + j, npts*ind1 + i] = val
                hess[npts*ind1 + j, npts*ind2 + i] = val
                hess[npts*ind2 + i, npts*ind1 + j] = val
                
    # fill in diagonal and sub-block diagonal elements of hessian
    for indpair in list(itertools.combinations_with_replacement(range(ndim), 2)):
        ind1 = indpair[0]
        ind2 = indpair[1]
        for pt in range(npts):
            hess[ind1*npts+pt][ind2*npts+pt] = -np.sum(hess[ind1*npts+pt][ind2*npts:(ind2+1)*npts])
            if ind1!=ind2:
                hess[ind2*npts+pt][ind1*npts+pt] = hess[ind1*npts+pt][ind2*npts+pt]
            
    return hess
    
    
def get_eigs_from_mesh(verts, faces, save_flag = True):
    """Get eigenvalues and eigenvectors of hessian, calculated from vertices and faces defining mesh."""

    # calculate hessian from faces and vertices as initial conditions with zero potential energy (rest shape)
    mat = get_hessian(verts, faces)
    np.save('eigs/hessian', mat)
    
    # use solver to get eigenvalues (w) and eigenvectors (v)
    w, v = np.linalg.eigh(mat)
    
    if save_flag:
        np.save('eigs/eigvals_'+np.str(verts.shape[0]), w)
        np.save('eigs/eigvecs_'+np.str(verts.shape[0]), v)
    
    return mat, w, v