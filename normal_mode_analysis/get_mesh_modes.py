import numpy as np
from stl import mesh
import pandas as pd
import itertools
from scipy.sparse.linalg import eigsh


def get_hessian_from_mesh(verts, faces):
    """Find Hessian for mesh defined by input vertices and faces.
    :param verts: vertices of mesh
    :param faces: faces of mesh
    :return: hessian matrix describing connectivity of mesh
    """
    
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
                if R==0:
                    val = 0
                    print('Identical vertices found at indices '+str(i)+' and '+str(j))
                else:
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
    
    
def get_eigs_from_mesh(verts, faces, save_flag = False, fname = None):
    """Get eigenvalues and eigenvectors of hessian, calculated from vertices and faces defining mesh.
    :param verts: vertices of mesh
    :param faces: faces of mesh
    :param save_flag: flag to save eigenvalues and eigenvectors to file
    :return: hessian, and its eigenvalues (w) and eigenvectors (v) (v[:,i] correcsponds to w[i])
    """

    # calculate hessian from faces and vertices as initial conditions with zero potential energy (rest shape)
    mat = get_hessian_from_mesh(verts, faces)
    if save_flag:
        np.save('nucleus_nma/hessian_'+fname, mat)
    
    # use solver to get eigenvalues (w) and eigenvectors (v)
    w, v = np.linalg.eigh(mat)
    
    if save_flag:
        np.save('nucleus_nma/eigvals_'+fname, w)
        np.save('nucleus_nma/eigvecs_'+fname, v)
    
    return mat, w, v


def process_all_eigvecs(res, shape='nuc', v=None, w=None):
    
    # set path to look for nuclear mesh info
    if shape=='nuc':
        verts = np.load('nucleus_mesh_data/sample_trimeshes_from_blair/mean_nuc_mesh_uniform_'+str(res)+'_vertices.npy')
    else:
        verts = np.load('nucleus_mesh_data/sample_trimeshes_from_blair/icosphere_'+str(res)+'_vertices.npy')


    if v is None:
        # load normal mode analysis results for this mesh
        v = np.load('nucleus_nma/eigvecs_nuc_mesh_'+str(res)+'.npy')
        w = np.load('nucleus_nma/eigvals_nuc_mesh_'+str(res)+'.npy')
    
    # get shape parameters
    nverts = verts.shape[0]
    ndim = verts.shape[1]
    nmodes = v.shape[1]
    
    # fill dataframe with a row for each mode
    # each row contains the frequency, and list of eigenvectors and their magnitudes
    # these vecotrs and magnitudes are listed in the same order as the stl mesh vertex order
    eigeninfo = pd.DataFrame(columns=['vecs', 'mags'])
    for j in range(nmodes):
        vecs = [[v[i,j], v[i+nverts,j], v[i+2*nverts,j]] for i in range(nverts)]
        mags = [np.linalg.norm(vecs[i], axis=0) for i in range(nverts)]
        mags /= max(mags)
        eigeninfo = eigeninfo.append({'vecs': vecs, 'mags': mags, 'mode w':w[j]}, ignore_index=True)
    
    # save and return dataframe
    if shape=='nuc':
        eigeninfo.to_pickle('nucleus_nma/mode_table_nuc_mesh_'+str(res)+'.pickle')
    else:
        eigeninfo.to_pickle('nucleus_nma/mode_table_ico_mesh_'+str(res)+'.pickle')
    return eigeninfo