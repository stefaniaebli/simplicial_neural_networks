
#!/usr/bin/env python3

"""
Input: Simplicial complex
Output: k-order Laplacians
"""


import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from random import shuffle

import time

def load():
    simplices = np.load('s2_processed/authors_collaboration_simplices.npy')
    cochains = np.load('s2_processed/authors_collaboration_cochains.npy')
    print('loading:')
    print(f'  {len(cochains)}-dimensional simplicial complex')
    sizes = np.array([len(s) for s in simplices])
    print('  {:,} simplices in total'.format(np.sum(sizes)))
    return simplices, cochains


def build_boundaries(simplices):
    """Build the boundary operators from a list of simplices.

    Parameters
    ----------
    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    Returns
    -------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension.
    """
    boundaries = list()
    for d in range(1, len(simplices)):
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1)**i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[d-1][face])
        assert len(values) == (d+1) * len(simplices[d])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                                     dtype=np.float32,
                                     shape=(len(simplices[d-1]), len(simplices[d])))
        boundaries.append(boundary)
    return boundaries


def build_laplacians(boundaries):
    """Build the Laplacian operators from the boundary operators.

    Parameters
    ----------
    boundaries: list of sparse matrices
       List of boundary operators, one per dimension.

    Returns
    -------
    laplacians: list of sparse matrices
       List of Laplacian operators, one per dimension.
    """
    laplacians = list()
    up = coo_matrix(boundaries[0] @ boundaries[0].T)
    laplacians.append(up)
    for d in range(len(boundaries)-1):
        down = boundaries[d].T @ boundaries[d]
        up = boundaries[d+1] @ boundaries[d+1].T
        laplacians.append(coo_matrix(down + up))
    down = boundaries[-1].T @ boundaries[-1]
    laplacians.append(coo_matrix(down))
    return laplacians





"CONTINUE: TEST EXAMPLE AND BUILD SEEN AND UNSEEN DATA"




##randomly delete a percenatge of the values of simplices in each dimension
#mask random is a dictionary containing the unseen collaboartion
def random_mask(not_see,simplices,max_dim=10):

    seen_percentage=not_see ## percentage of unseen valued
    mask_random = [dict() for _ in range(max_dim+1)]
    for i in range(max_dim+1):
        simp=list(simplices[i].keys())
        l=int(np.ceil((len(simp)/100)*seen_percentage))
        simp_copy=np.copy(simp)
        shuffle(simp_copy)
        loss_simp = simp_copy[:l]
        for index,simplex in enumerate(loss_simp):
            dim=len(simplex)
            mask_random[i][simplex]=simplices[dim-1][simplex]
    return(mask_random)

###craete input cochain by substituing median values in unseen collaboration
def random_input_cochain(signal_target,mask_random):
    ##Find median value
    max_dim=len(signal_target)
    signal = np.copy(signal_target)
    signal=np.array([np.array(list(signal[i].values())) for i in range(len(signal))])

    complmasks = [np.setdiff1d(np.arange(0, len(signal[d])), mask_random[d]) for d in range(len(signal))]

    median_random=[]
    for dim in range(len(signal)):
        m=[signal[dim][j] for j in range(len(signal[dim]))]
        median_random.append(np.median(m))
        #print('Median is ',np.median(m))

    ## Create input usining median value for unknown values
    random_input = np.copy(signal_target)
    for i in range(max_dim):
        simp=list(mask_random[i].keys())
        for index,simplex in enumerate(simp):
            dim=len(simplex)
            random_input[i][simplex]=median_random[dim-1]
    return(random_input)


###Inices and values of the "seen" simplices
def mask_random_loss(not_see,mask_random,simplices):
    seen_percentage=not_see ## percentage of unseen valued considerd by the loss
    max_dim=len(simplices)

    mask_random_loss_final = [dict() for _ in range(max_dim+1)]
    for i in range(max_dim):
        real_simp=list(set(simplices[i].keys())-set(mask_random[i].keys()))
        for index,simplex in enumerate(real_simp):
            dim=len(simplex)
            mask_random_loss_final[i][simplex]=simplices[dim-1][simplex]

        #mask_random_loss_final[i].update(mask_random_loss[i])
    return(mask_random_loss_final)
