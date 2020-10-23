#!/usr/bin/env python3

"""
Input: Simplicial complex , k-cochains and percentage of missing data
Output: k-cochains where the percentage of missing data has been replaced by a placehold values
"""


import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from random import shuffle

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
