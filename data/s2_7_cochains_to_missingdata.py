#!/usr/bin/env python3

"""
Input: Simplicial complex , k-cochains and percentage of missing data
Output: k-cochains where the percentage of missing data has been replaced by a placehold values
"""


import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from random import shuffle

import time


def build_missing_values(simplices,percentage_missing_values,max_dim=10):

    """
    The functions randomly deletes a given percenatge of the values of simplices in each dimension
    of a simplicial complex.

    Parameters
    ----------

    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    percenatge_missing_values: integer
        Percentage of values missing

    max_dim: integer
        maximal dimension of the simplices to be considered.

    Returns
    ----------
        missing_values: list of dictionaries

        List of dictionaries, one per dimension d. The dictionary's keys are the missing d-simplices.
        The dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    """
    missing_values = [dict() for _ in range(max_dim+1)]
    for i in range(max_dim+1):
        simp=list(simplices[i].keys())
        l=int(np.ceil((len(simp)/100)*percentage_missing_values))
        simp_copy=np.copy(simp)
        shuffle(simp_copy)
        loss_simp = simp_copy[:l]
        for index,simplex in enumerate(loss_simp):
            dim=len(simplex)
            missing_values[i][simplex]=simplices[dim-1][simplex]
    return(missing_values)

###craete input cochain by substituing median values in unseen collaboration
def build_damaged_dataset(cochains,missing_values,function=np.median):

    """
    The function replaces the missing values in the dataset with a value inferred
    from the known data (eg the missing values are replaced buy the median or median
    or mean of the known values).

    Parameters
    ----------
    cochains: list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The
        dictionary's values are the k-cochains

    missing_values: list of dctionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the missing k-simplices. The
        dictionary's values are their indices

    function: callable
        missing values are replaced by the function of the known values, defaut median

        Returns
    ----------
        damaged_dataset: list of dictionaries

        List of dictionaries, one per dimension d. The dictionary's keys are the d-simplices.
        The dictionary's values are the d-cochains where the damaged portion has been replaced
        by the given function value.

    """
    ##Find median value
    max_dim=len(cochains)
    signal = np.copy(cochains)
    signal=np.array([np.array(list(signal[i].values())) for i in range(len(signal))])


    median_random=[]
    for dim in range(len(signal)):
        m=[signal[dim][j] for j in range(len(signal[dim]))]
        median_random.append(function(m))
        #print('Median is ',np.median(m))

    ## Create input usining median value for unknown values
    damaged_dataset = np.copy(cochains)
    for i in range(max_dim):
        simp=list(missing_values[i].keys())
        for index,simplex in enumerate(simp):
            dim=len(simplex)
            damaged_dataset[i][simplex]=median_random[dim-1]
    return(damaged_dataset)


###Inices and values of the "seen" simplices
def built_known_values(missing_values,simplices):
    """
    The functions return the not missing simplices and cochains in each dimension


    Parameters
    ----------
    missing_values: list of dictionaries
        List of dictionaries, one per dimension d. The dictionary's keys are the missing d-simplices.
        The dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the 0-simplices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.


    Returns
    ----------
    known_values: list of dictionaries
        List of dictionaries, one per dimension d. The dictionary's keys are not missing d-simplices.
        The dictionary's values are their cochains.

    """
    max_dim=len(simplices)

    known_values = [dict() for _ in range(max_dim+1)]
    for i in range(max_dim):
        real_simp=list(set(simplices[i].keys())-set(missing_values[i].keys()))
        for index,simplex in enumerate(real_simp):
            dim=len(simplex)
            known_values[i][simplex]=simplices[dim-1][simplex]

    return(known_values)


if __name__ == '__main__':
    start = time.time()
    def timeit(name):
        print('wall time ({}): {:.0f}s'.format(name, time.time() - start))
    starting_node=150250
    percentage_missing_values=30

    cochains = np.load(f's2_3_collaboration_complex/{starting_node}_cochains.npy')
    simplices = np.load(f's2_3_collaboration_complex/{starting_node}_simplices.npy')

    missing_values=build_missing_values(simplices,percentage_missing_values=30,max_dim=10)
    damaged_dataset=build_damaged_dataset(cochains,missing_values,function=np.median)
    known_values=built_known_values(missing_values,simplices)

    timeit('process')
    np.save(f's2_3_collaboration_complex/{starting_node}_percentage_{percentage_missing_values}_missing_values.npy', missing_values)
    np.save(f's2_3_collaboration_complex/{starting_node}_percentage_{percentage_missing_values}_input_damaged.npy', damaged_dataset)
    np.save(f's2_3_collaboration_complex/{starting_node}_percentage_{percentage_missing_values}_known_values.npy', known_values)
    timeit('total')
