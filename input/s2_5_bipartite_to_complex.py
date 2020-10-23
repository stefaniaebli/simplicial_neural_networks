#!/usr/bin/env python3

"""
Input: collaboration bipartite graph X-Y and weights on X.
Output: collaboration simplicial complex and value of each collaboration.
(Each collaboration is represented as a simplex.)
"""

import os
import contextlib
import time

import numpy as np
import pandas as pd
import gudhi
from scipy import sparse
from scipy.sparse import coo_matrix
from random import shuffle




def bipart2simplex(bipartite,weights_x,indices_x=None,dimension=3):
    """Build a Gudhi Simplex Tree from the bipartite graph X-Y by projection on Y and extract the
    features corresponding to maximal dimensional simplices.
    Parameters
    ----------
    bipartite : scipy sparse matrix
        bipartite collaboration graph X-Y
    weights_x : ndarray
        Array of size bipartite.shape[0], containing the weights on the node of X
    indices_x = array
        Array of the indices of the X nodes to restrict to, default: all nodes of X
    dimension: int
        maximal dimension of the simplicial complex = maximal number of individuals collaborating.
    Returns
    -------
    simplex_tree:
        Gudhi simplex tree.
    signals_top:
        Features for every maximal dimensional simplex.
    """
    signals_top = [dict() for _ in range(dimension+1)]
    simplex_tree =gudhi.SimplexTree()
    if np.all(indices_x)==None:
        indices_x=np.arange(bipartite.shape[0])

    Al=bipartite.tolil()
    Al.rows[indices_x]
    for j,authors in enumerate(Al.rows[indices_x]):
        if len(authors)<=dimension+1:
            k = len(authors)
            simplex_tree.insert(authors)
            signals_top[k-1].setdefault(frozenset(authors),[]).append(weights_x[indices_x][j])
        else:
            continue

    return(simplex_tree,signals_top)


def extract_simplices(simplex_tree):
    """Create a list of simplices from a gudhi simplex tree."""
    simplices = [dict() for _ in range(simplex_tree.dimension()+1)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        k = len(simplex)
        simplices[k-1][frozenset(simplex)] = len(simplices[k-1])
    return simplices


def build_cochains(simplex_tree,signals_top,function=np.sum):
    """Build the k-cochains using the weights on X (form the X-Y bipartite graph)
     and a chosen aggregating function. Features are aggregated by the provided functions.
     The function takes as input a list of values  and must return a single number.

    Parameters
    ----------
    simplex_tree :
        Gudhi simplex tree
    signals_top : ndarray
        Features for every maximal dimensional simplex = weights on the nodes of X (from bipartite graph X-Y)
    function : callable
        Functions that will aggregate the features to build the k-coachains, default=np.sum

    Returns
    -------
    signals :
        k-cochains = List of dictionaries of simplices of order k and their collaboration
        value.
    signals_top:
        Features for every maximal dimensional simplex
    """
    signal = [dict() for _ in range(simplex_tree.dimension()+1)]
    for d in range(len(signals_top)-1, -1,-1):
        for simplex, values in signals_top[d].items():
            st=gudhi.SimplexTree()
            st.insert(simplex)
            for face, _ in st.get_skeleton(st.dimension()):
                face=frozenset(face)
                signal[len(face)-1].setdefault(face,[]).extend(signals_top[d][simplex])

    for d in range(len(signals_top)-1, -1,-1):
        for simplex, values in signals_top[d].items():
            st=gudhi.SimplexTree()
            st.insert(simplex)
            for face, _ in st.get_skeleton(st.dimension()):
                face=frozenset(face)
                value=np.array(signal[len(face)-1][face])
                signal[len(face)-1][face]=int(function(value))##Choose propagation function
               # signal[len(face)-1][face]=np.mean(value[value>0])
    return signal,signals_top


def bipart2simpcochain(bipartite,weights_x,indices_x=None,function=np.sum,dimension=3):
    """From a collaboration bipartite graph X-Y and its weights on X to a
    collaboration simplicial complex and its collaboration values on the
    simplices.

    Parameters
    ----------
    bipartite : scipy sparse matrix
        Sparse matrix representing the collaboration bipartite graph X-Y.
    weights_x : ndarray
        Array of size bipartite.shape[0], containing the weights on the node of X
    function : callable
        Functions that will aggregate the features to build the k-coachains, default=np.sum
    indices_x : array
        Array of the indices of the X nodes to restrict to, default = all nodes of X
    dimension : int
        Maximal dimension of the simplicial complex.

    Returns
    -------
    simplices:
        List of dictionaries of simplices of order k and their indices.
    cochains:
        List of dictionaries of simplices of order k and their collaboration
        value.
    signals_top:
        Features for every maximal dimensional simplex
    """
    st,v=bipart2simplex(bipartite,weights_x,indices_x,dimension)
    simplices=extract_simplices(st)
    cochains,signals_top=build_cochains(st,v,function)
    return simplices, cochains,signals_top



def test():
    r"""Test the transformation of a bipartite graph to a collaboration complex."""
    biadjacency = coo_matrix([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
    ])
    number_citations =np.array([100,50,10,4])
    indices=np.arange(biadjacency.shape[0])
    simplices, cochains,signals_top = bipart2simpcochain(biadjacency, number_citations,function=np.sum )

    cochains_true = [
        {   frozenset({0}): 100+50+10,
            frozenset({1}): 100+50,
            frozenset({2}): 100+4,
            frozenset({3}): 10+4
        },
        {   frozenset({0, 1}): 100+50,
            frozenset({0, 2}): 100,
            frozenset({1, 2}): 100,
            frozenset({0, 3}): 10,
            frozenset({2, 3}): 4
        },

        {
            frozenset({0, 1, 2}): 100
        }
    ]
    simplices_true = [
        {
            frozenset({0}): 0,
            frozenset({1}): 1,
            frozenset({2}): 2,
            frozenset({3}): 3
        },
        {
            frozenset({0, 1}): 0,
            frozenset({0, 2}): 1,
            frozenset({0, 3}): 2,
            frozenset({1, 2}): 3,
            frozenset({2, 3}): 4
        },
        {
            frozenset({0, 1, 2}): 0}
    ]
    assert cochains == cochains_true
    assert simplices == simplices_true


if __name__ == '__main__':
    test()

    starting_node=150250
    adjacency = scipy.sparse.load_npz('./preproces/paper_author_biadjacency.npz')
    papers = pd.read_csv('./data/s2_processed/papers.csv', index_col=0)
    citations=np.array(papers['citations_2019'])
    downsample_papers=np.load('./input/downsampled_'+str(starting_node)+'.npy')


    def timeit(name):
        print('wall time ({}): {:.0f}s'.format(name, time.time() - start))


    simplices, cochains, signals_top = bipart2simpcochain(adijacency, citations, indices_x=downsample_papers, dimension=10)
    timeit('process')
    np.save('./input/authors_collaboration_cochains_'+str(starting_node)+'.npy', cochains)
    np.save('./input/authors_collaboration_simplices_'+str(starting_node)+'.npy', simplices)
    timeit('total')
