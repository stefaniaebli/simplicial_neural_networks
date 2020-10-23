#!/usr/bin/env python3

"""
Input: collaboration bipartite graph and value of each paper.
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


def load(columns):
    biadjacency = sparse.load_npz('s2_processed/paper_author_biadjacency.npz')
    # papers = np.load('s2_processed/papers_features.npy')
    papers = pd.read_csv('s2_processed/papers.csv', index_col=0)
    print('loading:')
    print('  bipartite: {:,} papers, {:,} authors, {:,} edges'.format(
        *biadjacency.shape, biadjacency.nnz))
    print('  paper features: {:,} papers, {:,} features'.format(*papers.shape))
    print('                  keeping {} features'.format(len(columns)))
    return biadjacency, papers[columns]


def bipart2simplex(biadjacency, papers, dimension):
    """Build a Gudhi Simplex Tree from the bipartite graph and extract the
    features corresponding to top simplices.

    Parameters
    ----------
    dimension: int
        maximal dimension of the simplicial complex = maximal number of individuals collaborating.

    Returns
    -------
    simplex_tree:
        Gudhi simplex tree.
    signals_top:
        Features for every top simplices.
    """

    signals_top = [dict() for _ in range(dimension)]
    simplex_tree = gudhi.SimplexTree()
    biadjacency = biadjacency.tolil()

    for (_, features), authors in zip(papers.iterrows(), biadjacency.rows):
        dim = len(authors) - 1
        if dim < dimension:
            simplex_tree.insert(authors)
            # The same simplex might appear multiple times, hence we append.
            signals_top[dim].setdefault(frozenset(authors), []).append(features.values)
        else:
            continue

    print('extracting top simplices:')
    for dim, signal in enumerate(signals_top):
        duplicates = sum([len(s)-1 for s in signal.values()])
        print(f'  {len(signal)} top {dim}-simplices ({duplicates} duplicates)')

    return simplex_tree, signals_top


def extract_simplices(simplex_tree):
    """Create a list of simplices from a gudhi simplex tree."""
    simplices = [dict() for _ in range(simplex_tree.dimension()+1)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        dim = len(simplex) - 1
        simplices[dim][frozenset(simplex)] = len(simplices[dim])
    return simplices


def build_cochains(signals_top, concatenations):
    """Build the cochains by propagating down to faces the values of the real
    collaborations.

    Features are aggregated by the provided functions. The function takes as
    input a list of values (one per intrinsic or inherited value) and must
    return a single number. It should take care of special values, for example
    masked data for inpainting.

    Parameters
    ----------
    signals_top:
        Features for every top simplices.
    concatenations:
        List of functions that will concatenate the feature vectors.

    Returns
    -------
    signals:
        Features for every simplices, propagated and aggregated.
    """

    signals = [dict() for _ in range(len(signals_top))]

    # Propagate the values down from top simplices.
    # TODO: we might want to down-propagate with another function than copy.
    # TODO: could be combined with bipart2simplex.
    for dim in range(len(signals_top)-1, -1, -1):
        for simplex, features in signals_top[dim].items():
            st = gudhi.SimplexTree()
            st.insert(simplex)
            for face, _ in st.get_skeleton(st.dimension()):
                face = frozenset(face)
                signals[len(face)-1].setdefault(face, []).extend(features)

    # Concatenate the intrinsic and inherited features.
    for dim in range(len(signals)):
        for simplex, features in signals[dim].items():
            features = np.array(features).T
            features = [f(v) for f, v in zip(concatenations, features)]
            signals[dim][simplex] = features

    return signals


def bipart2simpcochain(biadjacency, papers, concatenations, dimension=3):
    """From a collaboration bipartite graph and its collaboration values to a
    collaboration simplicial complex and its collaboration values on the
    simplices.

    Parameters
    ----------
    biadjacency:
        Sparse matrix representing the collaboration bipartite graph.
    values_colab: ndarray
        Array of size bipartite.shape[0], containing the values of the collaboration.
    function: callable
        Function propagating down the signal. Default: np.sum.
    dimension: int
        Maximal dimension of the simplicial complex.

    Returns
    -------
    simplices:
        List of dictionaries of simplices of order k and their indices.
    cochains:
        List of dictionaries of simplices of order k and their collaboration
        value.
    """
    print(f'building collaboration complex of dimension at most {dimension}')
    print(f'  (collaborations of more than {dimension} authors are discarded)')
    simplex_tree, signals_top = bipart2simplex(biadjacency, papers, dimension)
    simplices = extract_simplices(simplex_tree)
    cochains = build_cochains(signals_top, concatenations)
    return simplices, cochains


def save(simplices, cochains):
    print('saving:')
    sizes = [len(s) for s in simplices]
    for k, size in enumerate(sizes):
        print(f'  {size:,} {k}-simplices')
    print('  {:,} simplices in total'.format(sum(sizes)))
    np.save('s2_processed/authors_collaboration_cochains.npy', cochains)
    np.save('s2_processed/authors_collaboration_simplices.npy', simplices)


def test():
    r"""Test the transformation of a bipartite graph to a collaboration complex."""
    biadjacency = sparse.csr_matrix([
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 1, 1],
    ])
    papers = pd.DataFrame([[3, 2, -7, 4], [3, 2, -7, 4]]).T
    concatenations = [sum, lambda x: sum(abs(x))]

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            simplices, cochains = bipart2simpcochain(
                    biadjacency, papers, concatenations)

    cochains_true = [
        {
            frozenset([0]): [4, 4],
            frozenset([2]): [4 + 3 - 7 + 2, 16],
            frozenset([3]): [4 + 3, 7],
        },
        {
            frozenset([0, 2]): [4, 4],
            frozenset([0, 3]): [4, 4],
            frozenset([2, 3]): [4 + 3, 7],
        },
        {
            frozenset([0, 2, 3]): [4, 4],
        },
    ]
    simplices_true = [
        {
            frozenset([0]): 0,
            frozenset([2]): 1,
            frozenset([3]): 2,
        },
        {
            frozenset([0, 2]): 0,
            frozenset([0, 3]): 1,
            frozenset([2, 3]): 2,
        },
        {
            frozenset([0, 2, 3]): 0,
        },
    ]
    assert cochains == cochains_true
    assert simplices == simplices_true


if __name__ == '__main__':
    test()

    # List of features to keep, with the corresponding aggregation function.
    # Python functions are faster than numpy as we aggregate over small lists.
    concatenations = {
        'citations_1994': sum,
        'citations_1999': sum,
        'citations_2004': sum,
        'citations_2009': sum,
        'citations_2014': sum,
        'citations_2019': sum,
        'references': sum,
        'year': lambda x: sum(x) / len(x),
    }

    start = time.time()
    def timeit(name):
        print('wall time ({}): {:.0f}s'.format(name, time.time() - start))

    biadjacency, papers = load(concatenations.keys())
    timeit('load')
    simplices, cochains = bipart2simpcochain(biadjacency, papers, concatenations.values(), dimension=10)
    timeit('process')
    save(simplices, cochains)
    timeit('total')

# TODO: drop gudhi and use itertools.combinations({2,3,4,5}, 2) ?
