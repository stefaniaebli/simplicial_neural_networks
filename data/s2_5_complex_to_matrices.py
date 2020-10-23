#!/usr/bin/env python3

import time

import numpy as np
from scipy import sparse


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

    for dim in range(1, len(simplices)):

        idx_simplices, idx_faces, values = [], [], []

        for simplex, idx_simplex in simplices[dim].items():
            for i, left_out in enumerate(np.sort(simplex)):
                idx_simplices.append(idx_simplex)
                values.append((-1)**i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[dim-1][face])

        assert len(values) == (dim+1) * len(simplices[dim])
        boundary = sparse.csr_matrix((values, (idx_faces, idx_simplices)),
                                     dtype=np.float32,
                                     shape=(len(simplices[dim-1]), len(simplices[dim])))
        boundaries.append(boundary)

    for dim, boundary in enumerate(boundaries):
        # The number of faces of a simplex is dictated by its dimensionality.
        assert boundary.nnz == boundary.shape[1] * (dim+2)

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
    up = boundaries[0] @ boundaries[0].T
    laplacians.append(up)
    for dim in range(len(boundaries)-1):
        down = boundaries[dim].T @ boundaries[dim]
        up = boundaries[dim+1] @ boundaries[dim+1].T
        laplacians.append(down + up)
    down = boundaries[-1].T @ boundaries[-1]
    laplacians.append(down)
    return laplacians


def build_features(simplices, cochains):
    r"""Build feature matrices from cochains."""
    n_features = len(next(iter(cochains[0].values())))
    features = [np.empty((len(cochain), n_features)) for cochain in cochains]
    for dim in range(len(cochains)):
        for simplex, cochain in cochains[dim].items():
            idx = simplices[dim][simplex]
            features[dim][idx] = np.array(cochain)
    return features


def save(boundaries, laplacians, features):
    base = 's2_processed/authors_collaboration'
    print('saving:')

    def save_sparse_matrices(matrices, name):
        arrays = dict()
        for dim, matrix in enumerate(matrices):
            arrays[f'{dim}_data'] = matrix.data
            arrays[f'{dim}_indices'] = matrix.indices
            arrays[f'{dim}_indptr'] = matrix.indptr
            arrays[f'{dim}_shape'] = matrix.shape
        print(f'  {len(arrays) // 4} {name}')
        np.savez(f'{base}_{name}.npz', **arrays)

    save_sparse_matrices(boundaries, 'boundaries')
    save_sparse_matrices(laplacians, 'laplacians')

    arrays = dict()
    for dim, features in enumerate(features):
        arrays[f'{dim}_features'] = features
    print(f'  {len(arrays)} feature matrices')
    np.savez(f'{base}_features.npz', **arrays)


def test():
    r"""Test the construction of the boundary and Laplacian operators.

    Simplicial complex and ground truths from "Control Using Higher Order
    Laplacians in Network Topologies", figure 3.
    """
    simplices = [
        {
            frozenset([0]): 0,
            frozenset([1]): 1,
            frozenset([2]): 2,
            frozenset([3]): 3,
            frozenset([4]): 4,
        },
        {
            frozenset([0, 1]): 0,
            frozenset([0, 2]): 1,
            frozenset([1, 2]): 2,
            frozenset([1, 3]): 3,
            frozenset([2, 3]): 4,
            frozenset([2, 4]): 5,
            frozenset([3, 4]): 6,
        },
        {
            frozenset([1, 2, 3]): 0,
        },
    ]

    boundaries_true = [
        [
            [-1, -1, +0, +0, +0, +0, +0],
            [+1, +0, -1, -1, +0, +0, +0],
            [+0, +1, +1, +0, -1, -1, +0],
            [+0, +0, +0, +1, +1, +0, -1],
            [+0, +0, +0, +0, +0, +1, +1],
        ],
        [
            [+0],
            [+0],
            [+1],
            [-1],
            [+1],
            [+0],
            [+0],
        ],
    ]

    laplacians_true = [
        [
            [+2, -1, -1, +0, +0],
            [-1, +3, -1, -1, +0],
            [-1, -1, +4, -1, -1],
            [+0, -1, -1, +3, -1],
            [+0, +0, -1, -1, +2],
        ],
        [
            [+2, +1, -1, -1, +0, +0, +0],
            [+1, +2, +1, +0, -1, -1, +0],
            [-1, +1, +3, +0, +0, -1, +0],
            [-1, +0, +0, +3, +0, +0, -1],
            [+0, -1, +0, +0, +3, +1, -1],
            [+0, -1, -1, +0, +1, +2, +1],
            [+0, +0, +0, -1, -1, +1, +2],
        ],
        [[3]],
    ]

    boundaries = build_boundaries(simplices)
    laplacians = build_laplacians(boundaries)

    for boundary, boundary_true in zip(boundaries, boundaries_true):
        np.testing.assert_allclose(boundary.toarray(), boundary_true)

    for laplacian, laplacian_true in zip(laplacians, laplacians_true):
        np.testing.assert_allclose(laplacian.toarray(), laplacian_true)


def test_features():
    cochains = [
        {
            frozenset([0]): [4, 4],
            frozenset([3]): [1, 2],
            frozenset([2]): [2, 9],
        },
        {
            frozenset([0, 2]): [2, -3],
            frozenset([2, 3]): [5, -1],
        },
    ]
    simplices = [
        {
            frozenset([2]): 1,
            frozenset([3]): 2,
            frozenset([0]): 0,
        },
        {
            frozenset([2, 3]): 1,
            frozenset([0, 2]): 0,
        },
    ]
    features = build_features(simplices, cochains)
    features_true = [
        np.array([
            [4, 4],
            [2, 9],
            [1, 2],
        ]),
        np.array([
            [2, -3],
            [5, -1],
        ]),
    ]
    np.testing.assert_equal(features[0], features_true[0])
    np.testing.assert_equal(features[1], features_true[1])


def load_matrices():
    r"""Return the preprocessed boundary, Laplacian, and feature matrices."""

    def load_sparse_matrices(arrays):
        matrices = list()
        for dim in range(len(arrays) // 4):
            data = arrays[f'{dim}_data']
            indices = arrays[f'{dim}_indices']
            indptr = arrays[f'{dim}_indptr']
            shape = arrays[f'{dim}_shape']
            matrix = sparse.csr_matrix((data, indices, indptr), shape)
            matrices.append(matrix)
        return matrices

    base = 's2_processed/authors_collaboration'

    with np.load(f'{base}_boundaries.npz') as arrays:
        boundaries = load_sparse_matrices(arrays)

    with np.load(f'{base}_laplacians.npz') as arrays:
        laplacians = load_sparse_matrices(arrays)

    with np.load(f'{base}_features.npz') as arrays:
        features = [array for array in arrays.values()]

    return boundaries, laplacians, features


if __name__ == '__main__':

    test()
    test_features()

    start = time.time()
    def timeit(name):
        print('wall time ({}): {:.0f}s'.format(name, time.time() - start))

    simplices, cochains = load()
    timeit('load')
    boundaries = build_boundaries(simplices)
    timeit('boundaries')
    laplacians = build_laplacians(boundaries)
    timeit('laplacians')
    features = build_features(simplices, cochains)
    timeit('features')
    save(boundaries, laplacians, features)
    timeit('total')
