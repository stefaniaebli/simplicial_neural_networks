#!/usr/bin/env python3

"""
Input: collaboration bipartite graph X-Y and weights on X.
Output: X'= Downsample set of nodes of X (from bipartite graph X-Y) such that each node connects to at most 10 nodes in Y
        (eg the paper has at most 10 authors) and its weights are at least 5 (eg the number of citation is at least 5).
        To ensure that the resulting bipartite graph X'-Y' is connected we downsampled X (with the above restrictions) by performing random walks on the X-graph.
        (eg performing random walks on the papers graph -restricted to papers that have at least 5 citations and at most 10 authors-
        where two papers are connected if they have at least one author in common)
"""


import numpy as np
from scipy import sparse
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite as nxb
import scipy
from scipy import sparse
from scipy.sparse import coo_matrix
from random import shuffle
import time


def starting_node_random_walk(bipartite,weights_x, min_weight=100, max_dim=10 ):
    """
    Sample random node in X (from bipartite graph X-Y) with the restriction that it does not connect to more
    than "max_dim" nodes in Y and that its weight is more than "min_weight"

    Parameters
    ----------
    bipartite : scipy sparse matrix
        bipartite collaboration graph X-Y
    weights_x : ndarray
        Array of size bipartite.shape[0], containing the weights on the node of X

    min_weight : float
        minimum weight of the sampled node

    max_dim : int
        maximum number of adjacent nodes in Y

    Returns
    -------
        start : starting node of the random walk
    """
    Al=bipartite.tolil()
    rows=Al.rows
    seeds_papers=[]
    for j, el in enumerate(rows[np.where(weights_x>100)]):
        if len(el)<max_dim:
            #print('Paper {} has {} authors and {} citations'.format(np.where(weights_x>100)[0][j],len(el),weights_x[np.where(weights_x>100)][j]))
            seeds_papers.append(np.where(weights_x>100)[0][j])
    copy_seed=np.copy(seeds_papers)
    shuffle(copy_seed)
    start=copy_seed[0]
    return int(start)



def subsample_node_x(adjaceny_graph_x,bipartite,weights_x, min_weight=5, max_dim=10,length_walk=80):
    """"
        Downsample set of nodes X' of X (from bipartite graph X-Y) such that each node connects to at most 10 nodes in Y
        (eg the paper has at most 10 authors) and its weights are at least 5 (eg the number of citation is at least 5).
        To ensure that the resulting bipartite graph X'-Y' is connected we downsampled X (with the above restrictions) by performing random walks on the X-graph.
        (eg performing random walks on the papers graph -restricted to papers that have at least 5 citations and at most 10 authors-
        where two papers are connected if they have at least one author in common)

        Parameters
        ----------
        adjaceny_graph_x : scipy sparse matrix
            adjacency matrix of X (from the bipartite graph X-Y)

        bipartite : scipy sparse matrix
            bipartite collaboration graph X-Y

        weights_x : ndarray
            Array of size bipartite.shape[0], containing the weights on the node of X

        min_weight : float
            minimum weight of the sampled node, default 5

        max_dim : int
            maximum number of adjacent nodes in Y, default 1-

        length_walk : int
            length of random walk with the above restrictions
        Returns
        -------
        p: array of the downsampled nodes in X = X'
    """

    start= starting_node_random_walk(bipartite,weights_x, min_weight=min_weight, max_dim=max_dim )
    Al=bipartite.tolil()
    rows=Al.rows
    G = nx.from_scipy_sparse_matrix(adjaceny_graph_x)

    new_start=start


    H=nx.algorithms.traversal.breadth_first_search.bfs_edges(G, new_start, reverse=False, depth_limit=1)
    e=list(H)
    B=nx.Graph()
    B.add_edges_from(e)
    nodes=np.array(B.nodes())
    down_cit=weights_x[nodes]
    p=nodes[np.where(down_cit>=min_weight)]


    list_seeds=[new_start]
    for iterations in range(0,length_walk):
        seed_papers=[]
        for j, el in enumerate(rows[nodes]):
            if len(el)<max_dim and weights_x[nodes[j]]>=min_weight:
                seed_papers.append(nodes[j])

        c=list(set(seed_papers).difference(list_seeds))
        if len(c)<=1:
            break
        new_start=c[np.argsort(weights_x[c])[-2]]
        H1=nx.algorithms.traversal.breadth_first_search.bfs_edges(G, new_start, reverse=False, depth_limit=1)
        e1=list(H1)
        B=nx.Graph()
        B.add_edges_from(e1)
        nodes=np.array(B.nodes())
        down_cit=weights_x[nodes]
        p1=nodes[np.where(down_cit>=min_weight)]
        final=np.concatenate((p,p1))
        p=np.unique(final)
        list_seeds.append(new_start)
       
    return p


if __name__ == '__main__':

    start = time.time()
    def timeit(name):
        print('wall time ({}): {:.0f}s'.format(name, time.time() - start))

    adjacency_papers = sparse.load_npz('s2_2_bipartite_graph/papers_adjacency.npz')
    adjacency = scipy.sparse.load_npz('s2_2_bipartite_graph/paper_author_biadjacency.npz')
    papers = pd.read_csv('s2_2_bipartite_graph/papers.csv', index_col=0)
    citations=np.array(papers['citations_2019'])

    starting_node=starting_node_random_walk(adjacency,weights_x=citations, min_weight=100, max_dim=10 )
    print("The starting node of the random walk has ID {}".format(starting_node))
    downsample= subsample_node_x(adjacency_papers,adjacency,weights_x=citations, min_weight=5, max_dim=10,length_walk=80)

    timeit('process')
    np.save(f's2_3_collaboration_complex/{starting_node}_downsampled.npy', downsample)
    timeit('total')
