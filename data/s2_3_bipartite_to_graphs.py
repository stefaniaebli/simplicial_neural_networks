#!/usr/bin/env python3

"""
Input:  bipartite graph X-Y
Output: projecttion of the bipartite graph on X and Y
"""

import time

import numpy as np
from scipy import sparse
import networkx as nx
from networkx.algorithms import bipartite as nxb


def load(path):

    biadjacency = sparse.load_npz(path)
    bipartite = nxb.from_biadjacency_matrix(biadjacency)

    print(f'{bipartite.number_of_edges():,} edges in the bipartite graph')
    print(f'connected: {nx.is_connected(bipartite)}')

    # nx.write_graphml(bipartite, 's2_2_bipartite_graph/paper_author.graphml')

    return bipartite


def project(bipartite):
    """Project the bipartite graph on both sides.

    Returns
    -------
    graph_papers : nx graph
        Graph where two papers are connected if they share an author.
    graph_authors : nx graph
        Graph where two authors are connected if they wrote a paper together.
    """

    nodes_papers = {n for n, d in bipartite.nodes(data=True) if d['bipartite']==0}
    nodes_authors = set(bipartite) - nodes_papers

    graph_papers = nxb.weighted_projected_graph(bipartite, nodes_papers)
    graph_authors = nxb.weighted_projected_graph(bipartite, nodes_authors)

    print(f'projection: {graph_papers.number_of_nodes():,} papers and {graph_papers.number_of_edges():,} edges')
    print(f'projection: {graph_authors.number_of_nodes():,} authors and {graph_authors.number_of_edges():,} edges')

    return graph_papers, graph_authors


def save(graph_papers, graph_authors):

    # nx.write_graphml(graph_papers, 's2_2_bipartite_graph/papers.graphml')
    # nx.write_graphml(graph_authors, 's2_2_bipartite_graph/authors.graphml')

    adjacency_papers = nx.to_scipy_sparse_matrix(graph_papers, dtype=np.int32)
    adjacency_authors = nx.to_scipy_sparse_matrix(graph_authors, dtype=np.int32)

    print('adjacency matrix: {:,} papers, {:,} edges'.format(adjacency_papers.shape[0], adjacency_papers.nnz // 2))
    print('adjacency matrix: {:,} authors, {:,} edges'.format(adjacency_authors.shape[0], adjacency_authors.nnz // 2))

    sparse.save_npz('s2_2_bipartite_graph/papers_adjacency.npz', adjacency_papers)
    sparse.save_npz('s2_2_bipartite_graph/authors_adjacency.npz', adjacency_authors)


if __name__ == '__main__':

    start = time.time()
    def timeit(name):
        print('wall time ({}): {:.0f}s'.format(name, time.time() - start))

    bipartite = load('s2_2_bipartite_graph/paper_author_biadjacency.npz')
    timeit('load')
    graph_papers, graph_authors = project(bipartite)
    timeit('project')
    save(graph_papers, graph_authors)
    timeit('total')
