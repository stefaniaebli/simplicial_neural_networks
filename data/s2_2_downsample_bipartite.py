#!/usr/bin/env python3

import pickle
import time

import numpy as np
from scipy import sparse
import pandas as pd


def count(papers, edges):
    print(f'  {len(papers):,} papers')
    print('  {:,} authors'.format(edges['author'].nunique()))
    print(f'  {len(edges):,} paper-author links')


def load(path):
    """Load the full bipartite graph."""

    with open(path, 'rb') as file:
        data = pickle.load(file)

    edges = pd.DataFrame(data['edges'], columns=['paper', 'author'])
    edges['author'] = pd.to_numeric(edges['author'])

    cols_citations = [f'citations_{year}' for year in range(1994, 2024, 5)]
    cols = cols_citations + ['references', 'year', 'missing_authors',
                             'missing_citations']
    papers = pd.DataFrame.from_dict(data['papers'], orient='index', columns=cols)
    papers.index.name = 'paper'

    print('input data:')
    count(papers, edges)
    print('  {:,} missed links'.format(papers['missing_authors'].sum()))
    print('  {:,} missed citations'.format(papers['missing_citations'].sum()))

    return papers, edges


def downsample(papers, edges):
    """Select a subset of the bipartite paper-author graph.

    Only drop papers. We want to keep all the authors from the selected papers.

    TODO: should we drop papers with missing in/out citations?
    There's probably citations S2 misses as well.
    """

    print('removing some papers:')

    def drop(pids, text):
        print(f'  drop {len(pids):,} {text}')
        papers.drop(pids, inplace=True)
        print(f'  {len(papers):,} papers remaining')

    def drop_edges(edges):
        keep = edges['paper'].isin(papers.index)
        print(f'  drop {len(edges) - keep.sum():,} edges from dropped papers')
        edges = edges[keep]
        print(f'  {len(edges):,} edges remaining')
        return edges

    # Papers that are not in the edge list, i.e., paper without identified authors.
    drop(papers.index.difference(edges['paper'].unique()), 'papers without authors')

    # Papers with at least one missing author ID. We want to identify all authors.
    drop(papers.index[papers['missing_authors'] != 0], 'papers with missing author IDs')

    # Papers with unknown publication year.
    drop(papers.index[papers['year'] == 0], 'papers without publication year')

    # Papers with no references.
    drop(papers.index[papers['references'] == 0], 'papers without references')

    # Papers with missing citations.
    drop(papers.index[papers['missing_citations'] != 0], 'papers with missing citations')

    # Papers that don't have enough citations. Too biased.
    # N_CITATIONS = 100
    # keep = (papers["citations_2019"] >= N_CITATIONS)
    # print(f'keep {keep.sum():,} papers with more than or equal to {N_CITATIONS} citations')
    # papers = papers[keep]

    edges = drop_edges(edges)

    # Papers written by too many authors (cap the simplex dimensionality).
    # Will also limit how fast the BFS grows.
    n_authors = 10
    size = edges.groupby('paper').count()
    drop(size[(size > n_authors).values].index, f'papers with more than {n_authors} authors')

    edges = drop_edges(edges)

    def grow_network(seed, n_papers):
        print(f'selecting at least {n_papers:,} papers around paper ID {seed}')
        new_papers = [seed]
        keep_papers = {seed}
        while len(keep_papers) < n_papers:
            print(f'  {len(keep_papers):,} papers currently selected')
            new_authors = edges['author'][edges['paper'].isin(new_papers)]
            new_papers = edges['paper'][edges['author'].isin(new_authors)]
            keep_papers = keep_papers.union(new_papers.values)
        print(f'  {len(keep_papers):,} papers selected')
        return keep_papers

    keep = grow_network(seed=papers.iloc[1].name, n_papers=30_000)

    papers = papers.loc[keep]
    edges = edges[edges['paper'].isin(papers.index)]

    papers.sort_index(inplace=True)
    edges.sort_values('paper', inplace=True)
    edges.reset_index(drop=True, inplace=True)

    print('remaining data:')
    count(papers, edges)

    return papers, edges


def add_node_ids(papers, edges):
    """Generate authors table and node IDs."""

    # Create author table with node IDs.
    authors = pd.DataFrame(edges['author'].unique(), columns=['author'])
    authors.sort_values('author', inplace=True)
    authors.set_index('author', inplace=True, verify_integrity=True)
    authors['author_node_id'] = np.arange(len(authors))
    print(f'author table: {len(authors):,} authors')

    # Create paper node IDs.
    papers['paper_node_id'] = np.arange(len(papers))

    # Insert paper and author node IDs in the edge list.
    edges = edges.join(papers['paper_node_id'], on='paper', how='right')
    edges = edges.join(authors['author_node_id'], on='author', how='right')
    edges.sort_index(inplace=True)

    return papers, authors, edges


def save(papers, authors, edges):
    """Save the paper-author biadjacency sparse matrix as `*_adjacency.npz` and
    the paper feature matrix as `*_citations.npy`."""

    biadjacency = sparse.coo_matrix((np.ones(len(edges), dtype=np.bool),
        (edges['paper_node_id'], edges['author_node_id'])))

    papers.drop('paper_node_id', axis=1, inplace=True)
    authors.drop('author_node_id', axis=1, inplace=True)
    edges.drop('paper_node_id', axis=1, inplace=True)
    edges.drop('author_node_id', axis=1, inplace=True)

    print('saving:')

    print('  paper table: {:,} papers, {:,} features'.format(*papers.shape))
    papers.to_csv('s2_2_bipartite_graph/papers.csv')
    print('  edges table: {:,} edges'.format(edges.shape[0]))
    edges.to_csv('s2_2_bipartite_graph/paper_author_edges.csv', index=False)

    print('  biadjacency matrix: {:,} papers, {:,} authors, {:,} edges'.format(
        *biadjacency.shape, biadjacency.nnz))
    sparse.save_npz('s2_2_bipartite_graph/paper_author_biadjacency.npz', biadjacency)

    # Not used because redundant. Moreover, papers.csv has column names.
    # print('  paper feature matrix: {:,} papers, {:,} features'.format(*papers.shape))
    # np.save('s2_2_bipartite_graph/papers_features.npy', papers.values)

    # Features would need to be aggregated for authors.
    # The features are actually propagated done when constructing the collaboration complex.
    # print('author feature matrix: {:,} authors, {:,} features'.format(*authors.shape))
    # np.save('s2_2_bipartite_graph/authors_features.npy', authors.values)


if __name__ == '__main__':

    start = time.time()
    def timeit(name):
        print('wall time ({}): {:.0f}s'.format(name, time.time() - start))

    papers, edges = load('s2_2_bipartite_graph/paper_author_full.pickle')
    timeit('load')
    papers, edges = downsample(papers, edges)
    timeit('downsample')
    papers, authors, edges = add_node_ids(papers, edges)
    save(papers, authors, edges)
    timeit('total')
