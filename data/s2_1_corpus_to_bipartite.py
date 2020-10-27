#!/usr/bin/env python3

import gzip
import glob
import json
import pickle
import time

import numpy as np
from tqdm import tqdm


def process_papers(corpus):
    """First pass through all the papers."""

    papers = dict()
    edges = list()

    for file in tqdm(sorted(glob.glob(corpus))):

        with gzip.open(file, 'rt') as file:

            for line in file:

                paper = json.loads(line)

                try:
                    year = paper['year']
                except KeyError:
                    year = 0

                missing_authors = 0
                for author in paper['authors']:
                    n_ids = len(author['ids'])
                    if n_ids == 0:
                        missing_authors += 1
                    elif n_ids == 1:
                        edges.append((paper['id'], author['ids'][0]))
                    else:
                        raise ValueError("No author should have multiple IDs.")

                papers[paper['id']] = (paper['inCitations'],
                        len(paper['outCitations']), year, missing_authors)

    print(f'processed {len(papers):,} papers')
    print(f'collected {len(edges):,} paper-author links')

    return papers, edges


def count_citations(papers, years):
    """Second pass to check the publication year of the referencing papers."""

    years = np.array(years)

    for pid, attributes in tqdm(papers.items()):

        missing_citations = 0
        counts = np.zeros_like(years)

        for citation in attributes[0]:

            try:
                year = papers[citation][2]
            except KeyError:
                missing_citations += 1  # unknown paper
                continue

            if year != 0:
                counts += year < years
            else:
                missing_citations += 1  # unknown year

        papers[pid] = tuple(counts) + attributes[1:] + (missing_citations,)


def save(papers, edges):
    data = dict(edges=edges, papers=papers)
    with open('s2_2_bipartite_graph/paper_author_full.pickle', 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    start = time.time()
    def timeit(name):
        print('wall time ({}): {:.0f}s'.format(name, time.time() - start))

    papers, edges = process_papers('s2_1_raw/s2-corpus-*.gz')
    timeit('reading data')
    count_citations(papers, range(1994, 2024, 5))
    timeit('citations')
    save(papers, edges)
    timeit('total')
