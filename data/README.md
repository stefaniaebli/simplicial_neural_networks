# Semantic Scholar Graphs

The [Open Research Corpus] from [Semantic Scholar] contains over 39 million published research papers in Computer Science, Neuroscience, and Biomedical.
This repository contains code that produces several graphs (a.k.a. networks) from this dataset.

[Semantic Scholar]: https://semanticscholar.org
[Open Research Corpus]: https://labs.semanticscholar.org/corpus

All the graphs (in xx format) can be downloaded from Zenodo.
More graphs might be available in the futures. PRs welcome!

## Citation graph

xx papers (nodes), xx citations (edges)

Features (per paper):
* S2 id
* journal
* venue
* field? arxiv category if there is an arxiv link
* abstract => word count?
* entities / topics
* year

### Neighborhood complex

All the papers referenced in one paper form a facet.
Subset closure: any subset of papers are also referenced together.
Extension of citation graph.

Data on the individual papers or the facets.

## Paper-author bipartite graph

Papers and authors are node in a bipartite graph.
A paper is connected to all the authors who wrote it.
Similarly, an author is connected to all the papers he wrote.
A citation count (the number of times the paper was cited) is available for each paper.

Original data:
* 39,219,709 papers
* 12,862,455 authors (identified with ID)
* 139,268,795 edges
* citations: minimum 0 per paper, maximum 37,230 per paper, total xx

Cleaned:
* xx papers
* xx authors
* xx edges

### Co-authorship (a.k.a. collaboration) graph

Projected from the paper-author bipartite graph.

xx authors, xx edges

### Co-authorship (a.k.a. collaboration) simplicial complex

Projected from the paper-author bipartite graph.

* xx 0-simplices (single author paper)
* xx 1-simplices (papers with two authors)
* xx 2-simplices (papers with three authors)
* xx 3-simplices
* xx 4-simplices

### Papers graph

Projected from the paper-author bipartite graph.

### Papers complex

Projected from the paper-author bipartite graph.

## Word co-occurrence graph

Papers are connected if the same words appear in the abstract.

Or topics

Then we can use citations as a signal.

## Dataset creation

* Download the full archive of the [Open Research Corpus], version 2018-05-03.
  The following should work: `wget -i https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/manifest.txt -P data/raw/`.
* Run `./preprocess.py`.
* Run the jupyter notebook associated to the dataset of interest.

## Citation

Please cite the following paper if you use the Semantic Scholar data.

@inproceedings{ammar:18,
  title = {Construction of the Literature Graph in Semantic Scholar},
  author = {Waleed Ammar and Dirk Groeneveld and Chandra Bhagavatula and Iz Beltagy
    and Miles Crawford and Doug Downey and Jason Dunkelberger and Ahmed Elgohary
    and Sergey Feldman and Vu Ha and Rodney Kinney and Sebastian Kohlmeier
    and Kyle Lo and Tyler Murray and Hsu-Han Ooi and Matthew Peters and Joanna Power
    and Sam Skjonsberg and Lucy Lu Wang and Chris Wilhelm and Zheng Yuan
    and Madeleine van Zuylen and Oren Etzioni},
  booktitle = {NAACL},
  year = {2018},
  url = {https://www.semanticscholar.org/paper/09e3cf5704bcb16e6657f6ceed70e93373a54618}
}
```
