# Simplicial Neural Networks

[Stefania Ebli], [Michaël Defferrard], [Gard Spreemann]

[Stefania Ebli]: https://people.epfl.ch/stefania.ebli
[Michaël Defferrard]: https://deff.ch/
[Gard Spreemann]: https://www.epfl.ch/labs/hessbellwald-lab/

> We present simplicial neural networks (SNNs), a generalization of graph neural networks to data that live on a class of topological spaces called [simplicial complexes].
> These are natural multi-dimensional extensions of graphs that encode not only pairwise relationships but also higher-order interactions between vertices—allowing us to consider richer data, including vector fields and nn-fold collaboration networks.
> We define an appropriate notion of convolution that we leverage to construct the desired convolutional neural networks.
> We test the SNNs on the task of imputing missing data on coauthorship complexes.

[PyTorch]: https://pytorch.org
[simplicial complexes]: https://en.wikipedia.org/wiki/Simplicial_complex

* Paper: [`arXiv:2010.03633`][paper]
* Poster: [Simplicial Neural Networks: Predicting Collaborations with Simplicial Complexes][poster]
* Data: [`doi:10.5281/zenodo.4144319`][data]

[paper]: https://arxiv.org/abs/2010.03633
[poster]: https://www.dropbox.com/s/nwzbizjiunqk3g6/Ebli.pdf
[data]: https://doi.org/10.5281/zenodo.4144319

## Installation

[![Binder](https://mybinder.org/badge_logo.svg)][binder_lab]
&nbsp; Click the binder badge to play with the notebooks from your browser without installing anything.

[binder_lab]: https://mybinder.org/v2/gh/stefaniaebli/simplicial_neural_networks/main

For a local installation, follow the below instructions.

1. Clone this repository.
    ```sh
    git clone https://github.com/stefaniaebli/simplicial_neural_networks.git
    cd simplicial_neural_networks
    ```

2. Create the environment.
    ```sh
    conda env create -f environment.yml
    conda activate snn
    ```

## Notebooks

* [`demo_simplicial_processing.ipynb`]: get a taste of simplicial complexes.
* [`s2_analysis.ipynb`]: analysis of the Semantic Scholar data.

[`demo_simplicial_processing.ipynb`]: https://nbviewer.jupyter.org/github/stefaniaebli/simplicial_neural_networks/blob/main/notebooks/demo_simplicial_processing.ipynb
[`s2_analysis.ipynb`]: https://nbviewer.jupyter.org/github/stefaniaebli/simplicial_neural_networks/blob/main/notebooks/s2_analysis.ipynb

## Reproducing our results

Run the below to train a SNN to impute missing data (citations) on the simplicial complex (which encodes collaborations between authors).

```sh
python ./experiments/impute_citations.py ./data/s2_3_collaboration_complex .experiments/output 150250 30
```

## Data

The data necessary to reproduce our experiment are found in the [`./data/s2_3_collaboration_complex`](./data/s2_3_collaboration_complex) folder.
The below three stages will recreate them.

[Semantic Scholar]: https://semanticscholar.org
[Open Research Corpus]: https://api.semanticscholar.org/corpus

1. Download the full archive of the [Open Research Corpus] from [Semantic Scholar], version 2018-05-03, which contains over 39 million published research papers in Computer Science, Neuroscience, and Biomedical.
    ```sh
    wget -i https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2018-05-03/manifest.txt -P data/s2_1_raw/
    ```
   This step populates the [`./data/s2_1_raw`](./data/s2_1_raw) folder.

2. Create a bipartite graph, whose vertices are papers (39,219,709 of them) in one part and authors (12,862,455 of them) in the other.
   A paper is connected to all its co-authors, and an author is connected to all the papers they wrote, leading to 139,268,795 edges.
   A citation count (the number of times the paper was cited) is available for each paper (from 0 to 37,230 citations per paper).
    ```sh
    # Create a bipartite graph from Semantic Scholar.
    python s2_1_corpus_to_bipartite.py
    # Clean and downsample that bipartite graph.
    python s2_2_downsample_bipartite.py
    # Project the bipartite graph to a graph between authors.
    python s2_3_bipartite_to_graphs.py
    ```
   Those steps populate the [`./data/s2_2_bipartite_graph`](./data/s2_2_bipartite_graph) folder.
   Alternatively, that processed data is available at [`doi:10.5281/zenodo.4144319`][data].

3. Build the collaboration complex (where each collaboration of authors is represented by a simplex) and citation cochains (which are the number of citations attributed to the collaborations).
    ```sh
    # Downsample the bipartite graph to have a connected simplicial complex.
    python s2_4_bipartite_to_downsampled.py
    # From a bipartite graph to a simplicial complex with k-cochains.
    python s2_5_bipartite_to_complex.py
    # From a simplicial complex to k-degree Laplacians.
    python s2_6_complex_to_laplacians.py
    # Artificially insert missing data on k-cochains.
    python s2_7_cochains_to_missingdata.py
    ```
   Those steps populate the [`./data/s2_3_collaboration_complex`](./data/s2_3_collaboration_complex) folder.

## License & citation

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
Please cite our paper if you use it.

```
@article{simplicial_nn,
  title = {Simplicial Neural Networks},
  author = {Ebli, Stefania and Defferrard, Micha\"el and Spreemann, Gard},
  year = {2020},
  archivePrefix = {arXiv},
  eprint = {2010.0363},
  url = {https://arxiv.org/abs/2010.0363},
}
```
