# Simplicial Neural Networks

[Stefania Ebli], [Michaël Defferrard], [Gard Spreemann]

[Stefania Ebli]: https://people.epfl.ch/stefania.ebli
[Michaël Defferrard]: https://deff.ch/
[Gard Spreemann]: https://www.epfl.ch/labs/hessbellwald-lab/

This repository is a [PyTorch] implementation of Simplicial Neural Networks (SNNs).

> We present simplicial neural networks (SNNs), a generalization of graph neural networks to data that live on a class of topological spaces called [simplicial complexes].
> These are natural multi-dimensional extensions of graphs that encode not only pairwise relationships but also higher-order interactions between vertices—allowing us to consider richer data, including vector fields and nn-fold collaboration networks.
> We define an appropriate notion of convolution that we leverage to construct the desired convolutional neural networks.
> We test the SNNs on the task of imputing missing data on coauthorship complexes.

[PyTorch]: https://pytorch.org
[simplicial complexes]: https://en.wikipedia.org/wiki/Simplicial_complex

* Paper: [`arXiv:2010.03633`][paper]
* Poster: [Simplicial Neural Networks: Predicting Collaborations with Simplicial Complexes][poster]

[paper]: https://arxiv.org/abs/2010.03633
[poster]: https://www.dropbox.com/s/nwzbizjiunqk3g6/Ebli.pdf

## Installation

[![Binder](https://mybinder.org/badge_logo.svg)][binder_lab]
&nbsp; Click the binder badge to play with the notebooks from your browser without installing anything.

[binder_lab]: https://mybinder.org/v2/gh/xxx/snn/outputs?urlpath=lab


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

3. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```

## Notebooks

The below notebooks contain examples and experiments to play with the model.

- coming soon.


## Reproducing the results of the paper

Follow the below steps to reproduce the paper's results.

1. **Get the data.**

      1.1  See the [data README] for details
      
The data can be found in the folder ``` ./data/s2_raw ```.


2. **Preprocess the data.**

   2.1 Create a bipartite graph
   ```sh
      python ./data/s2_1_corpus_to_bipartite.py
   ```
   2.2 Downsample the bipartite graph
   ```sh
      python ./data/s2_2_downsample_bipartite.py
   ```
   2.3 Project the bipartite graph
    ```sh
       python ./data/s2_3_bipartite_to_graphs.py
   ```
   
The already preprocesses data can be found in the folder ``` ./data/s2_processed ```.


3. **Input to SNNS**
     
    3.1 Downsample the bipartite graph to have a connected simplicial complex  
   ```sh
      python ./input/s2_4_bipartite_to_downsampled.py
   ```
    3.2 From a bipartite graph to a simplicial complex with k-cochains
   ```sh
      python ./input/s2_5_bipartite_to_complex.py
   ```     
    3.3 From a simplicial complex to k-degree Laplacians 
   ```sh
      python ./input/s2_6_complex_to_laplacians.py
   ```       
    3.5 Artificially insert missing data on k-cochains
   ```sh
      python ./input/s2_7_cochains_to_missingdata.py
   ```      
4. **Run the experiments.**

     4.1 Train SNN to impute missing data on the simplicial comlex
   ```sh
      python ./experiments/learn_citations.py ./input .experiments/output 150250 30
   ``` 
## License & citation

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
Please cite the following paper if you use it.

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
