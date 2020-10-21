# Simplicial Neural Networks

Stefania Ebli, Michaël Defferrard and Grad Spreemann

The code in this repository implements Simplicial Neural Networks (SNNs) a generalization of Convolutional Neural Networks (CNNs) to [simplicial complexes].

[simplicial complexes]: https://en.wikipedia.org/wiki/Simplicial_complex

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

## Reproducing the results of the paper

Follow the below steps to reproduce the paper's results.

1. **Get the data.**
   See the [data README](data/README.md) for details.

2. **Preprocess the data.**

3. **Run the experiments.**

## License & citation

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
Please cite the following paper if you use it.

```
@article{simplicial_nn,
  title = {Simplicial Neural Networks},
  author = {},
  year = {2020},
  archivePrefix = {arXiv},
  eprint = {19xx.xxxxx},
  url = {https://arxiv.org/abs/19xx.xxxxx},
}
```
