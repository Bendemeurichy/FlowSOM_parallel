# FlowSOM

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saeyslab/FlowSOM_Python/test.yaml?branch=main
[link-tests]: https://github.com/saeyslab/FlowSOM_Python/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/flowsom

The complete FlowSOM package known from R, now available in Python!
Edited by @BenDeMeurichy to support parallel execution trough XPYSOM.
## Getting started

Please refer to the [documentation][link-docs]. In particular, the following resources are available:

-   [Example FlowSOM notebook][link-docs-example]
-   [API documentation][link-api]
-   [FlowSOM Python Cheatsheet][cheatsheet]

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install FlowSOM:

original package:
<!--
1) Install the latest release of `FlowSOM` from `PyPI <https://pypi.org/project/FlowSOM/>`_:

```bash
pip install FlowSOM
```
-->

1. Install the latest development version:
- original version:
```bash
pip install git+https://github.com/saeyslab/FlowSOM_Python
```
- parallel version: 
```bash
pip install git+https://github.com/Bendemeurichy/FlowSOM_parallel
```

## Usage

Starting from an FCS file that is properly transformed, compensated and checked for quality, the following code can be used to run the FlowSOM algorithm:

```python
# Import the FlowSOM package
import flowsom as fs

# Load the FCS file
ff = fs.io.read_FCS("./tests/data/ff.fcs")

# Run the FlowSOM algorithm
fsom = fs.FlowSOM(
    ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], xdim=10, ydim=10, n_clusters=10, seed=42 ,
    variant='xpysom', batch=True, batch_size=1000
)

# Plot the FlowSOM results
p = fs.pl.plot_stars(fsom, background_values=fsom.get_cluster_data().obs.metaclustering)
p.show()
```

The FlowSOM class supports multiple implementations of the training algorithm.
Currently, the following variants are supported:
- 'original': The original version implemented by the [saeyslab](https://github.com/saeyslab/FlowSOM_Python?tab=readme-ov-file).
- 'numba': Implementation using the numbasom at [numbasom](https://github.com/nmarincic/numbasom/tree/master/?tab=readme-ov-file). This version does not support all the extra parameters but is faster than the original.
- 'xpysom' (default): Implementation using the [xpysom](https://github.com/Manciukic/xpysom) library. Performs better, scores quite similarly to the original implementation and supports batches. It also supports all the original parameters.
- 'lr': The original implementation but it uses cosine anealing for learning rate decay instead of linear decay.
- 'batch_som': Batch implementation of the original SOM training function.

Batch training is supported for the 'xpysom' and 'batch_som' variants.
The batch implementation can be enabled by setting the `batch` parameter to `True`.
The `batch_size` parameter can be used to specify the size of the batches if needed, the default for this is `#cells // cpu_core_count`.

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests or if you found a bug, please use the [issue tracker][issue-tracker].

## Citation

### FlowSOM:
If you use `FlowSOM` in your work, please cite the following papers:

> A. Couckuyt, B. Rombaut, Y. Saeys, and S. Van Gassen, “Efficient cytometry analysis with FlowSOM in Python boosts interoperability with other single-cell tools,” Bioinformatics, vol. 40, no. 4, p. btae179, Apr. 2024, doi: [10.1093/bioinformatics/btae179](https://doi.org/10.1093/bioinformatics/btae179).

> S. Van Gassen et al., “FlowSOM: Using self-organizing maps for visualization and interpretation of cytometry data,” Cytometry Part A, vol. 87, no. 7, pp. 636–645, 2015, doi: [10.1002/cyto.a.22625](https://doi.org/10.1002/cyto.a.22625).


## Extra libraries used in the implementation of FlowSOM:
### XPYSOM:
> M. Manciu, “xpysom: XPySom is a minimalistic implementation of the Self Organizing Maps (SOM).” 2021, doi: [xpysom](https://github.com/Manciukic/xpysom).

### NUMBASOM:
> N. Marincic, “numbasom: A fast Self-Organizing Map Python library implemented in Numba.,” 2021, doi: [numbasom](https://github.com/nmarincic/numbasom/tree/master/?tab=readme-ov-file).

[issue-tracker]: https://github.com/saeyslab/FlowSOM_Python/issues
[changelog]: https://flowsom.readthedocs.io/en/latest/changelog.html
[link-docs]: https://flowsom.readthedocs.io
[link-docs-example]: https://flowsom.readthedocs.io/en/latest/notebooks/example.html
[link-api]: https://flowsom.readthedocs.io/en/latest/api.html
[cheatsheet]: https://flowsom.readthedocs.io/en/latest/_static/FlowSOM_CheatSheet_Python.pdf
