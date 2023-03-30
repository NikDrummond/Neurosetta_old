# Neurosetta

In development tools from my PhD. Ultimately the idea is to be able to handle both high resolution EM reconstructions of neuron data, and lower resolution microscopy data to facilitate the topological and spatial analysis of Neuron morphology, as well as modeling both biophysical and morphological properties of neurons.

Work has focused on T4 dendrites.

For optimisation and speed, this toolbox is built around three different toolboxes - [Graph-tool](https://graph-tool.skewed.de/), [Vaex](https://vaex.io/), and [numba](https://numba.pydata.org/).

`Graph-tool` is a C++ library, wrapped for python, offering an extencive feature set of fast graph theory tools. `Vaex` is a python library with pandas-like functionality, but uses lazy out-of-core DataFrames, which is much more efficient in terms of memory use and a range of other things. Finaly, `Numba` offers just-in-time (JIT) compilation for a subset of python and numpy code into machine code, offering significant speed ups.

Each of these libraries needs to be installed, which can be tricky. I would highly recomend the use of conda and a virtual environment for installation, as this is by far the simplest route to take.

## Install

Instructions to install 'conda' can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

A new virtual environment can then be created and activated:

```
conda create --name nr
conda activate nr
```

Graph-tool is available using conda-forge. to install it once the virtual environment is activated (this will install it within this virtual environment):

```
conda install -c conda-forge graph-tool
```

Vaex is also availabel through conda-forge, so is installed the same way:

```
conda install -c conda-forge vaex
```

Finally, numba is afailabel through conda normally so:

```
conda install numba
```

Each of these steps may take some time, so be patient!

For Neurosetta, we can install using pip from github:

```
pip install git+https://github.com/nikdrummond/Neurosetta/
```

