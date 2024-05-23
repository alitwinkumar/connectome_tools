# connectome_tools
This repository contains tools for analyzing synaptic connectivity data.

## Installation
### Option 1: Install without cloning
If you just want to be able to use the package and don't care about making any package modifications, you can install the package from git:
```bash
pip install git+https://github.com/alitwinkumar/connectome_tools.git
```

### Option 2: Clone and install
This can be useful if you intend on editing or contributing to this package.
```bash
# clone the repo
git clone git@github.com:alitwinkumar/connectome_tools.git
cd connectome_tools

# Don't forget to activate your environment
[activate your environment]

# Install connectome_tools package as editable 
pip install -e .
```
The `-e` flag will make the package editable, so any changes made to the package will be reflected in python.

## Package tools

`connectome_loaders.py` contains functions that return synaptic connectivity for the [Janelia hemibrain (Scheffer et al. 2020)](https://elifesciences.org/articles/57443), [larval connectome (Winding et al. 2023)](https://www.science.org/doi/full/10.1126/science.add9330), and [FlyWire (Dorkenwald et al. 2022)](https://www.nature.com/articles/s41592-021-01330-0) datasets. Function docstrings contain instructions for how to download the data.

`plot_conn_flywire.py` shows example code for visualizing and calculating connectivity statistics for the FlyWire connectome, including analyses grouped by neurotransmitter type.

`shuffle.py` contains code for degree-matched shuffling of connectivity matrices and analysis of singular value spectra as compared to shuffles.

`filter_connectivity.py` contains code for (efficiently in the case of sparse matrices) subselecting the connectivity matrix from filtered sets of pre and post synaptic neurons.

`influence_metrics` contains code for ranking neurons in a connectome graph based on influence in behaviorally-relevant pathways. 

## Examples
The `examples` folder showcases some examples.

`plot_conn_flywire` shows example code for visualizing and calculating connectivity statistics for the FlyWire connectome, including analyses grouped by neurotransmitter type.
