# connectome_tools
This repository contains tools for analyzing synaptic connectivity data.

## installation
to install, clone this repo by running `git@github.com:alitwinkumar/connectome_tools.git` and then within connectome_tools with the environment you want to install it in active, run `pip install -e .`

## files

The file connectome_loaders.py contains functions that return synaptic connectivity for the [Janelia hemibrain (Scheffer et al. 2020)](https://elifesciences.org/articles/57443), [larval connectome (Winding et al. 2023)](https://www.science.org/doi/full/10.1126/science.add9330), and [FlyWire (Dorkenwald et al. 2022)](https://www.nature.com/articles/s41592-021-01330-0) datasets. Function docstrings contain instructions for how to download the data.

The file plot_conn_flywire.py shows example code for visualizing and calculating connectivity statistics for the FlyWire connectome, including analyses grouped by neurotransmitter type.

shuffle.py contains code for degree-matched shuffling of connectivity matrices and analysis of singular value spectra as compared to shuffles.

filter_connectivity.py contains code for (efficiently in the case of sparse matrices) subselecting the connectivity matrix from filtered sets of pre and post synaptic neurons.

`influence_metrics` contains code for ranking neurons in a connectome graph based on influence in behaviorally-relevant pathways. 


