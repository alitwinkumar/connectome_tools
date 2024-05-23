
from setuptools import setup, find_packages

setup(
    name='ConnectomeTools',
    version='1.0.0',
    url='https://github.com/alitwinkumar/connectome_tools',
    author='Litwin-Kumar Lab',
    # author_email='author@gmail.com',
    description='This repository contains tools for analyzing synaptic connectivity data. The file connectome_loaders.py contains functions that return synaptic connectivity for the Janelia hemibrain (Scheffer et al. 2020), larval connectome (Winding et al. 2023), and FlyWire (Dorkenwald et al. 2022) datasets. Function docstrings contain instructions for how to download the data. The file plot_conn_flywire.py shows example code for visualizing and calculating connectivity statistics for the FlyWire connectome, including analyses grouped by neurotransmitter type. shuffle.py contains code for degree-matched shuffling of connectivity matrices and analysis of singular value spectra as compared to shuffles. filter_connectivity.py contains code for (efficiently in the case of sparse matrices) subselecting the connectivity matrix from filtered sets of pre and post synaptic neurons. influence_metrics contains code for ranking neurons in a connectome graph based on influence in behaviorally-relevant pathways.',
    packages=find_packages(),    
    install_requires = [
        "numpy >= 1.11.1",
        "scipy >= 1.10.0",
        "pandas >= 1.5.0",
        "matplotlib >= 1.5.1",
    ],
)