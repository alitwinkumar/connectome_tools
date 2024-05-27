
from setuptools import setup, find_packages

setup(
    name='connectome_tools',
    version='1.0.0',
    url='https://github.com/alitwinkumar/connectome_tools',
    author='Litwin-Kumar Lab',
    # author_email='author@gmail.com',
    description='This repository contains tools for analyzing synaptic connectivity data.',
    packages=find_packages(),    
    install_requires = [
        "numpy >= 1.11.1",
        "scipy >= 1.10.0",
        "pandas >= 1.5.0",
        "matplotlib >= 1.5.1",
    ],
)
