import numpy as np
import scipy.sparse
import pandas as pd
import pytest
from connectome_tools.filter_connectivity import multihop_subselect_neurons

def old_subselect_neurons(filtered_neurons_pre, filtered_neurons_post, J):  
    """
    Subselect neurons based on pre and post indices.
    """
    if scipy.sparse.issparse(J):
        filtered_J = J.tocsr()[filtered_neurons_post.J_idx_post].tocsc()[:,filtered_neurons_pre.J_idx_pre].tocsr()
    else:
        filtered_J = J[filtered_neurons_post.J_idx_post][:,filtered_neurons_pre.J_idx_pre]
    return filtered_J


def test_multihop_equivalent_to_old_subselect():
    # Create sample data
    n_neurons = 10
    J_dense = np.random.rand(n_neurons, n_neurons)
    J_sparse = scipy.sparse.csr_matrix(J_dense)
    
    # Create sample filtered neurons dataframes
    neurons_pre = pd.DataFrame({
        'J_idx_pre': [1, 3, 5],  # Select a few pre-synaptic neurons
        'neuron_id': ['a', 'b', 'c']
    })
    
    neurons_post = pd.DataFrame({
        'J_idx_post': [2, 4, 6],  # Select a few post-synaptic neurons
        'neuron_id': ['d', 'e', 'f']
    })



    # Test dense matrix
    result_dense_old = old_subselect_neurons(neurons_pre, neurons_post, J_dense)
    result_dense_new = multihop_subselect_neurons(neurons_pre, neurons_post, J_dense, n_steps=1)
    np.testing.assert_array_almost_equal(result_dense_old, result_dense_new)

    # Test sparse matrix
    result_sparse_old = old_subselect_neurons(neurons_pre, neurons_post, J_sparse)
    result_sparse_new = multihop_subselect_neurons(neurons_pre, neurons_post, J_sparse, n_steps=1)
    assert scipy.sparse.issparse(result_sparse_new)  # Verify output is still sparse
    np.testing.assert_array_almost_equal(result_sparse_old.toarray(), result_sparse_new.toarray()) 

def test_multihop_filter_connectivity_nsteps():
    # Create sample data
    n_neurons = 10
    J_dense = np.random.rand(n_neurons, n_neurons)
    J_sparse = scipy.sparse.csr_matrix(J_dense)
    
    # Create sample filtered neurons dataframes
    neurons_pre = pd.DataFrame({
        'J_idx_pre': [1, 3, 5],  # Select a few pre-synaptic neurons
        'neuron_id': ['a', 'b', 'c']
    })
    
    neurons_post = pd.DataFrame({
        'J_idx_post': [2, 4, 6],  # Select a few post-synaptic neurons
        'neuron_id': ['d', 'e', 'f']
    })


    for n_steps in range(2,4):
       
        # Test dense matrix
        result_dense_old = old_subselect_neurons(neurons_pre, neurons_post, np.linalg.matrix_power(J_dense,n_steps))
        result_dense_new = multihop_subselect_neurons(neurons_pre, neurons_post, J_dense, n_steps=n_steps)
        np.testing.assert_array_almost_equal(result_dense_old, result_dense_new)

        # Test sparse matrix
        result_sparse_old = old_subselect_neurons(neurons_pre, neurons_post, scipy.sparse.linalg.matrix_power(J_sparse,n_steps))
        result_sparse_new = multihop_subselect_neurons(neurons_pre, neurons_post, J_sparse, n_steps=n_steps)
        assert scipy.sparse.issparse(result_sparse_new)  # Verify output is still sparse
        np.testing.assert_array_almost_equal(result_sparse_old.toarray(), result_sparse_new.toarray()) 

test_multihop_equivalent_to_old_subselect()
test_multihop_filter_connectivity_nsteps()