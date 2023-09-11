import scipy
import pandas as pd
def filter_connectivity(filtered_neurons_pre, filtered_neurons_post, J):
    """
    Filter the connectivity matrix to only include presynaptic neurons in filtered_neurons_pre, and post synaptic neurons in filtered_neurons_post.
    datapath is a path (including trailing /) to this data.
    returns: 
        neurons, a dataframe containing included neuron IDs and information with J_idx_pre and J_idx_post updated to match the filtered J matrix, with -1 indicating not included
        J, a synaptic connectivity matrix (rows postsynaptic) containing only connections between the selected pre and post synaptic neurons
        filtered_neurons_pre, the passed in filtered_neurons_pre with updated 'J_idx_pre' and removed 'J_idx_post'
        filtered_neurons_post, the passed in filtered_neurons_post with updated 'J_idx_post' and removed 'J_idx_pre'
    """
    
    # copy pre and post neuron data frames to avoid pandas warnings if the passed in dataframes are a view
    filtered_neurons_pre = filtered_neurons_pre.copy()
    filtered_neurons_post = filtered_neurons_post.copy()

    # subselect the J matrix, efficiently if sparse
    if scipy.sparse.issparse(J):
        filtered_J = J.tocsr()[filtered_neurons_post.J_idx_post].tocsc()[:,filtered_neurons_pre.J_idx_pre].tocsr()
    else:
        filtered_J = J[filtered_neurons_post.J_idx_post][:,filtered_neurons_pre.J_idx_pre]

    # Update J_idx_pre and drop J_idx_post for filtered_neurons_pre, and vice versa for filtered_neurons_post
    filtered_neurons_pre["J_idx_pre"]=filtered_neurons_pre.reset_index().index
    if 'J_idx_post' in filtered_neurons_pre.columns:
        filtered_neurons_pre = filtered_neurons_pre.drop(columns = 'J_idx_post')
    filtered_neurons_post["J_idx_post"] = filtered_neurons_post.reset_index().index
    if 'J_idx_pre' in filtered_neurons_post.columns:
        filtered_neurons_post = filtered_neurons_post.drop(columns = 'J_idx_pre')
    
    # Create the neurons dataframe, which is J_idx_pre and J_idx_post merged.
    neurons = pd.merge(filtered_neurons_post, filtered_neurons_pre, how='outer')
    neurons['J_idx_post'] = neurons.J_idx_post.fillna(-1).astype('int')
    neurons['J_idx_pre'] = neurons.J_idx_pre.fillna(-1).astype('int')
    return neurons, filtered_J, filtered_neurons_pre, filtered_neurons_post