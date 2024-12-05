import scipy
import pandas as pd

def multihop_subselect_neurons(filtered_neurons_pre, filtered_neurons_post, J, n_steps = 1):
    """
    Subselect neurons agnostic to sparse / dense format of J, and with the option of multiple steps or hops in the graph.
    """
    if scipy.sparse.issparse(J):
        filtered_J = J.tocsc()[:,filtered_neurons_pre.J_idx_pre].tocsr()
    else:
        filtered_J = J[:,filtered_neurons_pre.J_idx_pre]
    
    while n_steps > 1:
        filtered_J = J@filtered_J
        n_steps -= 1

    if scipy.sparse.issparse(J):
        filtered_J = filtered_J.tocsr()[filtered_neurons_post.J_idx_post]
    else:
        filtered_J = filtered_J[filtered_neurons_post.J_idx_post]
    return filtered_J

def filter_connectivity(filtered_neurons_pre, filtered_neurons_post, J,nts_Js = None, n_steps = 1):
    """
    Filter the connectivity matrix, J, to only include presynaptic neurons in filtered_neurons_pre, and post synaptic neurons in filtered_neurons_post.
    filtered_neurons_pre and filtered_neurons_post are dataframes as loaded by connectome_loaders.py
    J is a sparse or dense connectivity matrix, as returned by connectome_loaders.py, or previous passes to filter_connectivity.
    nts_Js is a dictionary of sparse or dense connectivity matrices, as returned by connectome_loaders.py when by_nts is True, or by previous passes to filter_connectivity with nts_Js included
    n_steps is the number of steps or hops in the graph to compute for the filtered connectivity matrix
    returns: 
        neurons, a dataframe containing included neuron IDs and information with J_idx_pre and J_idx_post updated to match the filtered J matrix, with -1 indicating not included
        filtered_J, a synaptic connectivity matrix (rows postsynaptic) containing only connections between the selected pre and post synaptic neurons
        filtered_neurons_pre, the passed in filtered_neurons_pre with updated 'J_idx_pre' and removed 'J_idx_post'
        filtered_neurons_post, the passed in filtered_neurons_post with updated 'J_idx_post' and removed 'J_idx_pre'
        filtered_nts_Js (optional, if nts_Js is passed in), a dictionary of synapic connectivity matrices by neurotransmitter (rows postsynaptic) with each matrix containing only connections between the selected pre and post synaptic neurons
    """
    
    # copy pre and post neuron data frames to avoid pandas warnings if the passed in dataframes are a view
    filtered_neurons_pre = filtered_neurons_pre.copy()
    filtered_neurons_post = filtered_neurons_post.copy()

    # subselect the J matrix, efficiently if sparse

    filtered_J = multihop_subselect_neurons(filtered_neurons_pre, filtered_neurons_post, J, n_steps = n_steps)
    
    if nts_Js is not None:
        filtered_nts_Js = {}
        for key, val in nts_Js.items():
            filtered_nts_Js[key] = multihop_subselect_neurons(filtered_neurons_pre, filtered_neurons_post, val, n_steps = n_steps)
            
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
    if nts_Js is None:
        return neurons, filtered_J, filtered_neurons_pre, filtered_neurons_post
    return neurons, filtered_J, filtered_neurons_pre, filtered_neurons_post, filtered_nts_Js