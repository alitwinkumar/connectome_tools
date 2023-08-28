import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_flywire(datapath):
    """
    download "neurons," "connections," and "classification" from https://codex.flywire.ai/api/download and extract.
    datapath is a path (including trailing /) to this data.
    returns: neurons, a dataframe containing neuron IDs and information and J, a synaptic connectivity matrix (rows postsynaptic)
    """
    neurons = pd.read_csv(datapath+"neurons.csv")
    classif = pd.read_csv(datapath+"classification.csv")
    neurons = neurons.merge(classif,on="root_id",how="left")
    conns = pd.read_csv(datapath+"connections.csv")

    N = len(neurons)
    idhash = dict(zip(neurons.root_id,np.arange(N)))
    preinds = [idhash[x] for x in conns.pre_root_id]
    postinds = [idhash[x] for x in conns.post_root_id]

    J = csr_matrix((conns.syn_count,(postinds,preinds)),shape=(N,N))

    return neurons,J


def load_hemibrain(datapath, sparse=False):
    """
    download and extract the file labeled "a compact (44 MB) data model" from https://dvid.io/blog/release-v1.2.
    datapath is a path (including trailing /) to this data.
    returns: neurons, a dataframe containing neuron IDs and information and J, a synaptic connectivity matrix (rows postsynaptic)
    """
    neurons = pd.read_csv(datapath+"traced-neurons.csv")
    conns = pd.read_csv(datapath+"traced-total-connections.csv")

    N = len(neurons)
    idhash = dict(zip(neurons.bodyId,np.arange(N)))
    preinds = [idhash[x] for x in conns.bodyId_pre]
    postinds = [idhash[x] for x in conns.bodyId_post]


    if sparse:
        J = csr_matrix((conns.syn_count,(postinds,preinds)),shape=(N,N))
    else: #(~4GB for 64-bit)
        J = np.zeros([N,N],dtype=int)
        J[postinds,preinds] = conns.weight

    return neurons,J


def load_larva(datapath):
    """
    download and extract supplemental information from Winding et al. (2023) "The connectome of an insect brain." at https://www.science.org/doi/10.1126/science.add9330#supplementary-materials.
    datapath is a path (including trailing /) to this data.
    returns: neurons, a dataframe containing neuron IDs and information and J, a synaptic connectivity matrix (rows postsynaptic)
    """
    neurons_all = pd.read_csv(datapath+"science.add9330_data_s2.csv")

    #convert to integer, unpaired neurons assigned index of -1
    neurons_all.left_id[neurons_all.left_id == "no pair"] = "-1"
    neurons_all.right_id[neurons_all.right_id == "no pair"] = "-1"
    neurons_all.left_id = pd.to_numeric(neurons_all.left_id)
    neurons_all.right_id = pd.to_numeric(neurons_all.right_id)

    #generate combined dataframe of left/right neurons
    neurons_left = neurons_all.set_index("left_id")
    neurons_left["side"] = "left"
    neurons_left.rename(columns={"right_id": "lr_match_id"},inplace=True)
    neurons_right = neurons_all.set_index("right_id")
    neurons_right["side"] = "right"
    neurons_right.rename(columns={"left_id": "lr_match_id"},inplace=True)
    neurons = pd.concat([neurons_left,neurons_right])

    conns = pd.read_csv(datapath+"Supplementary-Data-S1/all-all_connectivity_matrix.csv",index_col=0,dtype=int)
    assert((np.array(conns.columns.astype(int)) == np.array(conns.index)).all())
    #only select connections to/from identified neurons
    validinds = np.isin(conns.index,neurons.index) 
    conns = conns.iloc[validinds,:].iloc[:,validinds]
    #reorder neurons dataframe by connectivity matrix indices
    neurons = neurons.loc[conns.index]
    neurons = neurons.reset_index().rename(columns={"index": "id"})

    N = len(neurons)

    J = conns.to_numpy().T #transpose to enforce rows postsynaptic, columns presynaptic convention

    return neurons,J
