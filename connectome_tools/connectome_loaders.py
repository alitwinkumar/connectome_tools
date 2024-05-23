from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_flywire(datapath, by_nts = False, include_spatial=False, J_matrix_dtype="int16"):
    """Load the flywire connectome dataset.
    You will need to manually download files from https://codex.flywire.ai/api/download and extract.
    The required files are "neurons.csv", "connections.csv", and "classification.csv".
    If you want to include spatial information, you will also need to download "coordinates.csv".

    Args:
        datapath (path-like): Local path for dataset.
        by_nts (bool, optional): Whether to also return a dictionary mapping neurotransmitter type to a J matrix for those transmitters. Defaults to False.
        include_spatial (bool, optional): Whether to include neuron position information. Defaults to False.
        J_matrix_dtype (str, optional): Data type for the synaptic connectivity matrix. Defaults to "int16". Sparse matrix default is int64 but max synapse count appears to be 2405 so we can save some space.

    Returns:
        tuple:
            - neurons (pd.DataFrame): a dataframe containing neuron IDs and information
            - J (sparse row matrix): a synaptic connectivity matrix (rows postsynaptic; i.e., J[post, pre] = syn count from pre to post)
            - nts_J (dict, only returned if by_nts is True): a dictionary of synaptic connectivity matrices by neurotransmitter type. Together, they sum to J.
    """
    if type(datapath) is str:
        datapath = Path(datapath)

    # Helper method to attempt to load a file and print error message if it doesn't exist
    def _attempt_load(filename):
        path = datapath / filename
        if not path.exists():
            raise FileNotFoundError(f"Could not load {filename} from {datapath}. Please download the data from https://codex.flywire.ai/api/download and extract.")
        return pd.read_csv(path)

    neurons = _attempt_load("neurons.csv")
    classif = _attempt_load("classification.csv")
    neurons = neurons.merge(classif, on="root_id", how="left")
    conns = _attempt_load("connections.csv")
    if include_spatial:
        coordinates = _attempt_load('coordinates.csv')

        # commenting out synpase coords since the root ids don't match.
        # syn_coordinates = pd.read_csv('synapse_coordinates.csv')


        # syn_coordinates = syn_coordinates.fillna(method='ffill')

        # presyn_coor = syn_coordinates.groupby('pre_root_id').mean().reset_index().drop(['post_root_id'], axis=1)
        # presyn_coorVar = syn_coordinates.groupby('pre_root_id').var().reset_index().drop(['post_root_id'], axis=1)
        # presyn_coor.insert(4, "rho", presyn_coorVar[['x','y','z']].T.sum().pow(0.5))
        # presyn_coor.astype({'pre_root_id': 'int64'})

        # postsyn_coor = syn_coordinates.groupby('post_root_id').mean().reset_index().drop(['pre_root_id'], axis=1)
        # postsyn_coorVar = syn_coordinates.groupby('post_root_id').var().reset_index().drop(['pre_root_id'], axis=1)
        # postsyn_coor.insert(4, "rho", postsyn_coorVar[['x','y','z']].T.sum().pow(0.5))
        # postsyn_coor.astype({'post_root_id': 'int64'})

        neur_coor = pd.DataFrame({"root_id":[], "x":[], "y":[], "z":[]})
        ll = coordinates['position']
        xx_ = [ int(l[1:-1].split()[0]) for l in ll]
        yy_ = [ int(l[1:-1].split()[1]) for l in ll]
        zz_ = [ int(l[1:-1].split()[2]) for l in ll]

        neur_coor["root_id"]=coordinates['root_id']
        neur_coor["x"]=xx_
        neur_coor["y"]=yy_
        neur_coor["z"]=zz_

        neur_coor = neur_coor.groupby('root_id').mean().reset_index()

        neurons = neurons.merge(neur_coor,on="root_id",how="left")
        # neurons = neurons.merge(presyn_coor,left_on="root_id", right_on="pre_root_id",how="left", suffixes=['','_presyn']).drop('pre_root_id', axis=1)
        # neurons = neurons.merge(postsyn_coor,left_on="root_id",right_on="post_root_id",how="left", suffixes=['','_postsyn']).drop('post_root_id', axis=1)



    N = len(neurons)
    idhash = dict(zip(neurons.root_id,np.arange(N)))
    neurons['J_idx'] = neurons['J_idx_post'] = neurons['J_idx_pre'] = neurons.root_id.apply(lambda x: idhash[x])
    preinds = [idhash[x] for x in conns.pre_root_id]
    postinds = [idhash[x] for x in conns.post_root_id]
    J = csr_matrix((conns.syn_count, (postinds, preinds)), shape=(N, N), dtype=J_matrix_dtype)

    if not by_nts:
        return neurons, J
    
    nts_Js = {}
    for name, group in conns.groupby('nt_type', dropna=False):
        if type(name) == float:
            name = 'nan'
        preinds = [idhash[x] for x in group.pre_root_id]
        postinds = [idhash[x] for x in group.post_root_id]
        nts_Js[name] =  csr_matrix((group.syn_count, (postinds, preinds)), shape=(N, N), dtype=J_matrix_dtype)

    return neurons, J, nts_Js


def load_hemibrain(datapath, sparse=False):
    """
    download and extract the file labeled "a compact (44 MB) data model" from https://dvid.io/blog/release-v1.2.
    datapath is a path to this data. sparse=True returns the connectivity as a sparse matrix, sparse=False (default) returns a dense matrix (~4GB for 64-bit).
    returns: neurons, a dataframe containing neuron IDs and information and J, a synaptic connectivity matrix (rows postsynaptic)
    """
    if type(datapath) is str:
        datapath = Path(datapath)
    neurons = pd.read_csv(datapath / "traced-neurons.csv")
    conns = pd.read_csv(datapath / "traced-total-connections.csv")

    N = len(neurons)
    idhash = dict(zip(neurons.bodyId,np.arange(N)))
    neurons['J_idx'] = neurons['J_idx_post'] = neurons['J_idx_pre'] = neurons.bodyId.apply(lambda x: idhash[x])
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
    datapath is a path to this data.
    returns: neurons, a dataframe containing neuron IDs and information and J, a synaptic connectivity matrix (rows postsynaptic)
    """
    if type(datapath) is str:
        datapath = Path(datapath)
    neurons_all = pd.read_csv(datapath / "science.add9330_data_s2.csv")

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

    conns = pd.read_csv(datapath / "Supplementary-Data-S1" / "all-all_connectivity_matrix.csv", index_col=0, dtype=int)
    assert((np.array(conns.columns.astype(int)) == np.array(conns.index)).all())
    #only select connections to/from identified neurons
    validinds = np.isin(conns.index,neurons.index) 
    conns = conns.iloc[validinds,:].iloc[:,validinds]
    #reorder neurons dataframe by connectivity matrix indices
    neurons = neurons.loc[conns.index]
    neurons = neurons.reset_index().rename(columns={"index": "id"})
    neurons['J_idx'] = neurons['J_idx_post'] = neurons['J_idx_pre'] = neurons.index

    J = conns.to_numpy().T #transpose to enforce rows postsynaptic, columns presynaptic convention

    return neurons,J
