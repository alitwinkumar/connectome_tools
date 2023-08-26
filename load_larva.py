import numpy as np
import pandas as pd

#path to larval connectome data. download and extract supplemental information from Winding et al. (2023) "The connectome of an insect brain." at https://www.science.org/doi/10.1126/science.add9330#supplementary-materials.
#datapath = ""

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

N = len(neurons)

conns = pd.read_csv(datapath+"Supplementary-Data-S1/all-all_connectivity_matrix.csv",index_col=0,dtype=int)
assert((np.array(conns.columns.astype(int)) == np.array(conns.index)).all())
#only select connections to/from identified neurons
validinds = np.isin(conns.index,neurons.index) 
conns = conns.iloc[validinds,:].iloc[:,validinds]
#reorder neurons dataframe by connectivity matrix indices
neurons = neurons.loc[conns.index]
neurons = neurons.reset_index().rename(columns={"index": "id"})

J = conns.to_numpy().T #transpose to enforce rows postsynaptic, columns presynaptic convention
