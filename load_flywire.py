import numpy as np
import pandas as pd

#path to flywire data. download "neurons," "connections," and "classification" from https://codex.flywire.ai/api/download and extract.
#datapath = ""

neurons = pd.read_csv(datapath+"neurons.csv")
classif = pd.read_csv(datapath+"classification.csv")
neurons = neurons.merge(classif,on="root_id",how="left")
conns = pd.read_csv(datapath+"connections.csv")

N = len(neurons)
idhash = dict(zip(neurons.root_id,np.arange(N)))
preinds = [idhash[x] for x in conns.pre_root_id]
postinds = [idhash[x] for x in conns.post_root_id]

#sparse matrix format
from scipy.sparse import csr_matrix
J = csr_matrix((conns.syn_count,(postinds,preinds)),shape=(N,N))
