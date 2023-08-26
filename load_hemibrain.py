import numpy as np
import pandas as pd

#path to hemibrain data. download and extract the file labeled "a compact (44 MB) data model" from https://dvid.io/blog/release-v1.2.
#datapath = ""

neurons = pd.read_csv(datapath+"traced-neurons.csv")
conns = pd.read_csv(datapath+"traced-total-connections.csv")

N = len(neurons)
idhash = dict(zip(neurons.bodyId,np.arange(N)))
preinds = [idhash[x] for x in conns.bodyId_pre]
postinds = [idhash[x] for x in conns.bodyId_post]

#dense matrix format (~4GB for 64-bit)
J = np.zeros([N,N],dtype=int)
J[postinds,preinds] = conns.weight

#sparse matrix format
#from scipy.sparse import csr_matrix
#J = csr_matrix((conns.syn_count,(postinds,preinds)),shape=(N,N))
