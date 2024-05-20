import numpy as np
import matplotlib.pyplot as plt

def shuf(J):
    """
    shuffles adjacency matrix J preserving row-wise in-degrees and column-wise connection probabilities
    J is an adjacency matrix with 0/1 entries. 
    returns: Jshuf, a binary shuffled weight matrix
    """
    assert np.in1d(J,[0,1]).all() #J must be a binary adjacency matrix

    M,N = J.shape
    cprobs = np.sum(J,0)/np.sum(J)

    indeg = np.sum(J,1) #in-degree sequence

    Jshuf = np.zeros([M,N])
    for ii in range(M):
        inds = np.random.choice(N,indeg[ii],p=cprobs,replace=False)
        Jshuf[ii,inds] = 1

    return Jshuf

def compare_spectrum_shuf(J,shuf_func,Nshuf=200):
    """
    compares the squared singular value spectrum of J to that of shuffled matrices
    J is an adjacency matrix with 0/1 entries. shuf_func is a function that returns a random shuffle of J. Nshuf is the number of shuffles.
    """
    assert np.in1d(J,[0,1]).all() #J must be a binary adjacency matrix

    _,s,_ = np.linalg.svd(J) #singular values

    R = len(s)
    sshuf = np.zeros([R,Nshuf]) #shuffled singular values
    for si in range(Nshuf):
        Jshuf = shuf_func(J)
        _,sshuf[:,si],_ = np.linalg.svd(Jshuf)

    m = np.mean(sshuf**2,1)
    #95% confidence intervals
    qmin = np.quantile(sshuf**2,0.05,axis=1)
    qmax = np.quantile(sshuf**2,0.95,axis=1) 

    plt.scatter(1+np.arange(R),s**2,color="k",s=5,zorder=2)
    plt.scatter(1+np.arange(R),m,color="gray",s=5,zorder=1)
    plt.errorbar(1+np.arange(R),m,yerr=np.vstack([m-qmin,qmax-m]),ecolor="gray",capsize=1,linestyle="None",zorder=1)

    plt.xlabel("Component")
    plt.ylabel("Squared s.v.")
    plt.legend(["Data","Shuffle"])

    plt.tight_layout()
