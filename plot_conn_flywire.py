#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:48:54 2023

@author: mbeiran

Plots the connectivity matrix 
"""

import pandas as pd
import numpy as np
from scipy import sparse 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


#%%
# Select output path for figures
figurepath =''

# Load data
datapath=''

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


#%%

def make_sparse_matrix(connections, match, assign_sign=True, neurotransmitter="all", neuropil="all", form = 'csc'):
    if neurotransmitter != "all":
        connections = connections[connections.nt_type == neurotransmitter]
    if neuropil != "all":
        connections = connections[connections.neuronpil == neuropil]
    pre_idxs = np.array([match[pre_idx] for pre_idx in connections.pre_root_id])
    post_idxs = np.array([match[post_idx] for post_idx in connections.post_root_id])
    values = np.array(connections.syn_count)
    if assign_sign:
        signs = np.asarray([nt_signs[nt_t] for nt_t in connections.nt_type])
        values = signs * values
    if form == 'csr':
        Mat = sparse.coo_matrix((values, (post_idxs, pre_idxs)), shape=(N, N)).tocsr()
    elif form=='csc':
        Mat = sparse.coo_matrix((values, (post_idxs, pre_idxs)), shape=(N, N)).tocsc()
    else:
        Mat = sparse.coo_matrix((values, (post_idxs, pre_idxs)), shape=(N, N))
    return Mat 

def prepare_plot():
    
    plt.style.use('ggplot')
    
    fig_width = 1.5*2.2 # width in inches
    fig_height = 1.5*2.2  # height in inches
    fig_size =  [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True
     
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markeredgewidth'] = 0.003
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['font.size'] = 14#9
    plt.rcParams['legend.fontsize'] = 11#7.
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0'
    plt.rcParams['axes.linewidth'] = '0.7'
    
    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = 14#9
    plt.rcParams['xtick.labelsize'] = 11#7
    plt.rcParams['ytick.labelsize'] = 11#7
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
    
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams["axes.grid"] = False
#%%

# Build matrix
match = {}
for n in range(len(neurons)):
    match[neurons.root_id.iloc[n]] = n

count = 0
match2 = np.zeros(len(neurons))
for n in range(len(neurons)):
    if neurons.nt_type.iloc[n]=='ACH':
        match[neurons.root_id.iloc[n]] = count 
        match2[n] = count
        count +=1
for n in range(len(neurons)):
    if neurons.nt_type.iloc[n]=='GLUT':
        match[neurons.root_id.iloc[n]] = count 
        match2[n] = count
        count +=1
for n in range(len(neurons)):
    if neurons.nt_type.iloc[n]=='GAB':
        match[neurons.root_id.iloc[n]] = count 
        match2[n] = count
        count +=1
for n in range(len(neurons)):
    if neurons.nt_type.iloc[n]!='GAB' and neurons.nt_type.iloc[n]!='GLUT' and neurons.nt_type.iloc[n]!='ACH':
        match[neurons.root_id.iloc[n]] = count 
        match2[n] = count
        count +=1    
        
        
        
nt_types = np.unique(conns.nt_type)
nt_signs = {"GABA": -1, "GLUT": -1, "ACH": +1, "DA": 0, "OCT": 0, "SER": 0}

connections_gr = conns.groupby(by=['pre_root_id', 'post_root_id', 'nt_type'], as_index=False).sum()



#%%
W = make_sparse_matrix(connections_gr, match, assign_sign=True, neurotransmitter="all")
Wr = make_sparse_matrix(connections_gr, match, assign_sign=True, neurotransmitter="all", form='csr')
Wco = make_sparse_matrix(connections_gr, match, assign_sign=True, neurotransmitter="all", form='coo')


Wall = make_sparse_matrix(conns, match, assign_sign=False, neurotransmitter="all")
W_ns = make_sparse_matrix(connections_gr, match, assign_sign=False, neurotransmitter="all")

W_gaba = make_sparse_matrix(connections_gr, match, assign_sign=False, neurotransmitter="GABA")
W_glut = make_sparse_matrix(connections_gr, match, assign_sign=False, neurotransmitter="GLUT")
W_ach  = make_sparse_matrix(connections_gr, match, assign_sign=False, neurotransmitter="ACH")


W_gaba_co = make_sparse_matrix(connections_gr, match, assign_sign=False, neurotransmitter="GABA", form='coo')
W_glut_co = make_sparse_matrix(connections_gr, match, assign_sign=False, neurotransmitter="GLUT", form='coo')
W_ach_co  = make_sparse_matrix(connections_gr, match, assign_sign=False, neurotransmitter="ACH", form='coo')

#%%
prepare_plot()
Cols  = ['C0', 'C1', 'C5']
#%%
fig = plt.figure(dpi=600, figsize=[7,7])
ax = fig.add_subplot(111, facecolor='black')
ax.plot(Wco.col, Wco.row, 's', color='white', ms=0.1)

plt.xlabel('pre neuron')
plt.ylabel('post neuron')
plt.savefig(figurepath+'adjacency.png')
plt.show()

#%%
fig = plt.figure(dpi=600, figsize=[7,7])
ax = fig.add_subplot(111, facecolor='black')
ax.plot(W_ach_co.col, W_ach_co.row, 's', color=Cols[0], ms=0.1)

plt.xlabel('pre neuron')
plt.ylabel('post neuron')
plt.savefig(figurepath+'adjacency_ach.png')
plt.show()

#%%
fig = plt.figure(dpi=600, figsize=[7,7])
ax = fig.add_subplot(111, facecolor='black')
ax.plot(W_glut_co.col, W_glut_co.row, 's', color=Cols[1], ms=0.1)

plt.xlabel('pre neuron')
plt.ylabel('post neuron')
plt.savefig(figurepath+'adjacency_glut.png')
plt.show()

#%%
fig = plt.figure(dpi=600, figsize=[7,7])
ax = fig.add_subplot(111, facecolor='black')
ax.plot(W_gaba_co.col, W_gaba_co.row, 's', color=Cols[2], ms=0.1)

plt.xlabel('pre neuron')
plt.ylabel('post neuron')
plt.savefig(figurepath+'adjacency_gaba.png')
plt.show()

#%%
fig = plt.figure(dpi=600, figsize=[7,7])
ax = fig.add_subplot(111, facecolor='black')
ax.plot(W_gaba_co.col, W_gaba_co.row, 's', color=Cols[2], ms=0.2)
ax.plot(W_glut_co.col, W_glut_co.row, 's', color=Cols[1], ms=0.2)
ax.plot(W_ach_co.col, W_ach_co.row, 's', color=Cols[0], ms=0.2)
ax.plot([5000, 5000], [0, 5000], c='w')
ax.plot( [0, 5000], [5000, 5000], c='w')
ax.plot( [0, 0], [0, 5000], c='w')
ax.plot( [0, 5000], [0, 0], c='w')
plt.xlabel('pre neuron')
plt.ylabel('post neuron')
plt.savefig(figurepath+'adjacency_sign.png')
plt.show()

#%%
fig = plt.figure(dpi=600, figsize=[7,7])
ax = fig.add_subplot(111, facecolor='black')
ax.plot(W_gaba_co.col, W_gaba_co.row, 's', color=Cols[2], ms=2)
ax.plot(W_glut_co.col, W_glut_co.row, 's', color=Cols[1], ms=2)
ax.plot(W_ach_co.col, W_ach_co.row, 's', color=Cols[0], ms=2)
plt.savefig(figurepath+'adjacency_sign_zoom.png')
plt.xlim([0,5000])
plt.ylim([0,5000])
plt.xlabel('pre neuron')
plt.ylabel('post neuron')
plt.savefig(figurepath+'adjacency_sign_zoom.png')
plt.show()

#%%


# Give statistics of how neurons are connected based on type

Ms = [W_ach, W_glut, W_gaba]
Ss = ['ACH', 'GLUT', 'GABA']
Cols  = ['C0', 'C1', 'C5']

fig = plt.figure(figsize=[10,5])

ci = 0
nbins=400
for ipre in range(3):
    for ipost in range(3):
        ci+=1
        ax = fig.add_subplot(3, 3, ci)

        M = Ms[ipre][np.ix_(match2[neurons.nt_type==Ss[ipre]], match2[neurons.nt_type==Ss[ipost]])]

        binM = M>0
        sp = binM.sum().sum()/(np.shape(M)[0]*np.shape(M)[1])
        ax.hist(np.ravel(M[M>0]), range=[0, nbins], bins=nbins, color=Cols[ipost])
        ax.set_yscale('log')
        if ipre==0:
            ax.set_title('From '+str(Ss[ipost]))
        if ipost==0:
            ax.set_ylabel('To '+str(Ss[ipre]))
        anchored_text = AnchoredText(f"p = {sp: .1E}", loc=1)
        ax.add_artist(anchored_text)
plt.savefig(figurepath+'curr_hist.png')
plt.show()

