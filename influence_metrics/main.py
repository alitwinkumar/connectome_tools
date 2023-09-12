import argparse
import networkx as nx
import pandas as pd

from influence_metrics import heterogeneous_gains, homogeneous_gains

# User inputs
parser = argparse.ArgumentParser(description="Main script for running influence ranking algorithms on example graphs")
parser.add_argument('--exp_type', type=str, choices=['fixed_gains', 'learned_gains'],  help='Which experiment type to run: fixed, homogeneous gains or learned, heterogeneous gains')
parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu', help='Device type to be used (cpu or gpu)')
parser.add_argument('--graph', type=str, choices=['random_toy',  'trunc_geosmin', 'geosmin', 'walking', 'co2', 'courtship'], default='trunc_geosmin', help='which example graph to use')
parser.add_argument('--gain', type=float, default=0.5, help='gain to be used for each neuron for the fixed gain case')
parser.add_argument('--gain_bound', type=int, default=1000, help='sum of gains are bounded by this value')
parser.add_argument('--num_epochs', type=int, default=200, help='maximum number of epochs to run model for')
parser.add_argument('--early_convergence', type=bool, default=True, help='whether to stop training after a certain delta loss has been reached')
parser.add_argument('--random_init', type=bool, choices=[True, False], default=True, help='whether to randomly initialize gains')
parser.add_argument('--iter', type=int, default=0, help='optional label for which iteration is being run for batch runs')
parser.add_argument('--profile', type=bool, choices=[True, False], default=False, help='debugging mode to profile code')

user_args = parser.parse_args()

# ---------------------------------------------------------------------------------------------------
# Example circuits
# ---------------------------------------------------------------------------------------------------

# Truncated geosmin graph 
if user_args.graph == 'trunc_geosmin':
    # small graph example - 63 neurons
    df_G = pd.read_pickle('data/G_geosmin_trunc.pkl')
    source = [1670934213] # dl4_pn 
    sink = [981000564] # dnp42
   
    # connectivity mat removing source and sink
    G = nx.DiGraph()
    G= nx.from_pandas_edgelist(df_G, 'pre', 'post', ['weight'], create_using=nx.DiGraph())

    J = nx.to_pandas_adjacency(G).T

    label=user_args.graph

# ---------------------------------------------------------------------------------------------------
# Influence-ranking algorithms
# ---------------------------------------------------------------------------------------------------

# Influence-ranking algorithm with fixed, homogeneous gains
if user_args.exp_type == 'fixed_gains':
    homogeneous_gains(user_args.device, J, source, sink, user_args.gain, label)

# Influence-ranking algorithm with learnable gains
if user_args.exp_type == 'learned_gains':
    heterogeneous_gains(user_args.device, J, source, sink, user_args.gain_bound, user_args.num_epochs, label, user_args.early_convergence, user_args.random_init, user_args.iter, user_args.profile)

