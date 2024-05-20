Tools for identifying influential neurons in connectome pathways. 
* Influence-ranking algorithms with fixed, homogeneous gains and learnable gains in `./influence_metrics.py`
    * homogeneous_gains: Ranks each neuron in graph J based on how ablation of the neuron decreases the total pathway weight sums from source neuron(s) to sink neuron(s). Fixed gain for each neuron determines how quickly pathways of increasing length attenuate. 
    * heterogeneous_gains: Learns a gain for each neuron in graph J such that activity at the sink neuron(s) given input at source neuron(s) is maximized. Gain bound parameter determines upper bound on the sum of all gains in the network.
* Simple example using both methods in `./example_usage.ipynb`


