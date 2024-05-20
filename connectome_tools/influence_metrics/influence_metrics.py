import numpy as np
import tensorflow as tf  
import tensorflow.keras.optimizers as tfko
from tfdiffeq import odeint
import jax.numpy as jnp
from jax import default_backend
from jax import jit, vmap
import pandas as pd
import os
import time 

import warnings
warnings.filterwarnings('ignore')

def heterogeneous_gains(device, J, source, sink, gain_bound, num_epochs, label='', early_convergence=True, random_init=True, iter=0, profile=False):
    """ Learns a gain for each neuron in graph J such that activity at the sink neuron(s) given input at source neuron(s) is maximized. 

    Args:
        device (string): Device type to be used ('cpu' or 'gpu').
        J (pandas DataFrame): Graph in adjacency matrix format. Rows are postsynaptic, columns are presynaptic.
        source (array-like): Source neuron(s)
        sink (array-like): Target neuron(s)
        gain_bound (int): Sum of gains is bounded by this value
        num_epochs (int): Maximum number of epochs to run the model for
        label (str, optional): Descriptive experiment label for file output. Defaults to ''.
        early_convergence (bool, optional): Whether to stop training after a certain delta loss has been reached. Defaults to True.
        random_init (bool, optional): Whether to randomly initialize gains. Defaults to True.
        iter (int, optional): Optional label for which iteration is being run for batch runs. Defaults to 0.
        profile (bool, optional): Debugging mode to profile code. Defaults to False.

    Returns:
        final_learned_gains (array): List of final per-neuron gains 
    """

    exp_key = 'learned_gains_' + label + '_gainbound_' + str(gain_bound) + '_iter_' + str(iter)
    print(exp_key)

    if profile == True:
        print('profiling')

    # Device setup
    if device == 'gpu':
        device = '/GPU:0' if tf.test.is_gpu_available() else 'cpu'
    print('running on ' + device)

    # Remove inputs to source, outputs from sink
    J.loc[:, sink] = 0
    J.loc[source] = 0

    i_s = np.where(np.in1d(J.columns, source))[0][0]
    i_t = np.where(np.in1d(J.columns, sink))[0][0]

    nodes = np.array(J.index)
    num_neurons = J.shape[0]
   
    # Input at source
    s = np.zeros(num_neurons)
    s[i_s] = 1
 
    def ode(t, u, args):
        """ Explicitly defines ODE system 
            dx/dt = -x + args @ J @ x + s

        Args:
            t (int): current timestep
            u (array): initial conditions 
            args (array): ODE params to be fit 

        Returns:
            array: dx/dt for this timestep 
        """    

        x = tf.identity(u)
        dx_dt = -x + tf.squeeze(tf.multiply(tf.reshape(args, (num_neurons, 1)), tf.matmul(J_tensor, tf.reshape(x, (num_neurons, 1))))) + s
        return dx_dt
    
    # Input setup
    J_tensor = tf.constant(J)
    s = tf.constant(s)
    t_begin= 0. # start time
    t_end= 5 # end time 
    t_stepsize = 0.01 # step size
    t_nsamples = int((t_end - t_begin) / t_stepsize) # number of timesteps
    t_space = np.linspace(t_begin, t_end, t_nsamples) # array of timesteps 
    t_space_tensor = tf.constant(t_space)

    u_init = tf.constant(np.zeros((num_neurons,)), dtype=t_space_tensor.dtype) 

    # Per-gain constraint
    constraint = lambda x: tf.clip_by_value(x, 0.01, 0.99)

    # Initialize trainable gains
    if random_init == True:
        args = [tf.Variable(initial_value=np.random.rand(), name='g' + str(i+1), trainable=True,
            dtype=t_space_tensor.dtype, constraint=constraint) for i in range(num_neurons)]

    elif random_init == False:
        args = [tf.Variable(initial_value=0.5, name='g' + str(i+1), trainable=True,
            dtype=t_space_tensor.dtype, constraint=constraint) for i in range(num_neurons)] 

    def net():
        """ single-layer NN with one ODE layer
            Given mapping from timestep+current state to new state, inital values x(0) and y(0), and array of timesteps, solve IVP

        Returns:
            array: solutions y(t) for all given timepoints (first value is initial condition y(0))
        """    
        with tf.device(device):
            sol = odeint(lambda ts, u0: ode(ts, u0, args),
                    u_init, t_space_tensor)

        return sol
    
    # Training parameters
    learning_rate = 0.05 
    epochs = num_epochs 
    optimizer = tfko.RMSprop(learning_rate=learning_rate) 

    # sum of squares of differences between solutions and ground truth solutions 
    def loss_func(num_sol):
        """ Loss function based on activity at sink 

        Args:
            num_sol (array): ode result

        Returns:
            array: activity at target neuron(s) 
        """

        return -num_sol[-1, i_t]

    print('starting training...')
 
    # Training loop
    def training():
        losses = []
        gains_over_epochs = []
        grads_over_epochs = []
        sink_activities = []
        gain_sums = []
        with tf.device(device):
            # tf.debugging.set_log_device_placement(True)
            for epoch in range(epochs):
                if epoch == 1:
                    start_time = time.perf_counter()

                with tf.GradientTape(watch_accessed_variables=True) as tape: 
                    tape.watch(args)
                    num_sol = net() 
                    loss_value = loss_func(num_sol)
                    losses.append(loss_value)
                print("Epoch:", epoch)

                grads = tape.gradient(loss_value, args)
                optimizer.apply_gradients(zip(grads, args))

                # constraint on sum of gains
                max_arg = gain_bound 
                arg_sum = tf.stop_gradient(tf.reduce_sum(args)).numpy()
                gain_sums.append(arg_sum)
                if arg_sum > max_arg: 
                    for i in range(len(args)):
                        sum_clipped = args[i].numpy()*max_arg / arg_sum 
                        args[i].assign(sum_clipped)
                
                # convergence criteria
                if early_convergence == True:
                    if epoch > 1:
                        delta_loss = np.abs(losses[-1] - losses[-2])
                        if delta_loss <= 1e-10:
                            break

                # output diagnositcs
                gains_over_epochs.append([args[i].numpy() for i in range(len(args))])
                grads_over_epochs.append([grads[i].numpy() for i in range(len(grads))])
                sink_activities.append(num_sol[:, i_t])

                if epoch == 1:
                    end_time = time.perf_counter()
                
                    if profile == True:
                        elapsed_time = end_time - start_time
                        print('elapsed time: ' + str(elapsed_time) + ' s')

            return losses, gains_over_epochs, grads_over_epochs, sink_activities, gain_sums


    if profile == True:
        lp = LineProfiler()
        lp_wrapper = lp(training)
        lp_wrapper()
        lp.print_stats()
    else:
        losses, gains_over_epochs, grads_over_epochs, sink_activities, gain_sums = training()


    # Save diagnostic outputs
    file_prefix = exp_key+'/'
    if not os.path.isdir(file_prefix):
        os.makedirs(file_prefix)
    np.save(file_prefix+exp_key+'_nodes.npy', nodes) # list of neurons in graph
    np.save(file_prefix+exp_key+'_gains.npy', gains_over_epochs) # learned gains at each training epoch
    np.save(file_prefix+exp_key+'_source.npy', source) # list of source neuron(s)
    np.save(file_prefix+exp_key+'_sink.npy', sink) # list of target neuron(s)
    np.save(file_prefix+exp_key+'_losses.npy', losses) # loss at each training epoch
    np.save(file_prefix+exp_key+'_sink_activities.npy', sink_activities) # activity at sink at each training epoch
    np.save(file_prefix+exp_key+'_gradients_over_epoch.npy', grads_over_epochs) # gradients at each epoch
    np.save(file_prefix+exp_key+'_gain_sums.npy', gain_sums) # sum of gains at each epoch

    # Learned gains at the final training epoch
    final_learned_gains = gains_over_epochs[-1]
    results = pd.DataFrame({'nodes': nodes, 'learned_gains': final_learned_gains})

    return results

def homogeneous_gains(device, J, source, sink, gain, label=''):
    """ Ranks each node in graph J based on how much ablating it from the graph decreases total pathway sums from source neuron(s) to sink neuron(s).

    Args:
        device (string): Device type to be used ('cpu' or 'gpu').
        J (pandas DataFrame): Graph in adjacency matrix format. Rows are postsynaptic, columns are presynaptic.
        source (array-like): Source neuron(s)
        sink (array-like): Target neuron(s)
        gain (float): A single scalar gain value to be used for each neuron in J
        label (str, optional): Descriptive experiment label for file output. Defaults to ''.

    Returns:
        _type_: _description_
    """
    if device == 'gpu':
        assert default_backend() == 'gpu'

    exp_key = 'fixed_gains_' + label + '_gain_' + str(gain) 
    print('starting experiment ' + exp_key)
    
    J = J.T # switch to (presynaptic, postsynaptic) 
    s = source
    t = sink
    J.loc[:, s] = 0
    J.loc[t] = 0
    J_ = jnp.array(J)
    num_neurons = J.shape[0]
    i_t = np.where(np.in1d(J.columns, t))[0]
    i_s = np.where(np.in1d(J.index, s))[0]

    # Compute total influence from source(s) onto sink(s)
    u = np.float32(np.array(J.loc[s])) # source matrix u construction
    a = np.float32(np.eye(num_neurons) - gain * J_) # (I-aJ)
    A_inv = np.linalg.inv(a) # (I-aJ)^-1
    A_inv = jnp.array(A_inv)
    total = u @ A_inv[:, i_t] # v = u (I - aJ)^-1
    total = total.sum()
    print('total influence: ' + str(total))

    nodes = []
    outputs = []

    C_inv = jnp.array([1]).reshape((1, 1))
    start = time.time()

    # Remove a node from J and recompute influence
    def jax_ops(i, num_neurons, C_inv, A_inv, i_t, J_, gain): 
        U = jnp.zeros((num_neurons, 1)) 
        U = U.at[i,:].set(gain)
        V = J_[i, :]
        V = jnp.reshape(V, (1, num_neurons)) 

        # Woodbury identity 
        inv1 = 1 / (C_inv + jnp.matmul(V, jnp.matmul(A_inv, U))) 
        prod1 = jnp.matmul(inv1, (jnp.matmul(V, A_inv)))
        final = jnp.matmul(A_inv, U) * prod1[0, i_t]
        
        inverse = A_inv[:, i_t] - final

        v = jnp.matmul(u, inverse) # v = u (I - aJ)^-1
        sum = jnp.sum(v)
        return sum

    # Efficiently compute influence following node removal for all node in J (use GPU for larger speedup here)
    _jax_ops_partial = lambda i: jax_ops(i, num_neurons, C_inv, A_inv, i_t, J_, gain)
    jax_ops_partial = jit(_jax_ops_partial)

    sub_arrays = jnp.array_split(jnp.array(range(len(J.columns))), 145)

    vmapped = vmap(_jax_ops_partial)
    all_outputs = [] 
    for i_arr, i in enumerate(sub_arrays):
        print('batch ' + str(i_arr) + ' out of ' + str(len(sub_arrays)))
        output = vmapped(i)
        all_outputs.append(output)
    
    all_outputs = [np.float32(o) for o in all_outputs]

    outputs = np.concatenate(all_outputs)

    print('time elapsed: ' + str(time.time() - start))

    # Compute fraction of loss for each node
    nodes = jnp.array(J.columns)
    results = pd.DataFrame({'nodes': nodes, 'outputs': outputs})
    results = results.drop(index=i_t)
    results = results.drop(index=i_s)
    results['fraction of loss'] = (total - results['outputs']) / total
    results = results.sort_values(by='fraction of loss', ascending = False)

    # Save out results 
    file_prefix = exp_key+'/'
    if not os.path.isdir(file_prefix):
        os.makedirs(file_prefix)
    results.to_csv(file_prefix+'results.csv')

    return results
    