
'''

Utility classes for path-sampling algorithm in connectome data. Based on code from the Navis library (https://github.com/navis-org/navis)

'''

import multiprocessing as mp
import numpy as np
import pandas as pd

from typing import Iterable, Union, Optional, Callable


__all__ = ['TraversalModel', 'random_linear_activation_function']


class BaseNetworkModel:
    """Base model for network simulations."""

    def __init__(self, edges: pd.DataFrame, source: str, target: str):
        '''
        Initialize network model
        '''
        self.edges = edges
        self.source = source
        self.target = target


    def run_parallel(self,
                     n_cores: int = 5,
                     iterations: int = 100,
                     **kwargs: dict) -> None:
        """Run model using parallel processes."""
        with mp.Pool(processes=n_cores,
                     initializer=self.initializer) as pool:

            kwargs['iterations'] = int(iterations/n_cores)
            calls = [{**kwargs, **{'position': i}} for i in range(int(n_cores))]

            # Generate processes
            p = pool.imap_unordered(self._worker_wrapper, calls, chunksize=1)

            # Wait for processes to complete
            res = list(p)

        # Combine results
        self.results = np.concatenate(res, axis=0)
        self.iterations = iterations

    def _worker_wrapper(self, kwargs: dict):
        return self.run(**kwargs)

    def initializer(self):
        pass

    def run(self):
        pass


class TraversalModel(BaseNetworkModel):
    """Path-sampling model class
    What this does:
      1. Grab all already visited nodes (starting with ``seeds`` in step 1)
      2. Find all downstream nodes of these
      3. Probabilistically traverse based on the weight of the connecting edges
      4. Add those newly visited nodes to the pool & repeat from beginning
      5. Stop when every (connected) neuron was visited or we reached ``max_steps``
    Parameters
    ----------
    edges :             pandas.DataFrame
                        DataFrame representing an edge list. Must minimally have
                        a ``source`` and ``target`` column.
    seeds :             iterable
                        Seed nodes. Nodes that aren't found in
                        ``edges['source']`` will be (silently) removed.
    weights :           str
                        Name of a column in ``edges`` used as weights. If not
                        provided, all edges will be given a weight of 1. The weights should to be
                        between 0 and 1.
    max_steps :         int
                        Maximum path length that can be sampled

    """
    def __init__(self,
                 edges: pd.DataFrame,
                 seeds: Iterable[Union[str, int]],
                 terminals: Iterable[Union[str, int]],
                 source: str = 'source',
                 target: str = 'target',
                 weights: str = 'weight',
                 max_steps: int = 10):
        """Initialize model."""
        super().__init__(edges=edges, source=source, target=target)


        assert weights in edges.columns, f'"{weights}" must be column in edge list'

        self.terminals = terminals
        self.seeds = edges[edges[self.source].isin(seeds)][self.source].unique() #remove nonexistent seeds

        if len(self.seeds) == 0:
            raise ValueError('None of the seeds where among edge list sources.')

        self.weights = weights
        self.max_steps = max_steps



    @property
    def summary(self) -> pd.DataFrame:
        """Per-node summary."""
        return getattr(self, '_summary', self.make_summary())

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = f'{self.__class__}: {self.edges.shape[0]} edges; {self.n_nodes}' \
            f' unique nodes; {len(self.seeds)} seeds;' \
            f' traversal_func {self.traversal_func}.'
        if self.has_results:
            s += f' Model ran with {self.iterations} iterations.'
        else:
            s += ' Model has not yet been run.'
        return s



    def run(self, iterations: int = 100, **kwargs) -> None:
        """Run model (single process).
        Use ``.run_parallel`` to use parallel processes.
        """
        # For some reason this is required for progress bars in Jupyter to show
        print(' ', end='', flush=True)

        edges = self.edges[[self.source, self.target, self.weights]].values

        all_trav = None
        for it in np.arange(1, iterations + 1):

            
            
            #start with target nodes
            candidates = np.copy(self.seeds).reshape(-1, 1)


            #where sampled paths are stored
            successful_paths = []
            for i in range(2, self.max_steps + 1):
                
                out_edges = [edges[np.where(edges[:, 0] == candidates[ii, -1])[0]] for ii in range(candidates.shape[0])]
                
                #possible new paths based on extending current ones
                new_candidates = np.concatenate([np.tile(candidates[ii].reshape(-1, 1), (1, len(out_edges[ii]))).transpose() for ii in range(candidates.shape[0])], 0)
                                
                out_edges = np.concatenate(out_edges, axis=0)
                

                #probabilistic sampling
                mask = out_edges[:, 2] >= np.random.rand(out_edges.shape[0])

                
                candidates = np.concatenate([new_candidates[mask], out_edges[mask][:, 1:2]], axis=1)
                
                successful_bool = np.isin(candidates[:, -1], self.terminals)
                successful = np.where(successful_bool)[0]
                failure = np.where(np.logical_not(successful_bool))[0]
                
                #paths that reach back to source node
                successful_paths.extend(candidates[successful])
                
                #only continue extending paths that are not complete yet
                candidates = candidates[failure]
                


            # Save this round of traversal
            if all_trav is None:
                all_trav = successful_paths
            else:
                all_trav.extend(successful_paths)


        return all_trav


