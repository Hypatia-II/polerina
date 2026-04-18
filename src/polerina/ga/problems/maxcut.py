import numpy as np
from scipy import sparse
from typing import Optional
from polerina.ga.problems.base import Problem

from time import time
import numpy as np
import scipy.sparse as sparse
from numba import njit


@njit
def numba_repair_core_maxcut(individuals, indptr, indices, repair_entirely=True, random_seed=42):

    """
    Highly optimized JIT-compiled loop for Max Cut greedy local search.
    Features O(1) active list updates, correct gain math, and random tie-breaking.
    """
    np.random.seed(random_seed)
    pop_size, nb_nodes = individuals.shape
    
    # Pre-allocate arrays ONCE outside the population loop to avoid overhead
    gains = np.empty(nb_nodes, dtype=np.int32)
    active_nodes = np.empty(nb_nodes, dtype=np.int32)
    node_pos = np.empty(nb_nodes, dtype=np.int32) # Tracks position in active_nodes, -1 if inactive
    tied_nodes = np.empty(nb_nodes, dtype=np.int32)

    for indiv in range(pop_size):
        # Reset position tracking for this individual
        node_pos[:] = -1
        num_active = 0
        
        # 1. Initial Gain Calculation
        for node in range(nb_nodes):
            x = individuals[indiv, node]
            same_set_neighbors = 0
            
            for edge_idx in range(indptr[node], indptr[node+1]):
                if individuals[indiv, indices[edge_idx]] == x:
                    same_set_neighbors += 1
            
            degree = indptr[node+1] - indptr[node]
            diff_set_neighbors = degree - same_set_neighbors
            
            # Gain = edges gained (same) - edges lost (diff)
            gains[node] = same_set_neighbors - diff_set_neighbors
            
            # If flipping improves the cut, add to the active list
            if gains[node] > 0:
                active_nodes[num_active] = node
                node_pos[node] = num_active
                num_active += 1

        # 2. Local Search Loop
        while num_active > 0:
            max_gain = 0
            num_ties = 0
            
            # SINGLE PASS: Find max gain and collect all tied nodes among active nodes
            for i in range(num_active):
                node = active_nodes[i]
                g = gains[node]
                
                if g > max_gain:
                    max_gain = g
                    tied_nodes[0] = node
                    num_ties = 1
                elif g == max_gain:
                    tied_nodes[num_ties] = node
                    num_ties += 1
            
            # Safety break (should not trigger since active list strictly has gain > 0)
            if max_gain <= 0 or num_ties == 0:
                break
            
            # Randomly select ONE of the tied nodes
            if num_ties == 1:
                idx_to_flip = tied_nodes[0]
            else:
                idx_to_flip = tied_nodes[np.random.randint(num_ties)]
            
            # 3. Apply the Flip
            old_val = individuals[indiv, idx_to_flip]
            individuals[indiv, idx_to_flip] = 1 - old_val
            
            # The node's gain inverts
            gains[idx_to_flip] = -gains[idx_to_flip]
            
            # O(1) Removal of flipped node from active list
            pos_to_remove = node_pos[idx_to_flip]
            last_active_node = active_nodes[num_active - 1]
            
            active_nodes[pos_to_remove] = last_active_node
            node_pos[last_active_node] = pos_to_remove
            num_active -= 1
            node_pos[idx_to_flip] = -1
            
            # 4. Update Neighbors in O(Degree) Time
            for edge_idx in range(indptr[idx_to_flip], indptr[idx_to_flip+1]):
                neighbor = indices[edge_idx]
                
                # If they WERE the same, they are now DIFFERENT. 
                # Flipping neighbor now gains less -> subtract 2.
                if individuals[indiv, neighbor] == old_val:
                    gains[neighbor] -= 2
                # If they WERE different, they are now the SAME. 
                # Flipping neighbor now gains more -> add 2.
                else:
                    gains[neighbor] += 2
                
                # Dynamic O(1) Active List Management for the neighbor
                if gains[neighbor] > 0 and node_pos[neighbor] == -1:
                    # Neighbor became active
                    active_nodes[num_active] = neighbor
                    node_pos[neighbor] = num_active
                    num_active += 1
                elif gains[neighbor] <= 0 and node_pos[neighbor] != -1:
                    # Neighbor became inactive
                    pos_to_remove_nb = node_pos[neighbor]
                    last_active_nb = active_nodes[num_active - 1]
                    
                    active_nodes[pos_to_remove_nb] = last_active_nb
                    node_pos[last_active_nb] = pos_to_remove_nb
                    num_active -= 1
                    node_pos[neighbor] = -1
            
            # If not doing full local search, break after one flip
            if not repair_entirely:
                break
                
    return individuals

class MaxCut(Problem):
    @property
    def name(self) -> str:
        return "maxcut"

    @property
    def is_supervised(self) -> bool:
        return False

    def evaluate(self, individuals: np.ndarray, adj_matrix: np.ndarray, adj_sparse: sparse.csr_matrix) -> np.ndarray:

        neigbours_with_set_1 = adj_sparse @ individuals.T
        return np.sum((1 - individuals) * neigbours_with_set_1.T, axis=1)

    def repair(self, individuals: np.ndarray, adj_matrix: np.ndarray, adj_sparse: sparse.csr_matrix, strategy: Optional[str], random_seed: int) -> np.ndarray:
        """
        Max-Cut is unconstrained, so no repair is strictly needed.
        'repair' could be used for local search (greedy improvement).
        """

        if strategy is None:
            return individuals
        
        repair_entirely = True if strategy == "full" else False

        return numba_repair_core_maxcut(
            individuals, 
            adj_sparse.indptr, 
            adj_sparse.indices, 
            repair_entirely, 
            random_seed
        )


    def init_population(self, pop_size: int, nb_nodes: int, adj_matrix: np.ndarray, strategy: str, rng: np.random.Generator, prob_init: float) -> np.ndarray:
        if strategy == "random":
            return rng.integers(0, 2, (pop_size, nb_nodes))
        elif strategy == "prob":
            return rng.choice([0, 1], size=(pop_size, nb_nodes), p=[1 - prob_init, prob_init])
        else:
            raise ValueError(f"Unknown init strategy for Max-Cut: {strategy}")

    def get_reference_label(self) -> Optional[str]:
        return None
