import numpy as np
from scipy import sparse
from numba import njit
from typing import Optional
from polerina.ga.problems.base import Problem
from time import time


@njit
def numba_repair_core_mis(individuals, indptr, indices, repair_entirely=True, random_seed=42):
    """Core JIT-compiled loop for MIS repair."""

    np.random.seed(random_seed)
    pop_size, nb_nodes = individuals.shape
    
    for indiv in range(pop_size):
        violations = np.zeros(nb_nodes, dtype=np.int32)
        for node in range(nb_nodes):
            if individuals[indiv, node] == 1:
                nb_violations = 0
                for edge_idx in range(indptr[node], indptr[node+1]):
                    neighbour = indices[edge_idx]
                    if individuals[indiv, neighbour] == 1:
                        nb_violations += 1
                violations[node] = nb_violations

        while True:
            
            max_nb_viol = 0
            for node in range(nb_nodes):
                if violations[node] > max_nb_viol:
                    max_nb_viol = violations[node]
                    
            if max_nb_viol == 0:
                break
                
            max_count = 0
            for node in range(nb_nodes):
                if violations[node] == max_nb_viol:
                    max_count += 1
                    
            chosen_tie = np.random.randint(max_count)
            idx_to_remove = -1
            curr_tie = 0
            
            for node in range(nb_nodes):
                if violations[node] == max_nb_viol:
                    if curr_tie == chosen_tie:
                        idx_to_remove = node
                        break
                    curr_tie += 1
                    
            individuals[indiv, idx_to_remove] = 0
            violations[idx_to_remove] = 0
            
            if not repair_entirely:
                break
                
            for edge_idx in range(indptr[idx_to_remove], indptr[idx_to_remove+1]):
                neighbour = indices[edge_idx]
                if individuals[indiv, neighbour] == 1:
                    violations[neighbour] -= 1
    
    return individuals

class MIS(Problem):
    @property
    def name(self) -> str:
        return "mis"

    @property
    def is_supervised(self) -> bool:
        return True

    def evaluate(self, individuals: np.ndarray, adj_matrix: np.ndarray, adj_sparse: sparse.csr_matrix) -> np.ndarray:
        # Number of internal edges * 2
        neighbours_in_set = (adj_sparse @ individuals.T).T
        internal_edges_x2 = np.sum(neighbours_in_set * individuals, axis=1)
        
        scores = - internal_edges_x2 // 2
        mask = (scores == 0)
        # If independent, score is the size of the set
        scores[mask] += individuals[mask].sum(axis=1)
        return scores

    def repair(self, individuals: np.ndarray, adj_matrix: np.ndarray, adj_sparse: sparse.csr_matrix, strategy: Optional[str], random_seed: int) -> np.ndarray:
        if strategy is None:
            return individuals
        
        repair_entirely = True if strategy == "full" else False

        return numba_repair_core_mis(
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
        elif strategy == "independent":
            indpt_sets = np.zeros((pop_size, nb_nodes), int)
            for i in range(pop_size):
                nodes_random = rng.permutation(nb_nodes)
                for node in nodes_random:
                    if np.dot(adj_matrix[node], indpt_sets[i]) == 0:
                        indpt_sets[i][node] = 1
            return indpt_sets
        else:
            raise ValueError(f"Unknown init strategy for MIS: {strategy}")

    def get_reference_label(self) -> str:
        return "True MIS Size"
