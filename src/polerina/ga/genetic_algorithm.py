import numpy as np
import networkx as nx
import time
from typing import Optional, List, Dict, Any, Tuple, Callable, Literal, Annotated
from polerina.ga.utils import generate_output_metrics, compute_mean_hamming_distance
import logging
from scipy import sparse
from polerina.ga.problems.base import Problem

logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    def __init__(self,
                 adj_matrix: np.ndarray,
                 nb_nodes: int,
                 pop_size: int,
                 nb_offsprings: int,
                 problem: Problem,
                 mutation_rate: Optional[float] = None,
                 fast_mutation_upper_limit: Optional[int] = None,
                 fast_mutation_beta: Optional[float] = None,
                 random_seed: Optional[int] = None,
                 adj_sparse: Optional[sparse.csr_matrix] = None,
                 ):

        self.adj_matrix = adj_matrix
        self.adj_sparse = adj_sparse if adj_sparse is not None else sparse.csr_matrix(adj_matrix)
        self.nb_nodes = nb_nodes
        self.pop_size = pop_size
        self.nb_offsprings = nb_offsprings
        self.problem = problem
        self.prob_init = 0.25
        
        # Original heuristic for nb_iter
        self.nb_iter = int((40000 - self.pop_size) / self.nb_offsprings)
        
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        
        self.population: Optional[np.ndarray] = None
        self.score_population: Optional[np.ndarray] = None
        self.pre_offsprings: Optional[np.ndarray] = None
        self.offsprings: Optional[np.ndarray] = None
        self.score_offsprings: Optional[np.ndarray] = None
        self.best_score_history: List[int] = []  # stores native int per iteration
        self.diversity_history: List[float] = []
        self.best_individuals: Optional[np.ndarray] = None  # best individuals seen (repaired for baldwin/lb)
        self._best_individuals_set: set = set()
        self.best_max_score: Optional[int] = None

        if mutation_rate is None:
            self.mutation_rate = 1 / self.nb_nodes
        else: 
            self.mutation_rate = mutation_rate
        
        if fast_mutation_upper_limit is None:
            self.fast_mutation_upper_limit = int(self.nb_nodes / 2)
        else: 
            self.fast_mutation_upper_limit = fast_mutation_upper_limit
    
        if fast_mutation_beta is None:
            self.fast_mutation_beta = 1.5
        else: 
            self.fast_mutation_beta = fast_mutation_beta

    def uniform_crossover(self, nb_crossover: int):
        "Uniform crossover that generates offspring."
        parents_1 = self.population[self.rng.integers(self.pop_size, size=nb_crossover), :]
        parents_2 = self.population[self.rng.integers(self.pop_size, size=nb_crossover), :]
        mask = self.rng.random(size=(nb_crossover, self.nb_nodes)) < 0.5
        self.pre_offsprings = np.where(mask, parents_1, parents_2)
    
    def mutation(self):
        "Bernoulli bit-flip mutation applied to all offspring."
        mutation_mask = self.rng.random(size=(self.nb_offsprings, self.nb_nodes)) < self.mutation_rate
        self.offsprings = self.pre_offsprings ^ mutation_mask

    def fast_mutation(self):
        uniform = self.rng.uniform(size=(self.nb_offsprings, self.nb_nodes))
        exponent = 1 - self.fast_mutation_beta
        # Use np.ceil to avoid zero division if exponent is 1 (though beta is 1.5)
        self.fast_mutation_rates = np.ceil(( uniform * (self.fast_mutation_upper_limit ** exponent - 1) + 1 ) ** (1 / exponent)) / self.nb_nodes
        mutation_mask = self.rng.random(size=(self.nb_offsprings, self.nb_nodes)) < self.fast_mutation_rates
        self.offsprings = self.pre_offsprings ^ mutation_mask
    
    def individuals_skip_crossover(self, nb_crossover):
        "Sample individuals that bypass crossover and are mutated directly into offspring."
        nb_indiv_skip_crossover = self.nb_offsprings - nb_crossover
        indiv_skip_crossover = self.population[self.rng.integers(self.pop_size, size=nb_indiv_skip_crossover), :].copy()
        return indiv_skip_crossover
         
    def evaluate_potential(self, individuals):
        individuals_repaired = np.copy(individuals)
        # Lookahead always uses full repair to assess the best achievable fitness.
        individuals_repaired = self.problem.repair(individuals_repaired, self.adj_matrix, self.adj_sparse, "full", self.random_seed)
        scores = self.problem.evaluate(individuals_repaired, self.adj_matrix, self.adj_sparse)
        return scores, individuals_repaired

    def _update_best(self, candidates: np.ndarray, scores: np.ndarray):
        """Track best individuals seen so far (in their repaired form for baldwin/lb)."""
        iter_max = int(scores.max())
        if iter_max > self.best_max_score:
            self.best_max_score = iter_max
            self._best_individuals_set = {row.tobytes() for row in candidates[scores == iter_max]}
        elif iter_max == self.best_max_score:
            for row in candidates[scores == iter_max]:
                self._best_individuals_set.add(row.tobytes())

    def select(self):
        all_individuals = np.vstack((self.population, self.offsprings))
        all_scores = np.concatenate((self.score_population, self.score_offsprings))
        idx_best_individuals = np.argsort(all_scores)[-self.pop_size:]
        self.population = all_individuals[idx_best_individuals]
        self.score_population = all_scores[idx_best_individuals]
    
    def run(self,
            init_type: str,
            evolution_mode: Literal["darwin", "baldwin", "lamarck", "lb"],
            mutation_type: Literal["bernoulli", "fast_mutation"],
            crossover_rate: float,
            lamarckian_probability: Optional[Annotated[float, "strictly in (0, 1), required when evolution_mode='lb'"]] = None,
            callback: Optional[Callable] = None,
            track_metrics: bool = True,
            ) -> Tuple[dict, np.ndarray]:

        mutation_methods = {
            "bernoulli": self.mutation,
            "fast_mutation": self.fast_mutation,
            }

        # Initialize population via problem strategy
        self.population = self.problem.init_population(self.pop_size, self.nb_nodes, self.adj_matrix, init_type, self.rng, prob_init=self.prob_init)
        self.score_population = self.problem.evaluate(self.population, self.adj_matrix, self.adj_sparse)

        if track_metrics:
            self.best_max_score = int(self.score_population.max())
            self._best_individuals_set = {row.tobytes() for row in self.population[self.score_population == self.best_max_score]}
            self.best_score_history.append(self.best_max_score)

        for i in range(self.nb_iter):
            if track_metrics:
                diversity = compute_mean_hamming_distance(self.population)
                self.diversity_history.append(diversity)

                if callback:
                    callback({
                        "nb_iteration": i,
                        "scores": self.score_population,
                        "population": self.population,
                        "problem_name": self.problem.name,
                        "diversity": diversity,
                        })

            nb_crossovers = self.rng.binomial(self.nb_offsprings, crossover_rate)
            self.uniform_crossover(nb_crossovers)

            if crossover_rate != 1.0:
                indiv_skip_crossover = self.individuals_skip_crossover(nb_crossovers)
                self.pre_offsprings = np.vstack((self.pre_offsprings, indiv_skip_crossover))

            mutation_methods[mutation_type]()

            if evolution_mode == "darwin":
                self.score_offsprings = self.problem.evaluate(self.offsprings, self.adj_matrix, self.adj_sparse)
                if track_metrics:
                    self._update_best(self.offsprings, self.score_offsprings)
            elif evolution_mode == "lamarck":
                self.offsprings = self.problem.repair(self.offsprings, self.adj_matrix, self.adj_sparse, "full", self.random_seed)
                self.score_offsprings = self.problem.evaluate(self.offsprings, self.adj_matrix, self.adj_sparse)
                if track_metrics:
                    self._update_best(self.offsprings, self.score_offsprings)
            elif evolution_mode == "baldwin":
                self.score_offsprings, repaired_offsprings = self.evaluate_potential(self.offsprings)
                if track_metrics:
                    self._update_best(repaired_offsprings, self.score_offsprings)
            elif evolution_mode == "lb":
                if lamarckian_probability is None:
                    raise ValueError("lamarckian_probability must be specified when evolution_mode='lb'.")
                if not (0 < lamarckian_probability < 1):
                    raise ValueError(
                        f"lamarckian_probability must be strictly between 0 and 1, got {lamarckian_probability}."
                    )
                self.score_offsprings, repaired_offsprings = self.evaluate_potential(self.offsprings)
                mask = self.rng.random(size=self.nb_offsprings) < lamarckian_probability
                self.offsprings[mask] = repaired_offsprings[mask]
                if track_metrics:
                    self._update_best(repaired_offsprings, self.score_offsprings)
            else:
                raise ValueError(
                    f"Unknown evolution_mode: {evolution_mode!r}. "
                    "Must be one of 'darwin', 'baldwin', 'lamarck', 'lb'."
                )

            self.select()

            if track_metrics:
                self.best_score_history.append(int(self.score_population[-1]))

        if track_metrics:
            final_diversity = compute_mean_hamming_distance(self.population)
            self.diversity_history.append(final_diversity)

            if callback:
                callback({
                    "nb_iteration": self.nb_iter,
                    "scores": self.score_population,
                    "population": self.population,
                    "problem_name": self.problem.name,
                    "diversity": final_diversity,
                    })

            self.best_individuals = np.frombuffer(
                b"".join(self._best_individuals_set), dtype=self.population.dtype
            ).reshape(len(self._best_individuals_set), self.nb_nodes)

            metrics = generate_output_metrics(self.best_score_history, self.diversity_history, self.best_individuals, self.best_max_score)
        else:
            metrics = {}

        best_individual_found = self.population[-1]

        return metrics, best_individual_found
