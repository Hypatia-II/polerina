from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Literal

class Problem(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the problem (e.g., 'mis', 'maxcut')."""
        pass

    @property
    @abstractmethod
    def is_supervised(self) -> bool:
        """Whether the problem has a known reference value/ground truth."""
        pass

    @abstractmethod
    def evaluate(self, individuals: np.ndarray, adj_matrix: np.ndarray, adj_sparse: sparse.csr_matrix) -> np.ndarray:
        """Calculate fitness scores for a population."""
        pass

    @abstractmethod
    def repair(self, individuals: np.ndarray, adj_matrix: np.ndarray, adj_sparse: sparse.csr_matrix, strategy: Optional[str], random_seed: int) -> np.ndarray:
        """Fix or improve individuals."""
        pass

    @abstractmethod
    def init_population(self, pop_size: int, nb_nodes: int, adj_matrix: np.ndarray, strategy: str, rng: np.random.Generator, prob_init: float) -> np.ndarray:
        """Initialize the population using problem-specific heuristics."""
        pass

    @abstractmethod
    def get_reference_label(self) -> Optional[str]:
        """Return the label for the reference value in plots, or None if unsupervised."""
        pass
