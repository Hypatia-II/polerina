import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sys
from itertools import islice
from pathlib import Path
import itertools
import logging

logger = logging.getLogger(__name__)



def random_argmax(values, axis=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.random(values.shape) * 1e-6
    return np.argmax(values + noise, axis=axis)



def save_results_parquet(results_list, output_path, dataset_name, graph_id):
    

    output_dir = Path(output_path) / "parquet" / f"dataset_name={dataset_name}" / f"graph_id={graph_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir/ "results.parquet"

    table = pa.Table.from_pylist(results_list)
    pq.write_table(table, file_path, compression='zstd')

    # logging.info(f"File saved at {file_path}")

def compute_mean_hamming_distance(population):

    pop_size, nb_nodes = population.shape
    p = population.sum(axis=0)

    total_dist = (p * (pop_size - p)).sum()
    mean_hamming_distance = 2 * (total_dist) / (pop_size * (pop_size - 1))

    return float(mean_hamming_distance)


def find_improvements(best_score_history):
    """
    Generator that yields indices and values where a score 
    is higher than the previous one, searching backwards.
    """

    for i in range(len(best_score_history) - 1, 0, -1):
         if best_score_history[i] > best_score_history[i-1]:
             yield {"index": i, "score": int(best_score_history[i])} # index here is really the iteration rather than (pop_size + iteration * nb_offsprings)


def generate_output_metrics(best_score_history, diversity_history, best_individuals, best_max_score):

    improvements = list(islice(find_improvements(best_score_history), 5))
    max_score = int(best_score_history[-1])
    nb_unique_optimal_solutions = len(best_individuals)

    metrics = {
        "max_score_found": max_score,
        "score_init": int(best_score_history[0]),
        "improvements": improvements,
        "best_score_history": best_score_history,
        "nb_unique_optimal_solutions": nb_unique_optimal_solutions,
        "diversity_history": diversity_history,
        "best_max_score_ever": best_max_score
    }

    return metrics


def generate_grid_params_ga(params_ga):
    lb_probs = params_ga.get("lamarckian_probability", None)
    base_params = {k: v for k, v in params_ga.items() if k != "lamarckian_probability"}

    keys = list(base_params.keys())
    values = list(base_params.values())
    base_grid = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if lb_probs is None:
        return base_grid

    result = []
    for combo in base_grid:
        if combo["evolution_mode"] == "lb":
            for p in lb_probs:
                result.append({**combo, "lamarckian_probability": p})
        else:
            result.append({**combo, "lamarckian_probability": None})
    return result


def setup_path_results(params_data, output_path, timestamp, problem_name):

    if params_data:
        dataset_name = params_data.get('dataset_name')
        dataset_split = params_data.get('dataset_split')
        folder_name = "__".join(dataset_name)
        path_results = f"{output_path}/{problem_name}/{folder_name}/{dataset_split}/results_{timestamp}"

    else:
        path_results = f"{output_path}/{problem_name}/synthetic_data/results_{timestamp}"
    
    return path_results





