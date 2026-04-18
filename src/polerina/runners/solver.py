from typing import Optional
from datetime import datetime
import logging
import numpy as np

from polerina.data_handler.data_loader import load_data, sample_dataset, load_numpy_data
from polerina.ga.engine import run_experiment
from polerina.ga.utils import save_results_parquet, setup_path_results
from polerina.logging_utils import setup_logger
from polerina.ga.problems import get_problem

logger = logging.getLogger(__name__)

def run_solver(
        params_outputs: dict, #  "output_path", "visualize", "plot_display"
        problem_name: str,
        params_ga=None,
        synthetic_data: Optional[bool] = False,
        params_graph: Optional[dict] = None,
        params_data: Optional[dict] = None,
        ):
    """
    Execute the solver on a dataset.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = params_outputs.get('output_path')
    path_results = setup_path_results(params_data, output_path, timestamp, problem_name=problem_name)
    params_outputs['output_path'] = path_results
    setup_logger(path_results)

    logger.info(f"=== Starting Solver ===")
    logger.info(f"Problem: {problem_name}")
    logger.info(f"Results path: {path_results}")

    if params_ga:
        logger.info("--- GA Parameters ---")
        for k, v in params_ga.items():
            logger.info(f"  {k}: {v}")

    if params_data:
        logger.info("--- Data Parameters ---")
        for k, v in params_data.items():
            logger.info(f"  {k}: {v}")

    if params_graph:
        logger.info("--- Graph Parameters ---")
        for k, v in params_graph.items():
            logger.info(f"  {k}: {v}")

    logger.info("--- Output Parameters ---")
    for k, v in params_outputs.items():
        logger.info(f"  {k}: {v}")

    problem = get_problem(problem_name)

    if params_data and synthetic_data is False:
        dataset_names = params_data["dataset_name"]
        path_numpy_benchmark, dataset = load_data(params_data, synthetic_data)
        sample = params_data.get('sample', None)
        depth = len(dataset_names)
        if sample:
            logger.info("Sampling dataset...")
            dataset, nb_graphs_sampled = sample_dataset(dataset, sample, random_seed=42)
            logger.info(f"Number of graph types: {len(dataset)}")
            logger.info(f"Number of graphs sampled: {nb_graphs_sampled}")

    elif synthetic_data:
        dataset = [load_data(params_data, synthetic_data, params_graph)]
        depth = 1


    visualize = params_outputs.get('visualize')

    if params_ga is None:
        params_ga = {}
    
    for i in range(depth):
        for item in dataset[i]: # "graph_id", "adj_matrix", "nb_nodes", "mis_solution", "reference_value" // or just a list of data_id
            data = item if synthetic_data else load_numpy_data(path_numpy_benchmark[i], item)


            adj_matrix = data["adj_matrix"]
            nb_nodes = data["nb_nodes"]
            reference_value = data["reference_value"] if problem.is_supervised else None
            all_results_one_graph = []

            random_seed = 42
                    
            if visualize[0]:
                if synthetic_data is False:
                    plot_path = f"{path_results}/plots/{dataset_names[i]}/graph_{data['graph_id']}.png"
                else:
                    plot_path = f"{path_results}/plots/graph_{data['graph_id']}.png"
                plot_display = params_outputs.get('plot_display')
            else:
                plot_path = None
                plot_display = None
            
            metrics, max_found = run_experiment(
                adj_matrix,
                nb_nodes,
                params_ga=params_ga,
                random_seed=random_seed,
                problem_name=problem_name,
                reference_value=reference_value,
                synthetic_data=synthetic_data,
                plot_path=plot_path,
                plot_display=plot_display
                ) 
            
            results_entry = {
                "graph_id": data["graph_id"],
                "nb_nodes": nb_nodes,
                "reference_value": reference_value,
                **metrics, # "max_score_found", "score_init", "improvements"
                **{f"param_{k}": v for k, v in params_ga.items()}, # Flatten set_param_ga: {"pop_size": 100} becomes "param_pop_size": 100
                "plot_path": plot_path,
                "random_seed": random_seed,
                "problem": problem_name
            }

            all_results_one_graph.append(results_entry)

            if synthetic_data is False:
                save_results_parquet(all_results_one_graph, path_results, dataset_names[i], data["graph_id"])    # Save results function that adds results to the same file 
            else:
                save_results_parquet(all_results_one_graph, path_results, "synthetic", data["graph_id"])    # Save results function that adds results to the same file 


def test_solver(
        params_outputs: dict, #  "output_path", "visualize", "plot_display"
        problem_name: str,
        params_ga=None,
        synthetic_data: Optional[bool] = False,
        params_graph: Optional[dict] = None,
        params_data: Optional[dict] = None,
        ):
    """
    Test the solver for reproducibility: runs one random graph from each dataset 10 times with the same seed.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = params_outputs.get('output_path')
    path_results = setup_path_results(params_data, output_path, timestamp, problem_name=problem_name)
    params_outputs['output_path'] = path_results
    setup_logger(path_results)

    problem = get_problem(problem_name)

    if params_data and synthetic_data is False:
        dataset_names = params_data["dataset_name"]
        path_numpy_benchmark, dataset = load_data(params_data, synthetic_data)
        depth = len(dataset_names)
    elif synthetic_data:
        dataset = [load_data(params_data, synthetic_data, params_graph)]
        depth = 1
        dataset_names = ["synthetic"]
      
    visualize = params_outputs.get('visualize')

    if params_ga is None:
        params_ga = {}
    
    rng = np.random.default_rng(42) # Fixed seed for the selection itself
    
    for i in range(depth):
        # Pick a random graph from this dataset
        random_idx = rng.integers(len(dataset[i]))
        item = dataset[i][random_idx]
        
        data = item if synthetic_data else load_numpy_data(path_numpy_benchmark[i], item)   

        adj_matrix = data["adj_matrix"]
        nb_nodes = data["nb_nodes"]
        reference_value = data["reference_value"] if problem.is_supervised else None
        
        logger.info(f"=== Reproducibility Test | Dataset: {dataset_names[i]} | Graph ID: {data['graph_id']} ===")
        
        results = []
        fixed_seed = 42
        
        for run_idx in range(10):
            metrics, _ = run_experiment(
                adj_matrix,
                nb_nodes,
                params_ga=params_ga,
                random_seed=fixed_seed,
                problem_name=problem_name,
                reference_value=reference_value,
                synthetic_data=synthetic_data,
                plot_path=None,
                plot_display="not_displayed"
                ) 
            
            results.append(metrics["max_score_found"])
            logger.info(f"  Run {run_idx+1}/10 | Score: {metrics['max_score_found']}")

        # Verification
        all_identical = all(res == results[0] for res in results)
        if all_identical:
            logger.info(f"SUCCESS: Results are 100% deterministic for {dataset_names[i]}")
        else:
            logger.error(f"FAILURE: Results vary for {dataset_names[i]} despite using the same seed!")
            for idx, res in enumerate(results):
                logger.error(f"  Run {idx+1}: {res}")
