from typing import Optional
from datetime import datetime
import time
import logging

from polerina.ga.engine import run_all_experiments_parallel
from polerina.ga.utils import generate_grid_params_ga, setup_path_results
from polerina.logging_utils import setup_logger

logger = logging.getLogger(__name__)

def run_hyperparameter_tuning(
        params_outputs: dict,
        params_ga: dict,
        problem_name: str,
        synthetic_data: Optional[bool] = False,
        params_graph: Optional[dict] = None,
        params_data: Optional[dict] = None,
        nb_reps_per_graph: int = 10,
        timestamp: Optional[str] = None,
        track_metrics: bool = True,
        ):
    """
    Run hyperparameter tuning for the Genetic Algorithm in parallel.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    output_path = params_outputs.get('output_path')
    path_results = setup_path_results(params_data, output_path, timestamp, problem_name=problem_name)
    params_outputs['output_path'] = path_results
    setup_logger(path_results)

    logger.info(f"=== Starting Hyperparameter Tuning Experiment ===")
    logger.info(f"Problem: {problem_name}")
    logger.info(f"Results path: {path_results}")
    logger.info(f"Repetitions per graph: {nb_reps_per_graph}")

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

    grid_param_ga = generate_grid_params_ga(params_ga)
    logger.info(f"Number of parameter combinations: {len(grid_param_ga)}")

    time_start = time.time()

    run_all_experiments_parallel(
        params_outputs=params_outputs,
        grid_param_ga=grid_param_ga,
        synthetic_data=synthetic_data,
        params_graph=params_graph,
        params_data=params_data,
        nb_reps_per_graph=nb_reps_per_graph,
        problem_name=problem_name,
        track_metrics=track_metrics,
        )
    
    compute_time = time.time() - time_start
    logger.info(f"Compute time: {compute_time:.2f} seconds")
