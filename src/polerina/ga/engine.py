from polerina.ga.genetic_algorithm import GeneticAlgorithm
from polerina.ga.visualization import Visualizer
from polerina.ga.problems import get_problem
from polerina.data_handler.data_loader import load_data, sample_dataset, load_numpy_data
from typing import Optional
from polerina.ga.utils import save_results_parquet # save_results
from polerina.logging_utils import setup_logger
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import sparse
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def run_experiment(adj_matrix, nb_nodes, params_ga, random_seed, problem_name, reference_value=None, synthetic_data=None, plot_path=None, plot_display=None, problem=None, adj_sparse=None, track_metrics=True):

    pop_size = params_ga.get('pop_size')
    nb_offsprings = params_ga.get('nb_offsprings')
    init_type = params_ga.get('init_type')
    evolution_mode = params_ga.get('evolution_mode')
    mutation_type = params_ga.get('mutation_type')
    crossover_rate = params_ga.get('crossover_rate')
    lamarckian_probability = params_ga.get('lamarckian_probability')



    if plot_path:
        if plot_display == "live_plot":
            live_plot = True
        else:
            live_plot = False
        visualizer = Visualizer(pop_size, nb_offsprings, reference_value=reference_value, live_plot=live_plot, synthetic_data=synthetic_data, problem_name=problem_name)
        callback_func = visualizer.update
    else:
        callback_func = None

    if problem is None:
        problem = get_problem(problem_name)

    ga = GeneticAlgorithm(
        adj_matrix=adj_matrix,
        nb_nodes=nb_nodes,
        pop_size=pop_size,
        nb_offsprings=nb_offsprings,
        problem=problem,
        random_seed=random_seed,
        adj_sparse=adj_sparse,
        )
    
    t_start = time.perf_counter()
    metrics, max_set_found = ga.run(
        init_type=init_type,
        evolution_mode=evolution_mode,
        mutation_type=mutation_type,
        crossover_rate=crossover_rate,
        lamarckian_probability=lamarckian_probability,
        callback=callback_func,
        track_metrics=track_metrics,
        )
    metrics["ga_runtime_seconds"] = time.perf_counter() - t_start

    if plot_display == "not_live":
        visualizer.show_final()
    elif plot_display == "not_displayed":
        pass
    
    if plot_path:
        visualizer.save_plot(plot_path=plot_path)
    
    return metrics, max_set_found




def single_run_worker(data, count_set, param_ga, rep, output_path, dataset_name, problem_name, params_outputs, synthetic_data=None, adj_sparse=None, track_metrics=True):
    
    visualize = params_outputs.get('visualize')
    if visualize[0] and visualize[1] > rep:
        if synthetic_data is False:
            plot_path = f"{output_path}/plots/{dataset_name}/graph_{data['graph_id']}_set_param_{count_set}_rep_{rep}.png"
        else:
            plot_path = f"{output_path}/plots/graph_{data['graph_id']}_set_param_{count_set}_rep_{rep}.png"
        plot_display = params_outputs.get('plot_display')
    else:
        plot_path = None
        plot_display = None

    random_seed = 42 + rep

    problem = get_problem(problem_name)
    reference_value = data["reference_value"] if problem.is_supervised else None

    metrics, _ = run_experiment(
        data["adj_matrix"],
        data["nb_nodes"],
        params_ga=param_ga,
        random_seed=random_seed,
        problem_name=problem_name,
        reference_value=reference_value,
        synthetic_data=synthetic_data,
        plot_path=plot_path,
        plot_display=plot_display,
        problem=problem,
        adj_sparse=adj_sparse,
        track_metrics=track_metrics,
        )
    

    return {
        "graph_id": data["graph_id"],
        "repetition": rep,
        "nb_nodes": data["nb_nodes"],
        "true_size_set": reference_value,
        **metrics, # "max_score_found", "score_init", "improvements"
        **{f"param_{k}": v for k, v in param_ga.items()}, # Flatten param_ga: {"pop_size": 100} becomes "param_pop_size": 100
        "plot_path": plot_path,
        "random_seed": random_seed, 
        "dataset_name": dataset_name,
        "problem": problem_name
    }

def big_worker(payload, nb_reps_per_graph, grid_param_ga, output_path, params_outputs, synthetic_data, problem_name, track_metrics=True):
    
    # Initialize logging for the worker process
    setup_logger(output_path)

    if synthetic_data is False:
        data = load_numpy_data(payload["path_data"], payload["data_id"])
        dataset_name = payload["dataset_name"]
    else: 
        data = payload["data"]
        dataset_name = "synthetic"


    results_for_this_data = []
    graph_id = data['graph_id']

    # Check if results already exist for this graph
    output_dir = Path(output_path) / "parquet" / f"dataset_name={dataset_name}" / f"graph_id={graph_id}"
    parquet_file = output_dir / "results.parquet"
    if parquet_file.exists():
        log_file = Path(output_path) / "experiment.log"
        finish_marker = f"Finished graph analysis: Graph ID {graph_id}."
        if log_file.exists() and finish_marker in log_file.read_text():
            logger.info(f"Skipping graph analysis: Graph ID {graph_id} (Dataset: {dataset_name}) - Results already exist.")
            return []
        else:
            incomplete_file = parquet_file.with_suffix(".incomplete")
            parquet_file.rename(incomplete_file)
            logger.warning(
                f"Graph ID {graph_id} (Dataset: {dataset_name}): parquet found but no finish log entry — "
                f"renamed to {incomplete_file.name} and rerunning."
            )

    logger.debug(f"Starting graph analysis: Graph ID {graph_id} (Dataset: {dataset_name})")

    adj_sparse = sparse.csr_matrix(data["adj_matrix"])

    for count_set, param_ga in enumerate(grid_param_ga):

        for rep in range(nb_reps_per_graph):

            res = single_run_worker(data,
                                    count_set,
                                    param_ga,
                                    rep,
                                    output_path,
                                    dataset_name,
                                    problem_name,
                                    params_outputs,
                                    synthetic_data,
                                    adj_sparse=adj_sparse,
                                    track_metrics=track_metrics,
                                    )
            results_for_this_data.append(res)
    

    save_results_parquet(results_for_this_data, output_path, dataset_name, graph_id)  
 

    logger.info(f"Finished graph analysis: Graph ID {graph_id}. Results saved.")
            
    return results_for_this_data


def run_all_experiments_parallel(
        params_outputs: dict, #  "output_path", "visualize", "plot_display"
        grid_param_ga: dict,
        synthetic_data:Optional[bool],
        params_graph:Optional[dict],
        params_data:Optional[dict],
        nb_reps_per_graph:int,
        problem_name: str,
        track_metrics: bool = True,
        ):
    
    if params_data and synthetic_data is False:
        dataset_names = params_data["dataset_name"]
        logger.info(f"Loading dataset info for: {', '.join(dataset_names)}")
        path_numpy_benchmark, dataset = load_data(params_data, synthetic_data)
        sample = params_data.get('sample', None)
        if sample:
            logger.info(f"Sampling dataset (rate: {sample})...")
            dataset, nb_graphs_sampled = sample_dataset(dataset, sample, random_seed=42)
            logger.info(f"Number of graphs sampled: {nb_graphs_sampled}")
    elif synthetic_data:
        dataset = load_data(params_data, synthetic_data, params_graph)
        logger.info(f"Synthetic dataset loaded ({params_graph.get('num_nodes')} nodes, p={params_graph.get('p')})")

    output_path = params_outputs.get('output_path')

    if synthetic_data is False:
        total_tasks = sum(len(subdata) for subdata in dataset)
    else:
        total_tasks = len(dataset)

    logger.info(f"Total number of graphs to process: {total_tasks}")
    logger.info(f"Number of repetition per graph: {nb_reps_per_graph}")
    logger.info(f"Starting parallel execution of {total_tasks} tasks...")
 
    if synthetic_data is False:
        task_generator = (
            delayed(big_worker)(
                {"data_id": subdata_id, "path_data": subpath_numpy_benchmark, "dataset_name": dataset_name},
                nb_reps_per_graph,
                grid_param_ga,
                output_path,
                params_outputs,
                synthetic_data,
                problem_name,
                track_metrics)
            for subdata, subpath_numpy_benchmark, dataset_name in zip(dataset, path_numpy_benchmark, dataset_names)
            for subdata_id in subdata
        )
    else:
        task_generator = (
            delayed(big_worker)({"data": data}, nb_reps_per_graph, grid_param_ga, output_path, params_outputs, synthetic_data, problem_name, track_metrics)
            for data in dataset
        )
    
    with Parallel(n_jobs=-1, batch_size=1, return_as='generator', verbose=10) as parallel:
        result_iterator = parallel(task_generator)
        for _ in tqdm(result_iterator, total=total_tasks, desc="Running experiments"):
            pass


    logger.info("All experiments completed!")


