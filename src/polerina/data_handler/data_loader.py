import numpy as np
from typing import Optional
from .graph_utils import generate_synthetic_data
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

def load_numpy_data(path_numpy_benchmark:str, graph_id:int):

    path_numpy_data = f"{path_numpy_benchmark}/graph_id={graph_id}.npy"
    loaded_data = np.load(path_numpy_data, allow_pickle=True).tolist()
    return loaded_data


def load_data(
        params_data:Optional[dict],
        synthetic_data:Optional[bool]=False,
        params_graph:Optional[dict]=None,
        ):

    if synthetic_data:
        logger.info("Generating synthetic data...")
        dataset_size = params_graph.get('dataset_size')
        num_nodes = params_graph.get('num_nodes')
        p = params_graph.get('p')
        dataset = generate_synthetic_data(dataset_size, num_nodes, p)
        return dataset


    else:
        logger.info("Loading benchmark data...")
        path = params_data.get('path')
        dataset_name = params_data.get('dataset_name')
        dataset_split = params_data.get('dataset_split')

        path_numpy_benchmark = []
        dataset = []

        for i in range(len(dataset_name)):
            path_numpy_benchmark.append(Path(f"{path}/{dataset_name[i]}/{dataset_split}"))
            file_count = sum(1 for x in path_numpy_benchmark[i].iterdir() if x.is_file())
            logger.info(f"Total files in {dataset_name[i]}/{dataset_split}: {file_count}")
            dataset.append(np.arange(file_count))

        return path_numpy_benchmark, dataset


def sample_dataset(dataset, sample_percentage:float, random_seed:int=42):
    
    sampled_dataset = []
    nb_graphs_sampled = 0
    rng_sample = np.random.default_rng(random_seed)
    for i in range(len(dataset)):
        size_sample = int(len(dataset[i]) * sample_percentage)
        nb_graphs_sampled += size_sample
        # sampled_dataset.append(rng_sample.choice(dataset[i], size=size_sample, replace=False))
        sampled_dataset.append(dataset[i][:size_sample])
            
    return sampled_dataset, nb_graphs_sampled




    
if __name__ == "__main__":

    from polerina import DATA_DIR

    params_data = {
        "path": str(DATA_DIR / "numpy_data"),
        "dataset_name": "co_er_small",
        "dataset_split": "train",
    }

    data = load_data(params_data=params_data)
    



