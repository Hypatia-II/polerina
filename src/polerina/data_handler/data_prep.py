import os
import graphbench
from typing import Optional
import numpy as np
import shutil
from .graph_utils import edge_index_to_adj
import logging

logger = logging.getLogger(__name__)


def load_datasets_benchmark(root, dataset_name):
    
    if not os.path.exists(root):
        os.makedirs(root)
    
    loader = graphbench.Loader(root, dataset_name)
    datasets = loader.load()
    return datasets



def save_datasets_to_numpy(path_numpy_benchmark:Optional[str], datasets:list, splits:Optional[list]=None):

    if splits is None:
        raise ValueError("splits must be provided, e.g. ['train'], ['test'], or ['train', 'valid', 'test']")

    if not os.path.exists(path_numpy_benchmark):
        os.makedirs(path_numpy_benchmark)

    for split in splits:
        
        split_path = os.path.join(path_numpy_benchmark, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        
        for count, data in enumerate(datasets[0][split]):
            item = {
                "graph_id": count,
                "adj_matrix": edge_index_to_adj(data.edge_index, data.num_nodes),
                "nb_nodes": int(data.num_nodes),
                "mis_solution": data.mis_solution.detach().cpu().numpy(),
                "reference_value": int(data.num_mis.detach().cpu().item()) 
                }
            
            file_path = os.path.join(split_path, f"graph_id={count}.npy")
            np.save(file_path, item, allow_pickle=True)


            if count % 1000 == 0:
                print(f"Progression {split} data saved to numpy: {count}/{len(datasets[0][split])}")
            del item
        


def delete_datasets_benchmark(path_data_benchmark:Optional[str]):
    
    if os.path.isdir(path_data_benchmark):
        shutil.rmtree(path_data_benchmark)


def run_benchmark_conversion_pipeline(path_data_benchmark, root_numpy_benchmark, dataset_name, splits, delete_raw=True):

    datasets = load_datasets_benchmark(path_data_benchmark, dataset_name)
    print("finished loading datasets")
    path_numpy_benchmark = f"{root_numpy_benchmark}/{dataset_name}"
    save_datasets_to_numpy(path_numpy_benchmark, datasets, splits=splits)
    
    if delete_raw and os.path.isdir(path_data_benchmark):
        delete_datasets_benchmark(path_data_benchmark)


if __name__ == "__main__":

    from polerina import DATA_DIR

    path_data_benchmark = str(DATA_DIR / "benchmark_data")
    root_numpy_benchmark = str(DATA_DIR / "numpy_data")
    dataset_name = "co_er_small"
    path_numpy_benchmark = f"{root_numpy_benchmark}/{dataset_name}"


    print(f"--- Starting conversion for {dataset_name} ---")
    run_benchmark_conversion_pipeline(path_data_benchmark, root_numpy_benchmark, dataset_name, delete_raw=True)
    print(f"--- Conversion complete. Data saved to {path_numpy_benchmark} ---")