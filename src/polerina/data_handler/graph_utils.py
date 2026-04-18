import networkx as nx
import numpy as np
from networkx.algorithms import approximation as approx
import logging

logger = logging.getLogger(__name__)



def init_graph(num_nodes, p):

    graph_nx = nx.fast_gnp_random_graph(num_nodes, p)
    while not nx.is_connected(graph_nx):
        graph_nx = nx.fast_gnp_random_graph(num_nodes, p)
    return graph_nx


def generate_synthetic_data(dataset_size, num_nodes, p):

    dataset = []
    for i in range(dataset_size):
        graph = init_graph(num_nodes, p)
        adj_matrix = nx.to_numpy_array(graph, dtype=int)
        nb_nodes = graph.number_of_nodes()
        reference_value = approx.maximum_independent_set(graph)
        dataset.append({
            "graph_id": i,
            "adj_matrix": adj_matrix, 
            "nb_nodes": nb_nodes, 
            "reference_value": len(reference_value)
                        })
    return dataset


def edge_index_to_adj(edge_index, num_nodes):
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    edge_index = edge_index.cpu().numpy()
    adj[edge_index[0], edge_index[1]] = 1
    return adj