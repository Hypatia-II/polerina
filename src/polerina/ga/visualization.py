import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from typing import Optional
import networkx as nx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import logging
import warnings
from sklearn.manifold import TSNE  # Ajout pour les clusters
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import matplotlib.colors as colors
from polerina.ga.problems import get_problem

logger = logging.getLogger(__name__)


class Visualizer():
    def __init__(self, pop_size, nb_offsprings, problem_name: str, reference_value: Optional[int] = None, live_plot: bool = True, synthetic_data: Optional[bool] = False):
        self.nb_fitness_evals = []
        self.best_scores = []
        self.mean_scores = []
        self.mean_hamming_distance = []
        self.entropy = []
        self.clusters = []
        self.mds_clusters = []
        self.centroids_history = []
        self.population = None
        self.scores = None
        self.pop_size = pop_size
        self.nb_nodes = None
        self.nb_offsprings = nb_offsprings
        self.reference_value = reference_value
        self.live_plot = live_plot
        self.additional_mds_plot = "traj" # "traj" or "mds" or None
        self.additional_plot = False
        self.problem_name = problem_name
        
        try:
            self.problem = get_problem(problem_name)
        except Exception:
            self.problem = None
        
        plt.ioff() 
        
        if self.additional_plot is False:
            self.fig = plt.figure(figsize=(15, 6))
            self.ax1 = self.fig.add_subplot(1, 1, 1)
        else:
            self.fig = plt.figure(figsize=(20, 10), layout='constrained')
            gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 2], width_ratios=[2, 1])
            self.ax1 = self.fig.add_subplot(gs[0, 0])
            self.ax_cluster = self.fig.add_subplot(gs[0, 1])
            self.ax_entropy = self.fig.add_subplot(gs[1, :])
            self.cluster_scatter = None
            self.ax_cluster.set_title("Population Clusters (t-SNE)")
            self.ax_entropy.set_title("Entropy per Node (Convergence Map)")
            self.ax_entropy.set_xlabel("Number of fitness evaluations")
            self.ax_entropy.set_ylabel("Node Index")
            self.entropy_im = None
            self.entropy_cbar = None
            self.cluster_cbar = None
            self.cbar = None

            if self.additional_mds_plot =="mds":
                gs = self.fig.add_gridspec(2, 3, height_ratios=[1.1, 1], width_ratios=[4, 1, 1])
                self.ax_mds_cluster = self.fig.add_subplot(gs[0, 2])
                self.mds_cluster_cbar = None
            elif self.additional_mds_plot =="traj":
                gs = self.fig.add_gridspec(2, 3, height_ratios=[1.1, 1], width_ratios=[4, 1, 1])
                self.ax_traj = self.fig.add_subplot(gs[0, 2])
                self.ax_traj.set_title("Search Trajectory (Centroids)")

        self.ax2 = self.ax1.twinx()
        self.ax_ins = inset_axes(self.ax1, width="45%", height="45%", loc='center', borderpad=5)

        # Draw reference line only if we have a value and the problem provides a label
        if self.reference_value is not None:
            ref_label = self.problem.get_reference_label()
            self.ax1.axhline(y=self.reference_value, color='red', linestyle=':', 
                        label=f'{ref_label}({self.reference_value})')
            self.ax_ins.axhline(y=self.reference_value, color='red', linestyle=':')
            self.ax_ins.set_ylim(self.reference_value//2, self.reference_value + 5)
            
        self.line_best_score_x1, = self.ax1.plot([], [], label="Best Score", color="green")
        self.line_mean_score_x1, = self.ax1.plot([], [], label="Mean Score", color="blue", linestyle="--")
        self.line_hamming, = self.ax2.plot([], [], label="Mean Hamming Dist", color="orange", alpha=0.6)
        self.line_best_ins, = self.ax_ins.plot([], [], color="green")
        self.line_mean_score_ins, = self.ax_ins.plot([], [], color="blue", linestyle="--")
        
        self.ax1.set_title(f"GA Performance and Population Diversity - {self.problem_name.upper()}")
        self.ax1.set_xlabel("Number of fitness evaluations")
        self.ax1.set_ylabel("Fitness (Green/Blue)")
        self.ax2.set_ylabel("Mean Hamming Distance (Orange)")

        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        self.ax1.grid(True)
        self.ax_ins.grid(True)
        self.handle = None
        plt.close(self.fig)

    def update(self, metrics):

        if self.nb_nodes is None:
            self.nb_nodes = metrics["population"].shape[1]
        
        self.nb_fitness_evals.append(metrics["nb_iteration"] * self.nb_offsprings + self.pop_size)
        self.best_scores.append(metrics["scores"][-1])
        self.mean_scores.append(np.mean(metrics["scores"]))
        self.mean_hamming_distance.append(metrics["diversity"])
        
        if self.nb_fitness_evals[-1] == 40000:
            self.population = metrics["population"]
            self.scores = metrics["scores"]
        
        self.entropy.append(self._compute_entropy(metrics["population"]))
        centroide = np.mean(metrics["population"], axis=0)
        self.centroids_history.append(centroide)

        if self.live_plot:
            self._apply_data_to_plot()
            if self.additional_plot:
                if self.nb_fitness_evals[-1] % 500 == 0:
                    self.clusters = self._compute_clusters(metrics["population"]) 
                    self.scores = metrics["scores"]
                    self._update_clusters()
                    if self.additional_mds_plot == "mds":
                        self.mds_clusters = self._compute_mds(metrics["population"])
                        self._update_mds()
                    elif self.additional_mds_plot == "traj":
                        self._update_trajectory()
                self._update_entropy_heatmap()

            if self.handle is None:
                self.handle = display(self.fig, display_id=str(id(self)))
            else:
                self.handle.update(self.fig)

    def _compute_clusters(self, pop):
        if len(pop) < 3: return
        perp = min(30, len(pop) - 1)
        tsne = TSNE(n_components=2, metric='hamming', perplexity=perp, init='pca', random_state=42)
        pop_2d = tsne.fit_transform(pop)
        return pop_2d

    def _update_clusters(self):
        try:
            self.ax_cluster.clear()
            self.ax_cluster.set_title(f"Population Clusters (t-SNE)\nEval: {self.nb_fitness_evals[-1]}")
            sc = self.ax_cluster.scatter(self.clusters[:, 0], self.clusters[:, 1], 
                                         c=self.scores, cmap='turbo',
                                         s=30, alpha=0.7, vmax=self.reference_value if self.reference_value else None, vmin=0)
            if self.cbar is None:
                self.cbar = self.fig.colorbar(sc, ax=self.ax_cluster, label='Fitness',  pad=0.02)
            else:
                self.cbar.update_normal(sc)


        except Exception as e:
            print(f"Cluster visualization error: {e}")

    def _compute_mds(self, pop):
        if len(pop) < 3: return
        dist_matrix = pairwise_distances(pop, metric='hamming')
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        pop_2d = mds.fit_transform(dist_matrix)
        return pop_2d

    def _update_mds(self):
        self.ax_mds_cluster.clear()
        self.ax_mds_cluster.set_title(f"MDS Projection (Hamming)\nEval: {self.nb_fitness_evals[-1]}")
        sc = self.ax_mds_cluster.scatter(self.mds_clusters[:, 0], self.mds_clusters[:, 1], 
                                    c=self.scores, cmap='turbo', 
                                    s=40, edgecolors='white', linewidths=0.5, vmax=self.reference_value if self.reference_value else None, vmin=0)
        if self.mds_cluster_cbar is None:
            self.mds_cluster_cbar = self.fig.colorbar(sc, ax=self.ax_mds_cluster, label='Fitness', pad=0.02)
        best_idx = np.argmax(self.scores)
        self.ax_mds_cluster.annotate("Best", (self.mds_clusters[best_idx, 0], self.mds_clusters[best_idx, 1]), 
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        
    def _update_trajectory(self):
        if len(self.centroids_history) < 2: return
        step = max(1, len(self.centroids_history) // 200)
        history_sample = np.array(self.centroids_history[::step])
        try:
            dist_matrix = pairwise_distances(history_sample, metric='euclidean')
            traj_2d = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto', random_state=42).fit_transform(dist_matrix)
            self.ax_traj.clear()
            self.ax_traj.set_title(f"Centroid Trajectory\n(Subsampled every {step} gens)")
            self.ax_traj.plot(traj_2d[:, 0], traj_2d[:, 1], color='gray', alpha=0.3, zorder=1)
            sc = self.ax_traj.scatter(traj_2d[:, 0], traj_2d[:, 1], c=np.arange(len(traj_2d)), cmap='plasma', s=15, zorder=2)
            self.ax_traj.scatter(traj_2d[0, 0], traj_2d[0, 1], color='blue', marker='o', s=40, label='Start')
            self.ax_traj.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color='red', marker='*', s=100, label='Now')
        except Exception as e:
            print(f"Trajectory plotting error: {e}")

    def _compute_entropy(self, pop):
        p = np.mean(pop, axis=0)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def _update_entropy_heatmap(self):
        entropy_matrix = np.array(self.entropy).T 
        if self.entropy_im is None:
            self.entropy_im = self.ax_entropy.imshow(entropy_matrix, aspect='auto', 
                                                    cmap='magma', interpolation='nearest')
            self.fig.colorbar(self.entropy_im, ax=self.ax_entropy, label="Entropy (Bits)", fraction=0.02, pad=0.02)
        else:
            self.entropy_im.set_data(entropy_matrix)
            self.entropy_im.set_extent([self.nb_fitness_evals[0], self.nb_fitness_evals[-1], 0, self.nb_nodes])
        self.ax_entropy.relim()

    def _apply_data_to_plot(self):
        self.line_best_score_x1.set_data(self.nb_fitness_evals, self.best_scores)
        self.line_mean_score_x1.set_data(self.nb_fitness_evals, self.mean_scores)
        self.line_hamming.set_data(self.nb_fitness_evals, self.mean_hamming_distance)
        self.line_best_ins.set_data(self.nb_fitness_evals, self.best_scores)
        self.line_mean_score_ins.set_data(self.nb_fitness_evals, self.mean_scores)

        if self.best_scores:
            current_best = self.best_scores[-1]
            self.line_best_score_x1.set_label(f"Best Score: {current_best}")
            # Refresh legend
            lines1, labels1 = self.ax1.get_legend_handles_labels()
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

            if self.reference_value is None:
                self.ax_ins.set_ylim(current_best - 15, current_best + 5)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax_ins.relim()
        self.ax_ins.autoscale_view(scaley=False)

    def show_final(self):
        self._apply_data_to_plot()
        if self.additional_plot:
            self.clusters = self._compute_clusters(self.population)
            self._update_clusters()
            if self.additional_mds_plot == "mds":
                self.mds_clusters = self._compute_mds(self.population)
                self._update_mds()
            elif self.additional_mds_plot == "traj":
                self._update_trajectory()
            self._update_entropy_heatmap()
        display(self.fig, display_id=str(id(self)))
        plt.close(self.fig) # Prevent memory leak
        
    def save_plot(self, plot_path):
        self._apply_data_to_plot()
        path = Path(plot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(plot_path, bbox_inches='tight')
        plt.close(self.fig) # Prevent memory leak


def draw_graph(graph):
    nx.draw(graph)
