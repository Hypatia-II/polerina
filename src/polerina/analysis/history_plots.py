"""
Pipeline: mean convergence and diversity curves per evolution strategy, using the
best parameter combination per (dataset_name, evolution_mode, problem).

Usage:
    uv run python -m polerina.analysis.history_plots \\
        --path-results results/20240101_1200_mis \\
        --problem mis \\
        --output-dir plots/convergence \\
        --supervised/--no-supervised
"""

import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import logging
from pathlib import Path
from polerina.analysis.aggregate_results_unified import (
    validate_data,
    load_and_preprocess_results,
    get_best_results_graph_agg,
    get_param_cols,
)
from polerina.analysis.latex_tables import DATASET_LABELS_LONG, DATASET_ORDER

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

logger = logging.getLogger(__name__)

PROBLEM_LABELS = {
    "maxcut": "MC",
    "mis":    "MIS",
}

STRATEGY_COLORS = {
    "darwin":    "#1f77b4",  # blue
    "baldwin":   "#2ca02c",  # green
    "lamarck":   "#d62728",  # red
    "lb":        "#ff7f0e",  # orange
}

STRATEGY_LINESTYLES = {
    "darwin":  "-",
    "baldwin": "--",
    "lamarck": "-.",
    "lb":      ":",
}


def _format_label(strategy: str, best_params: dict, n_graphs: int) -> str:
    """Build legend label: strategy name + best hyperparams + graph count."""
    name = STRATEGY_LABELS.get(strategy, strategy.capitalize())
    pop  = best_params.get("param_pop_size", "?")
    lam  = best_params.get("param_nb_offsprings", "?")
    cr   = best_params.get("param_crossover_rate", "?")
    cr_str = f"{cr:.1f}" if isinstance(cr, float) else cr
    params = f"μ={pop}/λ={lam}/rc={cr_str}"
    plb = best_params.get("param_lamarckian_probability")
    if strategy == "lb" and plb is not None:
        params += f"/p={plb:.2g}"
    return f"{name} ({params}, n={n_graphs})"

STRATEGY_LABELS = {
    "darwin":    "Darwinian",
    "baldwin":   "Baldwinian",
    "lamarck":   "Lamarckian",
    "lb":        "L-B",
}


def _load_filtered(
    path_results: str,
    problem: str,
    supervised: bool,
    extra_cols: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """
    Shared setup: validate, preprocess, find best params, semi-join.

    Returns (filtered_df, best_df, param_cols).
    extra_cols: history columns to keep (e.g. ["best_score_history"]).
    """
    validate_data(path_results, problem)
    results = load_and_preprocess_results(path_results, problem)

    has_lb_col = "param_lamarckian_probability" in results.schema
    param_cols = get_param_cols(has_lb_col)

    best_df = get_best_results_graph_agg(results, problem, supervised).collect()

    join_keys = [*param_cols, "dataset_name"]

    base_cols = [
        "dataset_name", "param_evolution_mode",
        "graph_id", "repetition",
        "param_pop_size", "param_nb_offsprings",
    ]
    select_cols = base_cols + [c for c in extra_cols if c not in base_cols]

    filtered = (
        results
        .join(best_df.lazy().select(join_keys), on=join_keys, how="semi", join_nulls=True)
        .select([c for c in select_cols if c in results.schema])
        .collect()
    )
    return filtered, best_df, param_cols


def _aggregate_histories(
    filtered: pl.DataFrame,
    best_df: pl.DataFrame,
    param_cols: list[str],
    history_col: str,
    trim_infeasible: bool,
) -> dict:
    """
    Core aggregation shared by convergence and diversity.

    Per graph: mean and std over repetitions.
    Across graphs: mean of per-graph means (line) and mean of per-graph stds (band).

    trim_infeasible: skip leading steps where the mean curve is < 0 (convergence only).

    Returns nested dict::

        {
          dataset_name: {
            evolution_mode: {
              "fitness_evals": np.ndarray,
              "mean_history":  np.ndarray,
              "std_history":   np.ndarray,
              "n_graphs":      int,
              "best_params":   dict,
            }
          }
        }
    """
    data: dict = {}

    for keys, group in filtered.group_by(["dataset_name", "param_evolution_mode"]):
        dataset_name, evolution_mode = keys

        pop_size      = int(group["param_pop_size"][0])
        nb_offsprings = int(group["param_nb_offsprings"][0])

        unit_means, unit_stds = [], []
        for (_graph_id,), graph_group in group.group_by(["graph_id"]):
            histories = graph_group[history_col].to_list()
            min_len_g = min(len(h) for h in histories)
            arr_g = np.array([h[1:min_len_g] for h in histories], dtype=float)
            unit_means.append(arr_g.mean(axis=0))
            unit_stds.append(arr_g.std(axis=0))

        min_len  = min(min(len(h) for h in unit_means), min(len(h) for h in unit_stds))
        mean_arr = np.array([h[:min_len] for h in unit_means], dtype=float)
        std_arr  = np.array([h[:min_len] for h in unit_stds],  dtype=float)

        fitness_evals = np.array([pop_size + i * nb_offsprings for i in range(1, min_len + 1)])

        if trim_infeasible:
            first_feasible = int(np.argmax(mean_arr.mean(axis=0) >= 0))
            if first_feasible > 0:
                mean_arr      = mean_arr[:, first_feasible:]
                std_arr       = std_arr[:, first_feasible:]
                fitness_evals = fitness_evals[first_feasible:]

        best_row = best_df.filter(
            (pl.col("dataset_name") == dataset_name) &
            (pl.col("param_evolution_mode") == evolution_mode)
        )
        best_params = {col: best_row[col][0] for col in param_cols if col in best_row.columns}

        data.setdefault(dataset_name, {})[evolution_mode] = {
            "fitness_evals": fitness_evals,
            "mean_history":  mean_arr.mean(axis=0),
            "std_history":   std_arr.mean(axis=0),
            "n_graphs":      len(unit_means),
            "best_params":   best_params,
        }
        logger.info(
            f"  {dataset_name} / {evolution_mode}: "
            f"{len(unit_means)} graphs, {len(fitness_evals)} steps"
        )

    return data


NCOLS       = 3
FONT_SIZE   = 18
X_TICK_STEP = 10_000


def _plot_histories(
    history_data: dict,
    output_dir: str,
    problem_name: str,
    title_prefix: str,
    ylabel: str,
    filename_suffix: str,
    legend_loc: str,
):
    """All datasets in one PDF — 2×3 grid with shared x-axis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Canonical dataset order; append any extras not in DATASET_ORDER
    ordered = [d for d in DATASET_ORDER if d in history_data]
    ordered += [d for d in history_data if d not in ordered]

    nrows = (len(ordered) + NCOLS - 1) // NCOLS
    fig, axes = plt.subplots(
        nrows, NCOLS,
        sharex=True,
        figsize=(6 * NCOLS, 4 * nrows),
        squeeze=False,
    )

    for idx, dataset_name in enumerate(ordered):
        col, row = divmod(idx, nrows)
        ax = axes[row][col]

        for strategy, data in sorted(history_data[dataset_name].items()):
            color     = STRATEGY_COLORS.get(strategy, "black")
            linestyle = STRATEGY_LINESTYLES.get(strategy, "-")
            label     = _format_label(strategy, data["best_params"], data["n_graphs"])

            ax.plot(data["fitness_evals"], data["mean_history"],
                    color=color, linestyle=linestyle, linewidth=1.8, label=label)
            ax.fill_between(data["fitness_evals"],
                            data["mean_history"] - data["std_history"],
                            data["mean_history"] + data["std_history"],
                            color=color, alpha=0.15)

        ax.set_title(DATASET_LABELS_LONG.get(dataset_name, dataset_name), fontsize=FONT_SIZE)
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(X_TICK_STEP))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))

    # Shared axis labels: y on left column only, x on bottom row only
    for row in range(nrows):
        axes[row][0].set_ylabel(ylabel, fontsize=FONT_SIZE)
    for col in range(NCOLS):
        axes[nrows - 1][col].set_xlabel("Number of Fitness Evaluations", fontsize=FONT_SIZE)

    # Hide unused axes (when n_datasets is not a multiple of NCOLS)
    for idx in range(len(ordered), nrows * NCOLS):
        col, row = divmod(idx, nrows)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    save_path = output_dir / f"{problem_name}_{filename_suffix}.pdf"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Public API — convergence
# ---------------------------------------------------------------------------

def build_convergence_data(
    path_results: str,
    problem: str,
    supervised: bool,
) -> dict:
    """
    Find the best param combi per (dataset_name, evolution_mode) then compute
    mean ± std of best_score_history using graph_intra_std aggregation.
    """
    filtered, best_df, param_cols = _load_filtered(
        path_results, problem, supervised, extra_cols=["best_score_history"]
    )
    return _aggregate_histories(
        filtered, best_df, param_cols,
        history_col="best_score_history",
        trim_infeasible=True,
    )


def plot_convergence_by_strategy(
    convergence_data: dict,
    output_dir: str,
    problem_name: str,
):
    """One PDF per dataset_name with all evolution strategies overlaid as mean ± std."""
    _plot_histories(
        convergence_data, output_dir, problem_name,
        title_prefix="Mean Convergence",
        ylabel="Fitness Score",
        filename_suffix="convergence",
        legend_loc="lower right",
    )


def run_convergence_pipeline(
    path_results: str,
    problem: str,
    output_dir: str,
    supervised: bool,
):
    """End-to-end: find best params → collect fitness histories → plot."""
    logger.info("Building convergence data...")
    convergence_data = build_convergence_data(
        path_results=path_results,
        problem=problem,
        supervised=supervised,
    )
    logger.info(f"Plotting {sum(len(v) for v in convergence_data.values())} curves...")
    plot_convergence_by_strategy(
        convergence_data=convergence_data,
        output_dir=output_dir,
        problem_name=problem,
    )
    logger.info("Done.")


# ---------------------------------------------------------------------------
# Public API — diversity
# ---------------------------------------------------------------------------

def build_diversity_data(
    path_results: str,
    problem: str,
    supervised: bool,
) -> dict:
    """
    Find the best param combi per (dataset_name, evolution_mode) then compute
    mean ± std of diversity_history (mean Hamming distance over the population).

    Same graph_intra_std aggregation as build_convergence_data.
    """
    filtered, best_df, param_cols = _load_filtered(
        path_results, problem, supervised, extra_cols=["diversity_history"]
    )
    return _aggregate_histories(
        filtered, best_df, param_cols,
        history_col="diversity_history",
        trim_infeasible=False,
    )


def plot_diversity_by_strategy(
    diversity_data: dict,
    output_dir: str,
    problem_name: str,
):
    """One PDF per dataset_name comparing population diversity across evolution types."""
    _plot_histories(
        diversity_data, output_dir, problem_name,
        title_prefix="Population Diversity",
        ylabel="Mean Hamming Distance",
        filename_suffix="diversity",
        legend_loc="upper right",
    )


def run_diversity_pipeline(
    path_results: str,
    problem: str,
    output_dir: str,
    supervised: bool,
):
    """End-to-end: find best params → collect diversity histories → plot."""
    logger.info("Building diversity data...")
    diversity_data = build_diversity_data(
        path_results=path_results,
        problem=problem,
        supervised=supervised,
    )
    logger.info(f"Plotting {sum(len(v) for v in diversity_data.values())} curves...")
    plot_diversity_by_strategy(
        diversity_data=diversity_data,
        output_dir=output_dir,
        problem_name=problem,
    )
    logger.info("Done.")
