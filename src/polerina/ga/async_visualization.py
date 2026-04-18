"""
Post-hoc visualization from Parquet result files.

Usage (standalone):
    uv run python -m polerina.ga.async_visualization \
        --parquet-path results/parquet/dataset_name=rb200/graph_id=0/results.parquet \
        --output-dir plots/postprocess

Or import and call directly:
    from polerina.ga.async_visualization import plot_from_parquet
    plot_from_parquet(parquet_path="...", output_dir="...")
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.rcParams["pdf.fonttype"] = 42  # Embed TrueType fonts (required for most publishers)
matplotlib.rcParams["ps.fonttype"] = 42

logger = logging.getLogger(__name__)


def _reconstruct_fitness_evals(best_score_history: list, pop_size: int, nb_offsprings: int) -> np.ndarray:
    n = len(best_score_history)
    return np.array([pop_size + i * nb_offsprings for i in range(n)])


def _plot_single(
    ax1, ax2,
    fitness_evals: np.ndarray,
    best_scores: list,
    diversity: list,
    reference_value: Optional[float],
):
    ax1.plot(fitness_evals, best_scores, color="green", label="Best Score")
    ax2.plot(fitness_evals, diversity, color="orange", alpha=0.6, label="Mean Hamming")

    if reference_value is not None:
        ax1.axhline(y=reference_value, color="red", linestyle=":", label=f"Optimum ({int(reference_value)})")


def _plot_aggregated(
    ax1, ax2,
    fitness_evals: np.ndarray,
    all_best_scores: np.ndarray,   # shape (n_reps, n_iters)
    all_diversity: np.ndarray,     # shape (n_reps, n_iters)
    reference_value: Optional[float],
):
    mean_best = all_best_scores.mean(axis=0)
    std_best = all_best_scores.std(axis=0)
    mean_div = all_diversity.mean(axis=0)
    std_div = all_diversity.std(axis=0)

    ax1.plot(fitness_evals, mean_best, color="green", label="Best Score (mean)")
    ax1.fill_between(fitness_evals, mean_best - std_best, mean_best + std_best,
                     color="green", alpha=0.15, label="±1 std")

    ax2.plot(fitness_evals, mean_div, color="orange", alpha=0.6, label="Mean Hamming (mean)")
    ax2.fill_between(fitness_evals, mean_div - std_div, mean_div + std_div,
                     color="orange", alpha=0.1)

    if reference_value is not None:
        ax1.axhline(y=reference_value, color="red", linestyle=":", label=f"Optimum ({int(reference_value)})")


def _make_figure(title: str):
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    ax1.set_title(title)
    ax1.set_xlabel("Number of fitness evaluations")
    ax1.set_ylabel("Fitness (Green)")
    ax2.set_ylabel("Mean Hamming Distance (Orange)")
    ax1.grid(True)

    return fig, ax1, ax2


def _finalize_and_save(fig, ax1, ax2, save_path: Optional[Path]):
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Set ylim from line data only — fill_between patches can have extreme
    # y-extents (mean ± std) that collapse the mean line to top or bottom.
    for ax in (ax1, ax2):
        y_vals = np.concatenate([
            line.get_ydata() for line in ax.lines
            if len(line.get_ydata()) > 0
        ])
        if len(y_vals):
            ax.set_ylim(bottom=max(0, np.nanmin(y_vals) * 0.95), top=np.nanmax(y_vals) * 1.05)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved plot: {save_path}")

    plt.close(fig)


def plot_from_parquet(
    parquet_path: str,
    output_dir: Optional[str] = None,
    aggregate: bool = True,
    problem_name: Optional[str] = None,
):
    """
    Generate convergence plots from a single Parquet result file.

    Args:
        parquet_path: Path to a results.parquet file.
        output_dir:   Directory where plot PDFs will be saved. Defaults to
                      a ``plots/`` folder next to the ``parquet/`` directory
                      (i.e. the result timestamp folder).
        aggregate:    If True, plot mean ± std across repetitions per param set.
                      If False, plot every individual run.
        problem_name: Override the problem name shown in the title.
    """
    parquet_path = Path(parquet_path)
    # parquet_path = .../results_<ts>/parquet/dataset_name=X/graph_id=Y/results.parquet
    # parents[3]   = .../results_<ts>/
    output_dir = Path(output_dir) if output_dir else parquet_path.parents[3] / "plots"

    df = pl.read_parquet(parquet_path)

    if problem_name is None:
        problem_name = df["problem"][0] if "problem" in df.columns else "unknown"

    graph_id = df["graph_id"][0]
    dataset_name = df["dataset_name"][0] if "dataset_name" in df.columns else "unknown"

    param_cols = [c for c in df.columns if c.startswith("param_")]

    for keys, group in df.group_by(param_cols):
        param_dict = dict(zip(param_cols, keys if isinstance(keys, tuple) else (keys,)))
        param_str = "_".join(f"{k.removeprefix('param_')}={v}" for k, v in param_dict.items())

        pop_size = int(group["param_pop_size"][0])
        nb_offsprings = int(group["param_nb_offsprings"][0])

        reference_value = None
        if "true_size_set" in group.columns and group["true_size_set"][0] is not None:
            reference_value = float(group["true_size_set"][0])

        histories_best = [row.to_list() for row in group["best_score_history"]]
        histories_div = [row.to_list() for row in group["diversity_history"]]

        fitness_evals = _reconstruct_fitness_evals(histories_best[0], pop_size, nb_offsprings)

        title = (
            f"GA Convergence — {problem_name.upper()} | "
            f"Dataset: {dataset_name} | Graph: {graph_id}\n"
            f"{param_str}"
        )

        if aggregate:
            min_len = min(len(h) for h in histories_best)
            arr_best = np.array([h[:min_len] for h in histories_best])
            arr_div = np.array([h[:min_len] for h in histories_div])
            evals = fitness_evals[:min_len]

            fig, ax1, ax2 = _make_figure(title)
            _plot_aggregated(ax1, ax2, evals, arr_best, arr_div, reference_value)
            save_path = output_dir / f"graph_{graph_id}_{param_str}_aggregated.pdf"
            _finalize_and_save(fig, ax1, ax2, save_path)

        else:
            for rep_idx, (h_best, h_div) in enumerate(zip(histories_best, histories_div)):
                evals = _reconstruct_fitness_evals(h_best, pop_size, nb_offsprings)
                fig, ax1, ax2 = _make_figure(title + f" | rep {rep_idx}")
                _plot_single(ax1, ax2, evals, h_best, h_div, reference_value)
                save_path = output_dir / f"graph_{graph_id}_{param_str}_rep{rep_idx}.pdf"
                _finalize_and_save(fig, ax1, ax2, save_path)


def plot_dataset_from_parquet(
    parquet_dir: str,
    output_dir: Optional[str] = None,
    aggregate: bool = True,
    problem_name: Optional[str] = None,
):
    """
    Walk a hive-partitioned results directory and generate plots for every parquet file found.

    Args:
        parquet_dir: Root of the hive-partitioned results (contains dataset_name= subdirs).
        output_dir:  Directory where plots will be saved. Defaults to a ``plots/`` folder
                     next to the ``parquet/`` directory (i.e. the result timestamp folder).
        aggregate:   Passed through to plot_from_parquet.
        problem_name: Override the problem name shown in titles.
    """
    parquet_dir = Path(parquet_dir)
    # parquet_dir = .../results_<ts>/parquet  →  output = .../results_<ts>/plots
    resolved_output_dir = str(output_dir) if output_dir else None
    files = sorted(parquet_dir.rglob("results.parquet"))

    if not files:
        logger.warning(f"No results.parquet files found under {parquet_dir}")
        return

    logger.info(f"Found {len(files)} parquet file(s) to process.")

    for f in files:
        logger.info(f"Processing: {f}")
        plot_from_parquet(
            parquet_path=str(f),
            output_dir=resolved_output_dir,
            aggregate=aggregate,
            problem_name=problem_name,
        )


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        parquet_path: Optional[str] = typer.Option(None, help="Path to a single results.parquet file."),
        parquet_dir: Optional[str] = typer.Option(None, help="Root of hive-partitioned results dir (processes all files)."),
        output_dir: Optional[str] = typer.Option(None, help="Output directory for plots. Defaults to plots/ next to parquet/."),
        aggregate: bool = typer.Option(True, help="Aggregate over repetitions (mean ± std)."),
        problem_name: Optional[str] = typer.Option(None, help="Override problem name in plot titles."),
    ):
        logging.basicConfig(level=logging.INFO)
        if parquet_path:
            plot_from_parquet(parquet_path, output_dir, aggregate, problem_name)
        elif parquet_dir:
            plot_dataset_from_parquet(parquet_dir, output_dir, aggregate, problem_name)
        else:
            raise typer.BadParameter("Provide either --parquet-path or --parquet-dir.")

    app()
