"""POLERINA CLI — single entry point for run, download, and analyze commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import msgspec
import typer

app = typer.Typer(
    name="polerina",
    help="Genetic Algorithm for Optimization Problems (MIS, Max-Cut).",
    add_completion=False,
)

DEFAULT_RUN_HP_TUNING_CONFIG = "config/run_hp_tuning.toml"
DEFAULT_RUN_SOLVER_CONFIG = "config/run_solver.toml"
DEFAULT_DOWNLOAD_CONFIG = "config/download.toml"
DEFAULT_ANALYZE_CONFIG = "config/analyze.toml"


@app.command()
def run_hyperparameter_tuning(
    config: Path = typer.Option(
        DEFAULT_RUN_HP_TUNING_CONFIG, "--config", "-c", help="Path to run config TOML file."
    ),
    dataset: Optional[list[str]] = typer.Option(
        None, "--dataset", "-d", help="Override data.dataset_name (repeatable)."
    ),
    reps: Optional[int] = typer.Option(
        None, "--reps", "-r", help="Override nb_reps_per_graph."
    ),
    sample: Optional[float] = typer.Option(
        None, "--sample", "-s", help="Override data.sample fraction."
    ),
    problem: Optional[str] = typer.Option(
        None, "--problem", "-p", help="Optimization problem type (mis, maxcut)."
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", help="Resume experiment from a specific timestamp (e.g., 20260320_0007)."
    ),
) -> None:
    """Run the GA hyperparameter grid search."""
    from polerina.config import RunHPTuningConfig, load_toml, run_config_to_dicts
    from polerina.runners.hp_tuning import run_hyperparameter_tuning

    cfg = load_toml(config, RunHPTuningConfig)

    if cfg.synthetic_data:
        if cfg.graph is None:
            typer.secho("Error: 'synthetic_data' is true, but [graph] section is missing in TOML.", fg="red")
            raise typer.Exit(1)
    else:
        if cfg.data is None:
            typer.secho("Error: 'synthetic_data' is false, but [data] section is missing in TOML.", fg="red")
            raise typer.Exit(1)

    if dataset:
        cfg.data.dataset_name = dataset
    if reps is not None:
        cfg.nb_reps_per_graph = reps
    if sample is not None:
        cfg.data.sample = sample
    if problem:
        cfg.problem_name = problem

    kwargs = run_config_to_dicts(cfg)
    kwargs["timestamp"] = resume
    run_hyperparameter_tuning(**kwargs)


@app.command()
def run_solver(
    config: Path = typer.Option(
        DEFAULT_RUN_SOLVER_CONFIG, "--config", "-c", help="Path to run config TOML file."
    ),
    dataset: Optional[list[str]] = typer.Option(
        None, "--dataset", "-d", help="Override data.dataset_name (repeatable)."
    ),
    reps: Optional[int] = typer.Option(
        None, "--reps", "-r", help="Override nb_reps_per_graph."
    ),
    sample: Optional[float] = typer.Option(
        None, "--sample", "-s", help="Override data.sample fraction."
    ),
    problem: Optional[str] = typer.Option(
        None, "--problem", "-p", help="Optimization problem type (mis, maxcut)."
    ),
) -> None:
    """Run the GA solver on a dataset."""
    from polerina.config import RunSolverConfig, load_toml, run_config_to_dicts
    from polerina.runners.solver import run_solver

    cfg = load_toml(config, RunSolverConfig)

    if cfg.synthetic_data:
        if cfg.graph is None:
            typer.secho("Error: 'synthetic_data' is true, but [graph] section is missing in TOML.", fg="red")
            raise typer.Exit(1)
    else:
        if cfg.data is None:
            typer.secho("Error: 'synthetic_data' is false, but [data] section is missing in TOML.", fg="red")
            raise typer.Exit(1)

    if dataset:
        cfg.data.dataset_name = dataset
    if reps is not None:
        cfg.nb_reps_per_graph = reps
    if sample is not None:
        cfg.data.sample = sample
    if problem:
        cfg.problem_name = problem

    kwargs = run_config_to_dicts(cfg)
    run_solver(**kwargs)


@app.command()
def run_best_params(
    config: Path = typer.Option(..., "--config", "-c", help="Path to run-best-params config TOML file."),
    resume_path: Optional[str] = typer.Option(None, "--resume-path", help="Resume an interrupted run by reusing this existing run directory."),
) -> None:
    """Re-run GA on the best param combis found by a previous HP tuning experiment."""
    import logging
    from polerina.config import RunBestParamsConfig, best_params_config_to_dicts, load_toml
    from polerina.runners.best_params import run_best_params as _run_best_params

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_toml(config, RunBestParamsConfig)
    kwargs = best_params_config_to_dicts(cfg)
    if resume_path is not None:
        kwargs["resume_path"] = resume_path
    _run_best_params(**kwargs)


@app.command()
def download(
    config: Path = typer.Option(
        DEFAULT_DOWNLOAD_CONFIG, "--config", "-c", help="Path to download config TOML file."
    ),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Override dataset_name."
    ),
    keep_raw: bool = typer.Option(
        False, "--keep-raw", help="Keep raw benchmark data after conversion."
    ),
) -> None:
    """Download and convert benchmark graph data."""
    from polerina.config import DownloadConfig, download_config_to_args, load_toml
    from polerina.data_handler.data_prep import run_benchmark_conversion_pipeline

    cfg = load_toml(config, DownloadConfig)

    if dataset:
        cfg.dataset_name = dataset
    if keep_raw:
        cfg.delete_raw = False

    kwargs = download_config_to_args(cfg)
    run_benchmark_conversion_pipeline(**kwargs)


@app.command()
def analyze(
    config: Path = typer.Option(
        DEFAULT_ANALYZE_CONFIG, "--config", "-c", help="Path to analyze config TOML file."
    ),
    path_result: Optional[str] = typer.Option(
        None, "--path-result", "-p", help="Override path_result."
    ),
    problem: Optional[str] = typer.Option(
        None, "--problem", help="Optimization problem type (mis, maxcut)."
    ),
) -> None:
    """Analyze experiment results using the unified aggregator."""
    import polars as pl
    from polerina.analysis.aggregate_results_unified import run_analysis_unified
    from polerina.config import AnalyzeConfig, load_toml

    cfg = load_toml(config, AnalyzeConfig)

    if path_result is not None:
        cfg.path_result = path_result
    if problem is not None:
        cfg.problem = problem

    config_dict = {
        "path_result": cfg.path_result,
        "problem": cfg.problem,
        "sig_fig": cfg.sig_fig,
        "supervised": cfg.supervised,
    }

    typer.echo(f"Running unified analysis for problem={cfg.problem}...")
    outputs = run_analysis_unified(config_dict)
    
    # Print a summary
    agg_per_dataset = outputs["agg_per_dataset"].collect()
    typer.echo("\n--- Aggregated Results per Dataset ---")
    with pl.Config(tbl_cols=-1):
        typer.echo(agg_per_dataset)


@app.command()
def plot_convergence(
    path_results: str = typer.Option(..., "--path-results", help="Root of hive-partitioned results dir."),
    problem: str = typer.Option(..., "--problem", "-p", help="Problem: 'mis' or 'maxcut'."),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for PDFs."),
    supervised: bool = typer.Option(..., "--supervised/--no-supervised", help="Use MAE ranking for MIS (vs score)."),
) -> None:
    """Plot mean convergence curves per evolution strategy from parquet results."""
    from polerina.analysis.history_plots import run_convergence_pipeline
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_convergence_pipeline(
        path_results=path_results,
        problem=problem,
        output_dir=output_dir,
        supervised=supervised,
    )


@app.command()
def plot_diversity(
    path_results: str = typer.Option(..., "--path-results", help="Root of hive-partitioned results dir."),
    problem: str = typer.Option(..., "--problem", "-p", help="Problem: 'mis' or 'maxcut'."),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for PDFs."),
    supervised: bool = typer.Option(..., "--supervised/--no-supervised", help="Use MAE ranking for MIS (vs score)."),
) -> None:
    """Plot mean population diversity (Hamming distance) per evolution strategy from parquet results."""
    from polerina.analysis.history_plots import run_diversity_pipeline
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_diversity_pipeline(
        path_results=path_results,
        problem=problem,
        output_dir=output_dir,
        supervised=supervised,
    )


@app.command()
def generate_latex_tables(
    mis_path: Path = typer.Option(..., "--mis-path", help="Root of hive-partitioned MIS results dir."),
    mc_path: Path = typer.Option(..., "--mc-path", help="Root of hive-partitioned MaxCut results dir."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write LaTeX to this file instead of stdout."),
    supervised: bool = typer.Option(True, "--supervised/--no-supervised", help="Use MAE ranking for MIS (vs score)."),
) -> None:
    """Generate LaTeX performance and hyperparameter tables using graph_intra_std aggregation."""
    import logging
    import polars as pl
    from polerina.analysis.aggregate_results_unified import (
        validate_data, load_and_preprocess_results, get_best_results_graph_agg,
        get_best_results_normalized_agg, compute_cross_problem_selection_loss,
    )
    from polerina.analysis.latex_tables import (
        make_performance_table, make_combined_params_table, make_unique_solutions_table,
        make_normalized_agg_loss_table,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    def _load(path: Path, problem: str) -> tuple[pl.LazyFrame, pl.DataFrame]:
        validate_data(str(path), problem)
        results = load_and_preprocess_results(str(path), problem)
        df = get_best_results_graph_agg(results, problem, supervised).collect()
        return results, df

    results_mis, mis = _load(mis_path, "mis")
    results_mc,  mc  = _load(mc_path,  "maxcut")

    perf_table            = make_performance_table(mis, mc)
    combined_params_table = make_combined_params_table(mis, mc)
    unique_table          = make_unique_solutions_table(mis, mc)
    best_combined         = get_best_results_normalized_agg(results_mis, results_mc)
    loss_df               = compute_cross_problem_selection_loss(results_mis, results_mc)
    loss_table            = make_normalized_agg_loss_table(loss_df, best_combined)
    latex = perf_table + "\n\n" + combined_params_table + "\n\n" + unique_table + "\n\n" + loss_table

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(latex)
        typer.echo(f"Saved to {output}")
    else:
        typer.echo(latex)


@app.command()
def generate_runtime_table(
    mis_timing_path: Path = typer.Option(..., "--mis-path", help="Root of MIS timing results dir."),
    mc_timing_path: Path = typer.Option(..., "--mc-path", help="Root of MaxCut timing results dir."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write LaTeX to this file instead of stdout."),
) -> None:
    """Generate a LaTeX runtime table from timing experiment results (track_metrics=False runs)."""
    import logging
    from polerina.analysis.aggregate_results_unified import aggregate_runtime_graph_agg
    from polerina.analysis.latex_tables import make_runtime_table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    mis = aggregate_runtime_graph_agg(str(mis_timing_path))
    mc  = aggregate_runtime_graph_agg(str(mc_timing_path))
    latex = make_runtime_table(mis, mc)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(latex)
        typer.echo(f"Saved to {output}")
    else:
        typer.echo(latex)


if __name__ == "__main__":
    app()
