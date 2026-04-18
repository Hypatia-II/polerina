"""Runner for re-running GA experiments on best param combis from a previous HP tuning run."""

import ast
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from polerina.analysis.aggregate_results_unified import (
    get_best_results_graph_agg,
    load_and_preprocess_results,
)
from polerina.runners.hp_tuning import run_hyperparameter_tuning

logger = logging.getLogger(__name__)

_PARAM_COLS = [
    "param_evolution_mode",
    "param_pop_size",
    "param_nb_offsprings",
    "param_init_type",
    "param_crossover_rate",
    "param_mutation_type",
]
_OPT_PARAM_COLS = ["param_lamarckian_probability"]


def _row_to_params_ga(row: dict, has_lb: bool) -> dict:
    """Convert a best-result row into a single-element params_ga dict."""
    params = {col.replace("param_", "", 1): [row[col]] for col in _PARAM_COLS}
    if has_lb and row.get("param_lamarckian_probability") is not None:
        params["lamarckian_probability"] = [row["param_lamarckian_probability"]]
    return params


def _log_params_ga(params_ga: dict):
    logger.info("--- GA Parameters ---")
    for k, v in params_ga.items():
        logger.info(f"  {k}: {v[0]}")


def _write_selected_combis_graph_agg(best_df, allowed_datasets: set, group_cols: list, run_root: str):
    """Write the chosen param combi for every (dataset, evolution_mode) pair to a summary file."""
    from pathlib import Path
    param_keys = [c.replace("param_", "", 1) for c in group_cols]
    lines = ["Selected param combis per dataset (graph_agg)\n", "=" * 60 + "\n"]
    _DATASET_ORDER = ["co_er_small", "co_er_large", "co_ba_small", "co_ba_large", "co_rb_small", "co_rb_large"]
    _MODE_ORDER = ["darwin", "baldwin", "lamarck", "lb"]
    for row in sorted(
        best_df.iter_rows(named=True),
        key=lambda r: (
            _DATASET_ORDER.index(r["dataset_name"]) if r["dataset_name"] in _DATASET_ORDER else len(_DATASET_ORDER),
            _MODE_ORDER.index(r["param_evolution_mode"]) if r["param_evolution_mode"] in _MODE_ORDER else len(_MODE_ORDER),
        ),
    ):
        if row["dataset_name"] not in allowed_datasets:
            continue
        combi = {k: row[c] for k, c in zip(param_keys, group_cols) if row.get(c) is not None}
        lines.append(f"  {row['dataset_name']:20s}  {combi}\n")
    path = Path(run_root) / "selected_combis.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    logger.info(f"Selected combis written to {path}")



def _parse_selected_combis(path: Path) -> list[tuple[str, dict]]:
    """Parse selected_combis.txt into (key, combi_dict) tuples.

    Works for both graph_agg (key=dataset_name) and normalized_agg (key=evolution_mode).
    Lines without a ``{`` (headers, blank lines) are skipped.
    """
    entries = []
    for line in path.read_text().splitlines():
        if "{" not in line:
            continue
        idx = line.index("{")
        key = line[:idx].strip()
        combi = ast.literal_eval(line[idx:].strip())
        entries.append((key, combi))
    return entries


def _combi_to_params_ga(combi: dict) -> dict:
    """Wrap each combi value in a list to match params_ga format expected by run_hyperparameter_tuning."""
    return {k: [v] for k, v in combi.items()}


def _run_group(params_ga, params_data, params_outputs, problem_name, nb_reps_per_graph, timestamp, track_metrics):
    _log_params_ga(params_ga)
    logger.info(f"  datasets: {params_data['dataset_name']}")
    run_hyperparameter_tuning(
        params_outputs=dict(params_outputs),
        params_ga=params_ga,
        problem_name=problem_name,
        synthetic_data=False,
        params_data=params_data,
        nb_reps_per_graph=nb_reps_per_graph,
        timestamp=timestamp,
        track_metrics=track_metrics,
    )


def run_best_params(
    results_path: str,
    problem_name: str,
    aggregation_method: Literal["graph_agg"],
    nb_reps_per_graph: int,
    params_data: dict,
    params_outputs: dict,
    supervised: bool,
    resume_path: str | None = None,
    combis_path: str | None = None,
    track_metrics: bool = True,
):
    """
    Re-run GA experiments on the best param combis identified by graph_agg analysis:
    best combi per (dataset_name, evolution_mode). Datasets sharing the same best
    combi (including mode) are batched into one run. Only datasets listed in
    params_data are included.

    For cross-problem best-param selection (MIS + MaxCut combined), use
    get_best_results_normalized_agg directly from aggregate_results_unified.
    """
    # All groups for this run are nested under a single root folder so results
    # are self-contained and don't scatter across the existing results tree.
    if resume_path is not None:
        run_root = resume_path
        # Recover the original timestamp from the folder name (e.g. "graph_agg_20260408_1943"
        # → "20260408_1943") so that setup_path_results rebuilds the same subdirectory paths
        # and the per-graph skip logic in big_worker can find existing results.
        folder_name = Path(resume_path).name
        base_timestamp = "_".join(folder_name.split("_")[-2:])
        logger.info(f"Resuming from existing run directory: {run_root} (timestamp: {base_timestamp})")
    else:
        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_root = f"{params_outputs['output_path']}/best_params/{problem_name}/{aggregation_method}_{base_timestamp}"
    run_params_outputs = {**params_outputs, "output_path": run_root}

    logger.info("=== Starting Best Params Experiment ===")
    logger.info(f"Problem:              {problem_name}")
    logger.info(f"Aggregation method:   {aggregation_method}")
    logger.info(f"Source results path:  {results_path}")
    logger.info(f"Repetitions / graph:  {nb_reps_per_graph}")
    logger.info(f"Supervised:           {supervised}")
    logger.info(f"Run root:             {run_root}")
    logger.info("--- Data Parameters ---")
    for k, v in params_data.items():
        logger.info(f"  {k}: {v}")
    logger.info("--- Output Parameters ---")
    for k, v in run_params_outputs.items():
        logger.info(f"  {k}: {v}")

    time_start = time.time()

    # Determine source of param combis:
    # 1. combis_path explicitly provided → read from that file (fresh output dir)
    # 2. resume_path provided → read from run_root/selected_combis.txt (same output dir)
    # 3. neither → derive from results
    if combis_path is not None:
        _combis_file = Path(combis_path)
        resuming_from_file = _combis_file.exists()
        if not resuming_from_file:
            raise FileNotFoundError(f"combis_path not found: {_combis_file}")
        logger.info(f"Loading selected combis from {_combis_file} (skipping result preprocessing)")
        entries = _parse_selected_combis(_combis_file)
    else:
        _combis_file = Path(run_root) / "selected_combis.txt"
        resuming_from_file = resume_path is not None and _combis_file.exists()
        if resuming_from_file:
            logger.info(f"Resume: reusing selected combis from {_combis_file} (skipping result preprocessing)")
            entries = _parse_selected_combis(_combis_file)
        else:
            results = load_and_preprocess_results(results_path, problem_name)
            has_lb = "param_lamarckian_probability" in results.schema
            group_cols = _PARAM_COLS + (_OPT_PARAM_COLS if has_lb else [])

    if aggregation_method == "graph_agg":
        if resuming_from_file:
            allowed_datasets = set(params_data["dataset_name"])
            entries = [(ds, combi) for ds, combi in entries if ds in allowed_datasets]
            logger.info(f"graph_agg: {len(entries)} (dataset, mode) pairs to run (from file)")
            for i, (dataset_name, combi) in enumerate(entries):
                logger.info(f"\n--- Run {i + 1}/{len(entries)}: {dataset_name} / {combi['evolution_mode']} ---")
                _run_group(
                    _combi_to_params_ga(combi),
                    {**params_data, "dataset_name": [dataset_name]},
                    run_params_outputs,
                    problem_name,
                    nb_reps_per_graph,
                    timestamp=f"{base_timestamp}_{combi['evolution_mode']}",
                    track_metrics=track_metrics,
                )
        else:
            best_df = get_best_results_graph_agg(
                results, problem_name, supervised
            ).collect()

            allowed_datasets = set(params_data["dataset_name"])
            found_datasets = set(best_df["dataset_name"].to_list())
            missing = allowed_datasets - found_datasets
            if missing:
                logger.warning(
                    f"The following datasets from config were not found in analysis results "
                    f"and will be skipped: {sorted(missing)}"
                )

            rows = [
                row for row in best_df.iter_rows(named=True)
                if row["dataset_name"] in allowed_datasets
            ]

            logger.info(f"graph_agg: {len(rows)} (dataset, mode) pairs to run")
            _write_selected_combis_graph_agg(best_df, allowed_datasets, group_cols, run_root)

            for i, row in enumerate(rows):
                logger.info(f"\n--- Run {i + 1}/{len(rows)}: {row['dataset_name']} / {row['param_evolution_mode']} ---")
                params_ga = _row_to_params_ga(row, has_lb)
                _run_group(
                    params_ga,
                    {**params_data, "dataset_name": [row["dataset_name"]]},
                    run_params_outputs,
                    problem_name,
                    nb_reps_per_graph,
                    timestamp=f"{base_timestamp}_{row['param_evolution_mode']}",
                    track_metrics=track_metrics,
                )

    else:
        raise ValueError(
            f"Unknown aggregation_method: {aggregation_method!r}. "
            "Use 'graph_agg'."
        )

    compute_time = time.time() - time_start
    logger.info(f"=== Best Params Experiment complete. Total compute time: {compute_time:.2f}s ===")
