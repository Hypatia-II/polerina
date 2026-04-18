import polars as pl
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PARAM_COLS = [
    "param_evolution_mode",
    "param_pop_size",
    "param_nb_offsprings",
    "param_init_type",
    "param_crossover_rate",
    "param_mutation_type"
]

def get_param_cols(has_lb_col: bool = False):
    cols = DEFAULT_PARAM_COLS.copy()
    if has_lb_col:
        cols.append("param_lamarckian_probability")
    return cols

def validate_data(path_results: str, problem: str):
    """
    Validates that the data at path_results matches the expected problem configuration.
    """
    parquet_file = next(
        (os.path.join(root, f) for root, _, files in os.walk(path_results) for f in files if f.endswith(".parquet")),
        None
    )
    if parquet_file is None:
        raise FileNotFoundError(f"No parquet files found in {path_results}")

    schema = pl.read_parquet(parquet_file, n_rows=1).schema
    cols = schema.keys()

    required_ga_cols = [
        "param_evolution_mode", "param_pop_size", "param_nb_offsprings",
        "param_init_type", "param_crossover_rate",
        "param_mutation_type", "max_score_found", "improvements"
    ]
    missing_ga = [c for c in required_ga_cols if c not in cols]
    if missing_ga:
        raise ValueError(f"Missing required GA columns in data: {missing_ga}")

    if problem == "mis":
        if "true_size_set" not in cols:
            logger.warning("'true_size_set' not found for MIS. MAE calculation will be skipped or limited.")
    elif problem != "maxcut":
        raise ValueError(f"Unknown problem type: {problem}")

    logger.info(f"Validation successful for problem={problem}")
    return True

def load_and_preprocess_results(path_results: str, problem: str) -> pl.LazyFrame:
    """
    Preprocess results: calculate errors, speed of convergence.
    """
    # Build a union schema from one non-lb file and one lb file so that
    # param_lamarckian_probability is always present in the schema regardless of
    # scan order. This handles best_params runs where lb and non-lb modes are stored
    # in separate parquet files with different schemas. missing_columns="insert" fills
    # the column with null for non-lb files; extra_columns="ignore" is a safety net.
    _first = next(
        (os.path.join(r, f) for r, _, files in os.walk(path_results)
         for f in files if f.endswith(".parquet")),
        None,
    )
    _lb = next(
        (os.path.join(r, f) for r, _, files in os.walk(path_results)
         for f in files if f.endswith(".parquet") and "_lb" in r),
        None,
    )
    union_schema: dict = {}
    for _fp in filter(None, [_first, _lb]):
        for col, dtype in pl.read_parquet(_fp, n_rows=0).schema.items():
            if col not in union_schema:
                union_schema[col] = dtype

    results = pl.scan_parquet(
        f"{path_results}/**/*.parquet",
        hive_partitioning=True,
        schema=union_schema or None,
        missing_columns="insert",
        extra_columns="ignore",
    )

    # 1. Error calculations (for MIS mainly)
    if problem == "mis":
        schema = results.schema
        if "true_size_set" in schema:
            results = results.with_columns([
                (pl.col("true_size_set") - pl.col("max_score_found")).alias("error"),
                (pl.col("true_size_set") - pl.col("max_score_found")).abs().alias("abs_error")
            ])
        else:
            results = results.with_columns([
                pl.lit(None).alias("error"),
                pl.lit(None).alias("abs_error")
            ])

    # 2. Speed of convergence
    results = results.with_columns(
        pl.when(
            pl.col("improvements").list.get(0, null_on_oob=True).struct.field("score") == pl.col("max_score_found")
        )
        .then(
            pl.col("improvements").list.get(0, null_on_oob=True).struct.field("index") * pl.col("param_nb_offsprings") + pl.col("param_pop_size")
        )
        .otherwise(None)
        .alias("convergence_speed"),
    )

    return results


def _agg_per_param_dataset(
    results: pl.LazyFrame,
    problem: str,
    supervised: bool,
    param_cols: list[str],
) -> pl.LazyFrame:
    """
    Shared steps 1+2: average over repetitions per graph, then across graphs.
    Returns one row per (param_combi, dataset_name).

    Sorting before each group_by is required for determinism: scan_parquet with a
    glob pattern does not guarantee a consistent file scan order across runs, and
    floating-point mean() is order-dependent. Without the sort, the same dataset can
    produce slightly different aggregated scores on each run, which can flip which
    param combo is selected as best.
    """
    per_graph_agg = [
        pl.col("max_score_found").mean().alias("g_id_mean_max_score"),
        pl.col("max_score_found").std().alias("g_id_std_max_score"),
        pl.col("convergence_speed").mean().alias("g_id_mean_convergence_speed"),
        pl.col("convergence_speed").std().alias("g_id_std_convergence_speed"),
        pl.col("nb_unique_optimal_solutions").mean().alias("g_id_mean_nb_unique"),
        pl.col("nb_unique_optimal_solutions").std().alias("g_id_std_nb_unique"),
    ]
    if problem == "mis":
        per_graph_agg.append(pl.col("true_size_set").first())

    pl.set_random_seed(42)
    df = results.collect().sort([*param_cols, "dataset_name", "graph_id", "repetition"])

    agg_per_graph = (
        df
        .group_by([*param_cols, "dataset_name", "graph_id"])
        .agg(per_graph_agg)
        .sort([*param_cols, "dataset_name", "graph_id"])
    )

    across_graph_agg = [
        pl.len().alias("count_g_id"),
        pl.col("g_id_mean_max_score").mean().alias("mean_max_score"),
        pl.col("g_id_std_max_score").mean().alias("std_max_score"),
        pl.col("g_id_mean_convergence_speed").mean().alias("mean_convergence_speed"),
        pl.col("g_id_std_convergence_speed").mean().alias("std_convergence_speed"),
        pl.col("g_id_mean_nb_unique").mean().alias("mean_nb_unique"),
        pl.col("g_id_std_nb_unique").mean().alias("std_nb_unique"),
    ]
    if problem == "mis" and supervised:
        across_graph_agg.append(
            (pl.col("true_size_set") - pl.col("g_id_mean_max_score"))
            .abs().mean().alias("MAE")
        )

    return (
        agg_per_graph
        .group_by([*param_cols, "dataset_name"])
        .agg(across_graph_agg)
        .lazy()
    )


def get_best_results_graph_agg(
    results: pl.LazyFrame,
    problem: str,
    supervised: bool = True,
) -> pl.LazyFrame:
    """
    Find the best param combi per (dataset_name, param_evolution_mode) using a
    graph-first two-step aggregation:

      1. Average over repetitions per graph  → one score per (param_combi, graph_id)
      2. Mean ± std across graphs            → dataset-level stats per param_combi
      3. Pick the best per (dataset_name, param_evolution_mode)

    For supervised MIS, "best" = lowest MAE across graphs.
    For MaxCut / unsupervised MIS, "best" = highest mean_max_score.
    """
    has_lb_col = "param_lamarckian_probability" in results.schema
    param_cols = get_param_cols(has_lb_col)

    agg_per_param_dataset = _agg_per_param_dataset(results, problem, supervised, param_cols)

    if problem == "mis" and supervised:
        sort_col, descending_score = "MAE", False
    else:
        sort_col, descending_score = "mean_max_score", True

    # Param columns are used as tiebreakers to guarantee a deterministic winner when
    # two combos produce identical or near-identical scores due to floating-point noise.
    tiebreaker_cols = [c for c in param_cols if c != "param_evolution_mode"]
    return (
        agg_per_param_dataset
        .sort(
            ["dataset_name", "param_evolution_mode", sort_col] + tiebreaker_cols,
            descending=[False, False, descending_score] + [False] * len(tiebreaker_cols),
        )
        .unique(subset=["dataset_name", "param_evolution_mode"], keep="first", maintain_order=True)
    )


def get_best_results_normalized_agg(
    results_mis: pl.LazyFrame,
    results_maxcut: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Find the best param combi per evolution_mode across both problems (MIS unsupervised
    + MaxCut) and all datasets using normalised score averaging:

      1+2. Graph-level aggregation per problem separately (same as get_best_results_graph_agg,
           always unsupervised so metric = mean_max_score for both)
        3. Tag each result with its problem label, then concatenate
           → 12 (problem × dataset) rows per param_combi
        4. Normalise within (problem, dataset_name, param_evolution_mode):
           normalised = mean_max_score / max(mean_max_score in group)
           Scores within each problem+dataset are independent, so their raw scales
           never mix. When all combis are tied, fill_nan gives 1.0.
        5. Average normalised score across all 12 cells per param_combi
        6. Pick the best per evolution_mode (param columns used as tiebreakers)
    """
    has_lb_col = (
        "param_lamarckian_probability" in results_mis.schema
        or "param_lamarckian_probability" in results_maxcut.schema
    )
    param_cols = get_param_cols(has_lb_col)

    # Steps 1+2: aggregate per problem separately (MIS always unsupervised here)
    agg_mis = (
        _agg_per_param_dataset(results_mis, "mis", supervised=False, param_cols=param_cols)
        .with_columns(pl.lit("mis").alias("problem"))
    )
    agg_maxcut = (
        _agg_per_param_dataset(results_maxcut, "maxcut", supervised=False, param_cols=param_cols)
        .with_columns(pl.lit("maxcut").alias("problem"))
    )

    # Step 3: concatenate → 12 (problem, dataset) rows per param_combi
    combined = (
        pl.concat([agg_mis.collect(), agg_maxcut.collect()])
        .sort([*param_cols, "problem", "dataset_name"])
    )

    # Step 4: normalise within (problem, dataset_name, param_evolution_mode)
    window = ["problem", "dataset_name", "param_evolution_mode"]
    raw = pl.col("mean_max_score")
    norm = raw / raw.max().over(window)
    combined = combined.with_columns(norm.fill_nan(1.0).alias("normalised_score"))

    # Step 5: average normalised score across all 12 (problem, dataset) cells per param_combi
    agg_across = combined.group_by(param_cols).agg([
        pl.col("normalised_score").mean().alias("mean_normalised_score"),
        pl.col("normalised_score").std().alias("std_normalised_score"),
        pl.len().alias("count_cells"),
    ])

    # Step 6: pick best per evolution_mode, param columns used as tiebreakers for determinism
    tiebreaker_cols = [c for c in param_cols if c != "param_evolution_mode"]
    return (
        agg_across
        .sort(
            ["param_evolution_mode", "mean_normalised_score"] + tiebreaker_cols,
            descending=[False, True] + [False] * len(tiebreaker_cols),
        )
        .unique(subset=["param_evolution_mode"], keep="first", maintain_order=True)
    )


def compute_cross_problem_selection_loss(
    results_mis: pl.LazyFrame,
    results_maxcut: pl.LazyFrame,
) -> pl.DataFrame:
    """
    For each (evolution_mode, problem, dataset) cell, compare:
      - oracle score: mean_max_score of the best combo for that specific (problem, dataset, evolution_mode)
        — i.e. what get_best_results_graph_agg would pick
      - combined score: mean_max_score of the cross-problem best combo from
        get_best_results_normalized_agg when applied to that same cell

    Returns one row per cell (24 rows for 4 modes × 2 problems × 6 datasets) with:
      abs_loss     = oracle − combined  (≥ 0)
      rel_loss_pct = 100 × abs_loss / oracle
    """
    has_lb_col = (
        "param_lamarckian_probability" in results_mis.schema
        or "param_lamarckian_probability" in results_maxcut.schema
    )
    param_cols = get_param_cols(has_lb_col)

    # Step 1: oracle — best score per (evolution_mode, problem, dataset)
    oracle_mis = (
        get_best_results_graph_agg(results_mis, "mis", supervised=False)
        .with_columns(pl.lit("mis").alias("problem"))
        .collect()
        .select(["param_evolution_mode", "problem", "dataset_name", "mean_max_score"])
        .rename({"mean_max_score": "mean_max_score_oracle"})
    )
    oracle_maxcut = (
        get_best_results_graph_agg(results_maxcut, "maxcut", supervised=False)
        .with_columns(pl.lit("maxcut").alias("problem"))
        .collect()
        .select(["param_evolution_mode", "problem", "dataset_name", "mean_max_score"])
        .rename({"mean_max_score": "mean_max_score_oracle"})
    )
    oracle = pl.concat([oracle_mis, oracle_maxcut])

    # Step 2: cross-problem best combo per evolution_mode
    best_combined = get_best_results_normalized_agg(results_mis, results_maxcut)

    # Step 3: all combis' scores per (problem, dataset) — needed to look up the
    # combined combo's score in each cell
    all_scores = pl.concat([
        _agg_per_param_dataset(results_mis, "mis", supervised=False, param_cols=param_cols)
        .with_columns(pl.lit("mis").alias("problem"))
        .collect(),
        _agg_per_param_dataset(results_maxcut, "maxcut", supervised=False, param_cols=param_cols)
        .with_columns(pl.lit("maxcut").alias("problem"))
        .collect(),
    ])

    # Step 4: filter all_scores to the combined best combo for each evolution_mode,
    # giving its score on every (problem, dataset) cell.
    # param_lamarckian_probability is null for non-lb modes; Polars inner joins treat
    # null != null so those rows would be silently dropped. Fill with a sentinel before
    # joining and drop it from the result.
    _SENTINEL = -1.0
    lb_col = "param_lamarckian_probability"
    all_scores_filled = all_scores.with_columns(pl.col(lb_col).fill_null(_SENTINEL))
    best_combined_filled = best_combined.select(param_cols).with_columns(pl.col(lb_col).fill_null(_SENTINEL))
    combined_scores = (
        all_scores_filled
        .join(best_combined_filled, on=param_cols, how="inner")
        .select(["param_evolution_mode", "problem", "dataset_name", "mean_max_score"])
        .rename({"mean_max_score": "mean_max_score_combined"})
    )

    # Step 5: join and compute losses
    return (
        oracle
        .join(combined_scores, on=["param_evolution_mode", "problem", "dataset_name"])
        .with_columns([
            (pl.col("mean_max_score_oracle") - pl.col("mean_max_score_combined")).alias("abs_loss"),
            (
                100
                * (pl.col("mean_max_score_oracle") - pl.col("mean_max_score_combined"))
                / pl.col("mean_max_score_oracle")
            ).alias("rel_loss_pct"),
        ])
        .sort(["param_evolution_mode", "problem", "dataset_name"])
    )




def run_notebook_summary(outputs: dict, config: dict):
    """
    Notebook display of best results and hyperparameters.
    For LaTeX table generation use latex_tables.make_performance_table / make_combined_params_table.
    """
    try:
        from IPython.display import display
    except ImportError:
        logger.warning("IPython not found. Skipping visual summary.")
        return

    problem = config.get("problem").lower()
    supervised = config.get("supervised")
    best_results_df = outputs["best_results"].collect()

    score_label = "MAE (Lower is better)" if (problem == "mis" and supervised) else "Score (Higher is better)"
    print(f"\n--- Best Results by Evolution Mode and Dataset: {score_label} ---")
    with pl.Config(tbl_cols=-1, tbl_rows=50):
        display(best_results_df.sort(["dataset_name", "param_evolution_mode"]))

    param_cols_to_show = [
        "dataset_name", "param_evolution_mode", "MAE", "mean_max_score", "mean_convergence_speed",
        "param_pop_size", "param_nb_offsprings", "param_crossover_rate", "param_lamarckian_probability", "param_mutation_type", "param_init_type",
    ]
    print("\n--- Best Parameters per Evolution Mode and Dataset ---")
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        cols_available = [c for c in param_cols_to_show if c in best_results_df.columns]
        display(
            best_results_df
            .select(cols_available)
            .sort(["dataset_name", "param_evolution_mode"])
        )

def run_analysis_unified(config: dict):
    """
    Main entry point for unified analysis.
    Required config keys: path_result, problem
    Optional config keys: sig_fig, supervised (default True for MIS)
    """
    path_result = config["path_result"]
    problem = config.get("problem", "mis").lower()
    sig_fig = config.get("sig_fig", 5)
    supervised = config.get("supervised", True)

    validate_data(path_result, problem)
    results = load_and_preprocess_results(path_result, problem)

    best_results = get_best_results_graph_agg(results, problem, supervised)

    agg_per_dataset = (
        results
        .group_by("dataset_name")
        .agg([
            pl.len().alias("count"),
            pl.col("max_score_found").mean().round_sig_figs(sig_fig).alias("mean_max_score"),
            pl.col("max_score_found").std().round_sig_figs(sig_fig).alias("std_max_score"),
        ])
    )

    return {
        "results": results,
        "agg_per_dataset": agg_per_dataset,
        "best_results": best_results,
    }


def preprocess_timing_results(path_timing: str) -> pl.LazyFrame:
    """Read timing parquet files produced with track_metrics=False.

    Builds a union schema from one non-lb and one lb file so that
    param_lamarckian_probability is always present, then scans all parquet
    files under path_timing.
    """
    _first = next(
        (os.path.join(r, f) for r, _, files in os.walk(path_timing)
         for f in files if f.endswith(".parquet")),
        None,
    )
    _lb = next(
        (os.path.join(r, f) for r, _, files in os.walk(path_timing)
         for f in files if f.endswith(".parquet") and "_lb" in r),
        None,
    )
    union_schema: dict = {}
    for _fp in filter(None, [_first, _lb]):
        for col, dtype in pl.read_parquet(_fp, n_rows=0).schema.items():
            if col not in union_schema:
                union_schema[col] = dtype

    return pl.scan_parquet(
        f"{path_timing}/**/*.parquet",
        hive_partitioning=True,
        schema=union_schema or None,
        missing_columns="insert",
        extra_columns="ignore",
    )


def aggregate_runtime_graph_agg(path_timing: str) -> pl.DataFrame:
    """Two-step graph-agg aggregation of ga_runtime_seconds.

    Step 1: mean over repetitions per graph.
    Step 2: mean and std over graphs → one row per (dataset_name, param_evolution_mode).

    Returns columns: dataset_name, param_evolution_mode, mean_runtime_s, std_runtime_s.
    """
    results = preprocess_timing_results(path_timing)

    param_cols = ["param_evolution_mode"]

    pl.set_random_seed(42)
    df = results.collect().sort([*param_cols, "dataset_name", "graph_id", "repetition"])

    agg_per_graph = (
        df
        .group_by([*param_cols, "dataset_name", "graph_id"])
        .agg([
            pl.col("ga_runtime_seconds").mean().alias("g_id_mean_runtime"),
            pl.col("ga_runtime_seconds").std().alias("g_id_std_runtime"),
        ])
        .sort([*param_cols, "dataset_name", "graph_id"])
    )

    return (
        agg_per_graph
        .group_by([*param_cols, "dataset_name"])
        .agg([
            pl.col("g_id_mean_runtime").mean().alias("mean_runtime_s"),
            pl.col("g_id_std_runtime").mean().alias("std_runtime_s"),
            pl.len().alias("count_graphs"),
        ])
        .sort([*param_cols, "dataset_name"])
    )


def get_experiment_params(path_results: str):
    """
    Retrieve unique parameter values used in an experiment from the parquet results.
    """
    parquet_path = f"{path_results}/parquet/**/*.parquet" if not path_results.endswith("parquet") else f"{path_results}/**/*.parquet"
    
    # We use scan_parquet and collect a small subset or just the unique values for efficiency
    df = pl.scan_parquet(parquet_path, hive_partitioning=True)
    param_cols = [c for c in df.schema.keys() if c.startswith("param_")]
    
    unique_params = {}
    for col in param_cols:
        # Collect unique values for each parameter column
        vals = df.select(col).unique().collect().get_column(col).to_list()
        # Sort for readability if possible
        try:
            vals.sort()
        except TypeError:
            pass
        unique_params[col] = vals
        
    return unique_params

if __name__ == "__main__":
    # Example usage / test
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        prob = sys.argv[2] if len(sys.argv) > 2 else "mis"
        sup = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True

        config = {
            "path_result": path,
            "problem": prob,
            "supervised": sup
        }

        outputs = run_analysis_unified(config)
        print("Analysis complete.")
        print(outputs["agg_per_dataset"].collect())
        print(outputs["best_results"].collect())
    else:
        print("Usage: python aggregate_results_unified.py <path_result> <problem> <supervised>")
