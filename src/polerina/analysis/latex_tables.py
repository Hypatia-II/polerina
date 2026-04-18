"""
Generate LaTeX tables from best-result DataFrames produced by get_best_results_graph_agg.

Expected columns: dataset_name, param_evolution_mode,
  mean_max_score, std_max_score,
  param_pop_size, param_nb_offsprings, param_crossover_rate
"""
import polars as pl

MEAN_COL = "mean_max_score"
STD_COL  = "std_max_score"

DATASET_ORDER = ["co_er_small", "co_er_large", "co_ba_small", "co_ba_large", "co_rb_small", "co_rb_large"]

DATASET_LABELS = {
    "co_ba_small": "BA Small",
    "co_ba_large": "BA Large",
    "co_er_small": "ER Small",
    "co_er_large": "ER Large",
    "co_rb_small": "RB Small",
    "co_rb_large": "RB Large",
}

DATASET_LABELS_LONG = {
    "co_er_small": "ER Small",
    "co_er_large": "ER Large",
    "co_ba_small": "BA Small",
    "co_ba_large": "BA Large",
    "co_rb_small": "RB Small",
    "co_rb_large": "RB Large",
}

MODE_ORDER  = ["darwin", "baldwin", "lamarck", "lb"]
MODE_LABELS = {"darwin": "Darwinian", "baldwin": "Baldwinian", "lamarck": "Lamarckian", "lb": "L-B"}

BENCHMARK_ORDER = ["GIN", "GT", "MLP", "DeepSet", "Solver"]

# {method: {dataset_name: (mean, std)}}
MIS_BENCHMARKS: dict[str, dict[str, tuple[float, float]]] = {
    "GIN":     {"co_er_small": (25.418, 0.407), "co_er_large": (26.276, 0.408), "co_ba_small": (100.16, 3.674), "co_ba_large": (135.00, 0.720), "co_rb_small": (17.294, 0.328), "co_rb_large": (13.999, 0.321)},
    "GT":      {"co_er_small": (22.984, 0.473), "co_er_large": (24.980, 0.292), "co_ba_small": (99.579, 6.448), "co_ba_large": (114.26, 0.601), "co_rb_small": (16.542, 0.477), "co_rb_large": (13.406, 0.140)},
    "MLP":     {"co_er_small": (23.183, 0.016), "co_er_large": (24.259, 0.449), "co_ba_small": (95.108, 2.042), "co_ba_large": (114.49, 0.758), "co_rb_small": (16.105, 0.097), "co_rb_large": (13.040, 0.214)},
    "DeepSet": {"co_er_small": (23.050, 0.061), "co_er_large": (24.220, 0.056), "co_ba_small": (95.076, 0.173), "co_ba_large": (114.89, 0.016), "co_rb_small": (16.021, 0.032), "co_rb_large": (13.183, 0.035)},
    "Solver":  {"co_er_small": (33.604, 1.428), "co_er_large": (45.637, 0.631), "co_ba_small": (142.86, 16.54), "co_ba_large": (433.77, 19.17), "co_rb_small": (20.803, 1.817), "co_rb_large": (42.547, 4.449)},
}

MC_BENCHMARKS: dict[str, dict[str, tuple[float, float]]] = {
    "GIN":     {"co_er_small": (2327.9, 24.78),  "co_er_large": (20878.0, 107.9), "co_ba_small": (397.00, 0.605), "co_ba_large": (1044.1, 0.649),  "co_rb_small": (2106.7, 14.62),  "co_rb_large": (24748.0, 87.76)},
    "GT":      {"co_er_small": (2172.7, 91.75),  "co_er_large": (16534.0, 278.0), "co_ba_small": (363.76, 0.639), "co_ba_large": (986.93, 3.128),  "co_rb_small": (1925.7, 32.75),  "co_rb_large": (21524.0, 184.0)},
    "MLP":     {"co_er_small": (1866.7, 67.64),  "co_er_large": (7335.4, 57.49),  "co_ba_small": (308.73, 0.224), "co_ba_large": (929.20, 4.060),  "co_rb_small": (1727.7, 165.1),  "co_rb_large": (20357.0, 249.6)},
    "DeepSet": {"co_er_small": (33.634, 20.84),  "co_er_large": (27.663, 6.763),  "co_ba_small": (1.0669, 0.800), "co_ba_large": (154.31, 151.5),  "co_rb_small": (140.02, 155.5),  "co_rb_large": (3575.9, 730.0)},
    "Solver":  {"co_er_small": (2835.5, 607.6),  "co_er_large": (23884.0, 1809.0),"co_ba_small": (460.91, 50.13), "co_ba_large": (1260.4, 48.81),  "co_rb_small": (2920.1, 97.23),  "co_rb_large": (33914.0, 7861.0)},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pivot(df: pl.DataFrame, value_col: str) -> dict:
    """Return {evolution_mode: {dataset: value}}."""
    result = {}
    for row in df.iter_rows(named=True):
        result.setdefault(row["param_evolution_mode"], {})[row["dataset_name"]] = row[value_col]
    return result


def _best_mode_per_dataset(df: pl.DataFrame) -> dict[str, set]:
    """Return {dataset_name: set of evolution_modes tied for highest mean}."""
    max_per_dataset = df.group_by("dataset_name").agg(pl.col(MEAN_COL).max().alias("max_mean"))
    joined = df.join(max_per_dataset, on="dataset_name").filter(pl.col(MEAN_COL) == pl.col("max_mean"))
    result: dict[str, set] = {}
    for row in joined.iter_rows(named=True):
        result.setdefault(row["dataset_name"], set()).add(row["param_evolution_mode"])
    return result


def _best_benchmark_per_dataset(benchmarks: dict[str, dict[str, tuple[float, float]]]) -> dict[str, set]:
    """Return {dataset_name: set of benchmark method names tied for highest mean}."""
    max_per_dataset: dict[str, float] = {}
    for ds_vals in benchmarks.values():
        for ds, (mean, _) in ds_vals.items():
            if mean > max_per_dataset.get(ds, float("-inf")):
                max_per_dataset[ds] = mean
    result: dict[str, set] = {}
    for method, ds_vals in benchmarks.items():
        for ds, (mean, _) in ds_vals.items():
            if mean == max_per_dataset.get(ds):
                result.setdefault(ds, set()).add(method)
    return result


def _score_decimals(df: pl.DataFrame) -> int:
    max_mean = df[MEAN_COL].max()
    return 1 if (max_mean is not None and max_mean > 1000) else 2


def _fmt_score(mean: float, std: float, bold: bool, decimals: int = 2) -> str:
    fmt = f"{{:.{decimals}f}}"
    mean_str = fmt.format(mean)
    std_str  = fmt.format(std) if std is not None else None
    if bold:
        mean_part = f"$\\mathbf{{{mean_str}}}$"
        std_part  = f" {{\\scriptsize $\\pm$ $\\mathbf{{{std_str}}}$}}" if std_str else ""
    else:
        mean_part = f"${mean_str}$"
        std_part  = f" {{\\scriptsize $\\pm$ ${std_str}$}}" if std_str else ""
    return mean_part + std_part


def _fmt_params(pop: int, off: int, cr: float) -> str:
    return f"${pop}/{off}/{cr:.1f}$"


def _col_header(datasets: list[str]) -> str:
    labels = " & ".join(f"\\textbf{{{DATASET_LABELS[d]}}}" for d in datasets)
    return f"\\textbf{{Problem}} & \\textbf{{Type}} & {labels}"


def _problem_rows_perf(
    df: pl.DataFrame,
    datasets: list[str],
    best_per_dataset: dict,
    decimals: int,
) -> list[str]:
    means = _pivot(df, MEAN_COL)
    stds  = _pivot(df, STD_COL)
    lines = []
    for mode in MODE_ORDER:
        cells = []
        for ds in datasets:
            m = means.get(mode, {}).get(ds)
            s = stds.get(mode, {}).get(ds)
            if m is None:
                cells.append("--")
            else:
                cells.append(_fmt_score(m, s, mode in best_per_dataset.get(ds, set()), decimals))
        lines.append(f"   & {MODE_LABELS[mode]} & " + " & ".join(cells) + r" \\")
    return lines


def _benchmark_rows(
    benchmarks: dict[str, dict[str, tuple[float, float]]],
    datasets: list[str],
    best_per_dataset: dict[str, set],
    decimals: int,
    problem_label: str = "",
    multirow_count: int = 0,
) -> list[str]:
    lines = []
    methods = [m for m in BENCHMARK_ORDER if m in benchmarks]
    for i, method in enumerate(methods):
        prefix = f"\\multirow{{{multirow_count}}}{{*}}{{{problem_label}}}" if (i == 0 and problem_label) else ""
        ds_vals = benchmarks[method]
        cells = []
        for ds in datasets:
            entry = ds_vals.get(ds)
            if entry is None:
                cells.append("--")
            else:
                mean, std = entry
                cells.append(_fmt_score(mean, std, method in best_per_dataset.get(ds, set()), decimals))
        lines.append(f"  {prefix} & {method} & " + " & ".join(cells) + r" \\")
    return lines



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_performance_table(
    mis: pl.DataFrame,
    mc: pl.DataFrame,
    mis_benchmarks: dict[str, dict[str, tuple[float, float]]] = MIS_BENCHMARKS,
    mc_benchmarks: dict[str, dict[str, tuple[float, float]]] = MC_BENCHMARKS,
    caption: str = (
        "Best mean score ($\\pm$ std) per evolution type and dataset. "
        "Bold = best among GA types; bold among benchmark rows = best benchmark per dataset."
    ),
    label: str = "tab:performance",
) -> str:
    """Table 1: mean ± std, rows = evolution modes + benchmarks, columns = datasets."""
    datasets = [d for d in DATASET_ORDER if d in mis["dataset_name"].unique().to_list()]
    col_spec = "ll" + "r" * len(datasets)

    mis_bench_methods = [m for m in BENCHMARK_ORDER if m in mis_benchmarks]
    mc_bench_methods  = [m for m in BENCHMARK_ORDER if m in mc_benchmarks]
    mis_total = len(MODE_ORDER) + len(mis_bench_methods)
    mc_total  = len(MODE_ORDER) + len(mc_bench_methods)

    mis_decimals = _score_decimals(mis)
    mc_decimals  = _score_decimals(mc)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        _col_header(datasets) + r" \\",
        "\\midrule",
    ]
    cmidrule = "\\cmidrule{2-" + str(2 + len(datasets)) + "}"
    lines += _benchmark_rows(mis_benchmarks, datasets, _best_benchmark_per_dataset(mis_benchmarks), mis_decimals, "MIS", mis_total)
    if mis_bench_methods:
        lines += [cmidrule]
    lines += _problem_rows_perf(mis, datasets, _best_mode_per_dataset(mis), mis_decimals)
    lines += ["\\midrule"]
    lines += _benchmark_rows(mc_benchmarks, datasets, _best_benchmark_per_dataset(mc_benchmarks), mc_decimals, "MC", mc_total)
    if mc_bench_methods:
        lines += [cmidrule]
    lines += _problem_rows_perf(mc, datasets, _best_mode_per_dataset(mc), mc_decimals)
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)



def _best_mode_per_dataset_col(df: pl.DataFrame, value_col: str, higher_is_better: bool = True) -> dict[str, set]:
    """Return {dataset_name: set of evolution_modes tied for best value_col}."""
    if higher_is_better:
        best_per_dataset = df.group_by("dataset_name").agg(pl.col(value_col).max().alias("best_val"))
    else:
        best_per_dataset = df.group_by("dataset_name").agg(pl.col(value_col).min().alias("best_val"))
    joined = df.join(best_per_dataset, on="dataset_name").filter(pl.col(value_col) == pl.col("best_val"))
    result: dict[str, set] = {}
    for row in joined.iter_rows(named=True):
        result.setdefault(row["dataset_name"], set()).add(row["param_evolution_mode"])
    return result


def _problem_rows_unique(
    df: pl.DataFrame,
    datasets: list[str],
    best_per_dataset: dict,
    decimals: int,
    problem_label: str = "",
) -> list[str]:
    means = _pivot(df, "mean_nb_unique")
    stds  = _pivot(df, "std_nb_unique")
    lines = []
    for i, mode in enumerate(MODE_ORDER):
        label_cell = f"\\multirow{{{len(MODE_ORDER)}}}{{*}}{{{problem_label}}}" if (i == 0 and problem_label) else ""
        cells = []
        for ds in datasets:
            m = means.get(mode, {}).get(ds)
            s = stds.get(mode, {}).get(ds)
            if m is None:
                cells.append("--")
            else:
                cells.append(_fmt_score(m, s, mode in best_per_dataset.get(ds, set()), decimals))
        lines.append(f"   {label_cell} & {MODE_LABELS[mode]} & " + " & ".join(cells) + r" \\")
    return lines


def make_unique_solutions_table(
    mis: pl.DataFrame,
    mc: pl.DataFrame,
    caption: str = (
        "Mean number of unique optimal solutions found ($\\pm$ std) per evolution type and dataset. "
        "Bold = most unique solutions among GA types per dataset."
    ),
    label: str = "tab:unique_solutions",
) -> str:
    """Table: mean ± std of nb_unique_optimal_solutions, rows = evolution modes, columns = datasets."""
    datasets = [d for d in DATASET_ORDER if d in mis["dataset_name"].unique().to_list()]
    col_spec = "ll" + "r" * len(datasets)

    mis_best = _best_mode_per_dataset_col(mis, "mean_nb_unique", higher_is_better=True)
    mc_best  = _best_mode_per_dataset_col(mc,  "mean_nb_unique", higher_is_better=True)

    # Use 1 decimal for unique solution counts (they are averaged over graphs/repetitions)
    decimals = 1

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        _col_header(datasets) + r" \\",
        "\\midrule",
    ]
    lines += _problem_rows_unique(mis, datasets, mis_best, decimals, problem_label="MIS")
    lines += ["\\midrule"]
    lines += _problem_rows_unique(mc,  datasets, mc_best,  decimals, problem_label="MC")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def _fmt_combined_combos(best_combined: pl.DataFrame) -> str:
    """Format the cross-problem best combo per evolution mode as a LaTeX caption snippet.

    Produces one entry per mode in MODE_ORDER, e.g.:
      Darwinian: $50/10/0.8$; Baldwinian: $100/20/0.6$; ...; L-B: $50/10/0.8$ ($p_\\text{LB}=0.5$).
    """
    index = {row["param_evolution_mode"]: row for row in best_combined.iter_rows(named=True)}
    parts = []
    for mode in MODE_ORDER:
        row = index.get(mode)
        if row is None:
            continue
        params = _fmt_params(row["param_pop_size"], row["param_nb_offsprings"], row["param_crossover_rate"])
        entry = f"{MODE_LABELS[mode]}: {params}"
        plb = row.get("param_lamarckian_probability")
        if mode == "lb" and plb is not None:
            entry += f" ($p_{{\\text{{LB}}}}={plb:.2g}$)"
        parts.append(entry)
    return "; ".join(parts) + "."


def _fmt_loss(value: float, bold: bool) -> str:
    inner = f"{value:.3f}"
    return f"$\\mathbf{{{inner}}}$" if bold else f"${inner}$"


def _problem_rows_loss(
    df: pl.DataFrame,
    problem: str,
    datasets: list[str],
    best_per_dataset: dict[str, set],
    problem_label: str = "",
) -> list[str]:
    sub = df.filter(pl.col("problem") == problem)
    losses = _pivot(sub, "rel_loss_pct")

    # avg loss per mode for this problem; bold the minimum
    avg_loss = {
        row["param_evolution_mode"]: row["rel_loss_pct"]
        for row in sub.group_by("param_evolution_mode")
                      .agg(pl.col("rel_loss_pct").mean())
                      .iter_rows(named=True)
    }
    min_avg = min(avg_loss.values()) if avg_loss else None

    lines = []
    for i, mode in enumerate(MODE_ORDER):
        label_cell = f"\\multirow{{{len(MODE_ORDER)}}}{{*}}{{{problem_label}}}" if (i == 0 and problem_label) else ""
        cells = []
        for ds in datasets:
            v = losses.get(mode, {}).get(ds)
            cells.append("--" if v is None else _fmt_loss(v, mode in best_per_dataset.get(ds, set())))
        avg = avg_loss.get(mode)
        avg_cell = "--" if avg is None else _fmt_loss(avg, avg == min_avg)
        lines.append(f"   {label_cell} & {MODE_LABELS[mode]} & " + " & ".join(cells) + f" & {avg_cell}" + r" \\")
    return lines


def _col_header_loss_grouped(datasets: list[str], include_avg: bool = True) -> tuple[str, str, str]:
    """Return (header_row1, cmidrule_line, header_row2) for a two-level grouped header.

    Datasets are expected in pairs (Small, Large) per graph family (ER, BA, RB).
    Column indices start at 3 (after Problem and Type).
    include_avg: whether to append a multirow Avg. column in the header.
    """
    # Build ordered groups, preserving insertion order
    groups: dict[str, dict] = {}
    for col_idx, ds in enumerate(datasets, start=3):
        family = ds.split("_")[1].upper()  # co_er_small -> ER
        if family not in groups:
            groups[family] = {"start": col_idx, "end": col_idx, "sub": []}
        else:
            groups[family]["end"] = col_idx
        sub = DATASET_LABELS[ds].split(" ")[1]  # "Small" or "Large"
        groups[family]["sub"].append(f"\\textbf{{{sub}}}")

    group_cells = []
    cmidrules = []
    sub_cells = []
    for family, g in groups.items():
        n = len(g["sub"])
        group_cells.append(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{family}}}}}")
        cmidrules.append(f"\\cmidrule(lr){{{g['start']}-{g['end']}}}")
        sub_cells.extend(g["sub"])

    avg_suffix  = " & \\multirow{2}{*}{\\textbf{Avg.}} \\\\" if include_avg else " \\\\"
    avg_row2    = " & \\\\" if include_avg else " \\\\"
    row1 = (
        "\\multirow{2}{*}{\\textbf{Problem}} & \\multirow{2}{*}{\\textbf{Type}} & "
        + " & ".join(group_cells)
        + avg_suffix
    )
    cmidrule_line = "".join(cmidrules)
    row2 = " & & " + " & ".join(sub_cells) + avg_row2
    return row1, cmidrule_line, row2


def make_normalized_agg_loss_table(
    loss_df: pl.DataFrame,
    best_combined: pl.DataFrame | None = None,
    caption: str = (
        "Relative loss (\\%) when using the cross-problem best parameter combo "
        "instead of the problem- and dataset-specific oracle. "
        "Bold = evolution type with smallest loss per dataset or column."
    ),
    label: str = "tab:normalized_agg_loss",
) -> str:
    """Table: rel_loss_pct per (evolution_mode, problem, dataset) + avg loss column.

    Rows = evolution modes grouped by problem (MIS then MaxCut), with multirow
    problem labels in the first column.
    Columns = datasets + Avg. (mean rel_loss_pct across datasets for that problem).
    Bold = minimum loss per (problem, dataset) column; minimum avg loss per problem section.
    """
    if best_combined is not None:
        caption = caption + " Best cross-problem combos ($\\mu/\\lambda/r_c$): " + _fmt_combined_combos(best_combined)

    datasets = [d for d in DATASET_ORDER if d in loss_df["dataset_name"].unique().to_list()]
    col_spec = "ll" + "r" * len(datasets) + "|r"

    mis_sub = loss_df.filter(pl.col("problem") == "mis")
    mc_sub  = loss_df.filter(pl.col("problem") == "maxcut")
    mis_best = _best_mode_per_dataset_col(mis_sub, "rel_loss_pct", higher_is_better=False)
    mc_best  = _best_mode_per_dataset_col(mc_sub,  "rel_loss_pct", higher_is_better=False)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\setlength{\\tabcolsep}{5pt}%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        _col_header(datasets) + r" & \textbf{Avg.} \\",
        "\\midrule",
    ]
    lines += _problem_rows_loss(loss_df, "mis",    datasets, mis_best, "MIS")
    lines += ["\\midrule"]
    lines += _problem_rows_loss(loss_df, "maxcut", datasets, mc_best,  "MC")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def _fmt_runtime(mean: float, std: float, bold: bool) -> str:
    return _fmt_score(mean, std, bold, decimals=2)


def _problem_rows_runtime(
    df: pl.DataFrame,
    datasets: list[str],
    best_per_dataset: dict,
    problem_label: str = "",
) -> list[str]:
    means = _pivot(df, "mean_runtime_s")
    stds  = _pivot(df, "std_runtime_s")
    lines = []
    for i, mode in enumerate(MODE_ORDER):
        label_cell = f"\\multirow{{{len(MODE_ORDER)}}}{{*}}{{{problem_label}}}" if (i == 0 and problem_label) else ""
        cells = []
        for ds in datasets:
            m = means.get(mode, {}).get(ds)
            s = stds.get(mode, {}).get(ds)
            if m is None:
                cells.append("--")
            else:
                cells.append(_fmt_runtime(m, s, mode in best_per_dataset.get(ds, set())))
        lines.append(f"   {label_cell} & {MODE_LABELS[mode]} & " + " & ".join(cells) + r" \\")
    return lines


def make_runtime_table(
    mis: pl.DataFrame,
    mc: pl.DataFrame,
    caption: str = (
        "Mean GA execution time in seconds ($\\pm$ std) per evolution type and dataset. "
        "Bold = fastest evolution type per dataset."
    ),
    label: str = "tab:runtime",
) -> str:
    """Table: mean ± std of ga_runtime_seconds, rows = evolution modes, columns = datasets."""
    datasets = [d for d in DATASET_ORDER if d in mis["dataset_name"].unique().to_list()]
    col_spec = "ll" + "r" * len(datasets)

    mis_best = _best_mode_per_dataset_col(mis, "mean_runtime_s", higher_is_better=False)
    mc_best  = _best_mode_per_dataset_col(mc,  "mean_runtime_s", higher_is_better=False)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        _col_header(datasets) + r" \\",
        "\\midrule",
    ]
    lines += _problem_rows_runtime(mis, datasets, mis_best, problem_label="MIS")
    lines += ["\\midrule"]
    lines += _problem_rows_runtime(mc,  datasets, mc_best,  problem_label="MC")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def _fmt_plb(prob, strategy: str) -> str:
    if strategy != "lb" or prob is None:
        return "---"
    return f"{prob:.2g}"


def make_combined_params_table(
    mis: pl.DataFrame,
    mc: pl.DataFrame,
    caption: str = (
        r"Best Evolutionary Type Hyperparameter Configuration for MIS and MC. "
        r"All configurations used Bernoulli mutation and Random initialization. "
        r"Values shown as $\mu/\lambda/r_c$; Darwin, Baldwin, and Lamarck always "
        r"have $p_\text{LB}=0$ (---)."
    ),
    label: str = "tab:evolution_params_combined",
) -> str:
    """Combined params table: rows = datasets × evolution modes, columns = MIS | MC."""

    def _index(df: pl.DataFrame) -> dict:
        return {
            (row["dataset_name"], row["param_evolution_mode"]): row
            for row in df.iter_rows(named=True)
        }

    mis_idx = _index(mis)
    mc_idx  = _index(mc)
    datasets = [d for d in DATASET_ORDER if d in mis["dataset_name"].unique().to_list()]
    num_modes = len(MODE_ORDER)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.0}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        r"& & \multicolumn{2}{c}{\textbf{MIS}} & \multicolumn{2}{c}{\textbf{MC}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"\textbf{Dataset} & \textbf{Type} & $\boldsymbol{\mu/\lambda/r_c}$ & $\boldsymbol{p_\text{LB}}$"
        r" & $\boldsymbol{\mu/\lambda/r_c}$ & $\boldsymbol{p_\text{LB}}$ \\ \midrule",
    ]

    for d_idx, dataset in enumerate(datasets):
        dataset_label = DATASET_LABELS_LONG.get(dataset, dataset.replace("_", " ").title())

        for m_idx, mode in enumerate(MODE_ORDER):
            mis_row = mis_idx.get((dataset, mode))
            mc_row  = mc_idx.get((dataset, mode))

            mis_params = _fmt_params(mis_row["param_pop_size"], mis_row["param_nb_offsprings"], mis_row["param_crossover_rate"]) if mis_row else "--"
            mis_plb    = _fmt_plb(mis_row.get("param_lamarckian_probability") if mis_row else None, mode)
            mc_params  = _fmt_params(mc_row["param_pop_size"],  mc_row["param_nb_offsprings"],  mc_row["param_crossover_rate"])  if mc_row  else "--"
            mc_plb     = _fmt_plb(mc_row.get("param_lamarckian_probability")  if mc_row  else None, mode)

            dataset_cell = f"\\multirow{{{num_modes}}}{{*}}{{{dataset_label}}}" if m_idx == 0 else ""
            mode_name = MODE_LABELS[mode]

            lines.append(f"    {dataset_cell} & {mode_name} & {mis_params} & {mis_plb} & {mc_params} & {mc_plb} \\\\")

        lines.append("\\cmidrule{1-6}" if d_idx < len(datasets) - 1 else "\\bottomrule")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

