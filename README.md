# POLERINA — Problem Optimization via Learning & Evolutionary Research for Intelligent Network Analysis


**Polerina: Small, efficient, and handles NP-hard problems with total grace! 🩰✨**

A research codebase that uses a **Genetic Algorithm (GA)** to solve the **Maximum Independent Set (MIS)** and **Maximum Cut (MC)** problems on graphs, with a focus on hyperparameter tuning and performance benchmarking.

## Overview

Given an undirected graph, the MIS problem asks for the largest subset of vertices such that no two vertices in the subset are connected by an edge. The MC problem asks for a partition of vertices into two sets such that the number of edges between them is maximized.

This project implements a GA-based solver and provides tooling to:

- Run the GA on **synthetic graphs** (Erdős-Rényi) or **benchmark datasets** from [GraphBench](https://github.com/graphbench/package) (Erdős-Rényi, Barabási-Albert, RB).
- Perform **grid-search hyperparameter tuning** with parallel execution across graphs.
- **Analyze results** with a unified aggregator for MIS and MC.

## Installation

**Requirements:** Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd polerina

# Install the package in editable mode
uv sync

# Include Jupyter support (needed for live visualization during GA runs)
uv sync --extra notebook
```

This installs the `polerina` package with a single CLI command `polerina` that provides several subcommands.

## Quick Start

### 1. Download and convert benchmark data

Benchmark graphs are provided by [GraphBench](https://github.com/graphbench/package) and converted to `.npy` files under `data/numpy_data/`.

Datasets follow the naming convention `co_<model>_<size>` where `co` stands for Combinatorial Optimization, `<model>` is one of `er` (Erdős-Rényi), `ba` (Barabási-Albert), or `rb` (RB-Graphs), and `<size>` is `small` (200–300 nodes) or `large` (700–1200 nodes).

```bash
polerina download                              # uses config/download.toml defaults
polerina download --dataset co_er_small        # override dataset
```

### 2. Run the solver or hyperparameter search

The GA can be run in two modes: a single solver run with specific parameters, or a parallelized hyperparameter grid search.

**Run the GA solver:**
```bash
polerina run-solver                            # uses config/run_solver.toml defaults
polerina run-solver --dataset co_er_small      # override datasets
polerina run-solver --config my_solver.toml    # use a custom config
```

**Run the hyperparameter tuning:**
```bash
polerina run-hyperparameter-tuning             # uses config/run_hp_tuning.toml defaults
polerina run-hyperparameter-tuning --reps 5    # override repetitions per graph
```

These commands perform the GA execution, optionally in parallel for grid searches. Results are saved as hive-partitioned Parquet files under `results/`.

### 3. Analyze results

```bash
polerina analyze                               # uses config/analyze.toml defaults
polerina analyze --path-result results/.../parquet --problem mis
```

Edit `config/analyze.toml` to set `path_result` to your actual results directory.

## Project Structure

```
polerina/
├── pyproject.toml
├── README.md
├── config/
│   ├── run_solver.toml              # GA single-run solver config
│   ├── run_hp_tuning.toml           # GA hyperparameter tuning config
│   ├── download.toml                # Dataset download config
│   └── analyze.toml                 # Results analysis config
├── src/polerina/
    ├── cli.py                       # Typer CLI: run-solver, analyze, etc.
    ├── config.py                    # msgspec Structs and TOML loading
    ├── runners/                     # High-level workflow orchestrators
    │   ├── hp_tuning.py             # Parallel hyperparameter tuning
    │   ├── solver.py                # Single GA solver execution
    ├── ga/
    │   ├── genetic_algorithm.py     # GeneticAlgorithm class (core GA loop)
    │   ├── engine.py                # Experiment runners and parallel dispatch
    │   └── problems/                # Problem definitions (MIS, MC)
    ├── data_handler/
    │   ├── data_loader.py           # Load .npy benchmark data
    │   └── data_prep.py             # graphbench -> .npy conversion
    └── analysis/
        └── aggregate_results_unified.py # Unified aggregator for all problems
```

## How It Works

### Genetic Algorithm

Each individual is a **binary vector** of length *n* (number of nodes).

| Component | Description |
|---|---|
| **Initialization** | `"independent"` — greedy construction of valid independent sets (MIS only); `"random"` — uniform random binary vectors; `"prob"` — probabilistic initialization |
| **Crossover** | Uniform crossover with a configurable crossover rate. Individuals that skip crossover are cloned from the population. |
| **Mutation** | `"bernoulli"` — flip each bit with probability 1/*n*; `"fast_mutation"` — power-law distributed mutation rates |
| **Local search** | Problem-specific Numba JIT-compiled operator run to local optimality. For MIS: iteratively removes the most-violating node until the solution is feasible. For MC: iteratively flips the node yielding the highest cut gain. Used internally by Lamarck and Baldwin modes. |
| **Evaluation** | MIS: valid set → set size; invalid set → negative penalty (number of internal edges). MC: cut size (all individuals are valid). |
| **Evolution mode** | Controls lifetime learning: `"darwin"` (none), `"baldwin"` (lookahead scoring), `"lamarck"` (in-place local search), `"lb"` (probabilistic assimilation). See Evolution Modes below. |
| **Selection** | (mu + lambda) selection: keep the best *pop_size* individuals from parents + offspring |

### Evolution Modes

The GA supports four evolutionary strategies via a single `evolution_mode` parameter:

| `evolution_mode` | `lamarckian_probability` | Description |
|---|---|---|
| `"darwin"` | — | Standard GA: no local search. Raw offspring evaluated directly. |
| `"baldwin"` | — | Lookahead: offspring are scored using their local-search result, but the originals stay in the population. |
| `"lamarck"` | — | Full in-place local search before evaluation. The offspring is permanently replaced by its improved version. |
| `"lb"` | *p* ∈ (0, 1) | Lookahead + probabilistic assimilation: like Baldwin, but each offspring is replaced by its local-search result with probability *p*. |

`lamarckian_probability` is only required when `evolution_mode = "lb"`. For HP tuning, list it alongside `"lb"` in `evolution_mode`; the grid generator automatically cross-products probabilities only for `lb` entries.

**Evaluation budget:** Fixed at 40,000 fitness evaluations per run. The number of iterations is computed as `(40000 - pop_size) / nb_offsprings`.

### Parallelization

The grid search distributes work using `joblib.Parallel(n_jobs=-1)`. Each parallel task processes **one graph** through all parameter combinations and repetitions, then writes a single Parquet file.

### Result Format

Results are stored as hive-partitioned Parquet files:

```
results/parquet/dataset_name={name}/graph_id={id}/results.parquet
```

Each row contains:
- **Graph metadata**: `graph_id`, `dataset_name`, `problem`, `nb_nodes`, `true_size_set` (MIS only), `repetition`, `random_seed`
- **GA parameters** (prefixed with `param_`): `param_evolution_mode`, `param_pop_size`, `param_nb_offsprings`, `param_init_type`, `param_crossover_rate`, `param_mutation_type`, `param_lamarckian_probability` (L-B only)
- **Metrics**: `max_score_found`, `score_init`, `best_max_score_ever`, `nb_unique_optimal_solutions`, `ga_runtime_seconds`
- **Histories**: `best_score_history` (score at each iteration), `diversity_history` (mean Hamming distance at each iteration), `improvements` (up to 5 most recent score improvements)

## GA Parameters Reference

| Parameter | Values | Description |
|---|---|---|
| `pop_size` | int | Population size |
| `nb_offsprings` | int | Number of offspring per generation |
| `init_type` | `"independent"`, `"random"`, `"prob"` | Population initialization strategy |
| `evolution_mode` | `"darwin"`, `"baldwin"`, `"lamarck"`, `"lb"` | Evolutionary strategy (see Evolution Modes above) |
| `lamarckian_probability` | float in (0, 1) | Assimilation probability — only required when `evolution_mode = "lb"` |
| `crossover_rate` | 0.0 -- 1.0 | Probability of crossover (vs. cloning) |
| `mutation_type` | `"bernoulli"`, `"fast_mutation"` | Mutation operator |

## Dependencies

Core: `numpy`, `scipy`, `numba`, `networkx`, `joblib`, `polars`, `pyarrow`, `matplotlib`, `scikit-learn`, `tqdm`, `pandas`, `graphbench-lib`, `msgspec`, `typer`

Optional (live visualization): `ipython`, `jupyter`, `marimo`
