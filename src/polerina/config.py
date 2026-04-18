"""Typed configuration schemas and TOML loading for POLERINA CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal, Optional

import msgspec

from polerina import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# RunHPTuningConfig structs
# ---------------------------------------------------------------------------

class DataHPTuningConfig(msgspec.Struct, kw_only=True):
    dataset_name: list[str]
    dataset_split: str
    sample: float = None

class GraphHPTuningConfig(msgspec.Struct, kw_only=True):
    dataset_size: int
    num_nodes: int
    p: float


class GAValuesHPTuningConfig(msgspec.Struct, kw_only=True):
    pop_size: list[int]
    nb_offsprings: list[int]
    init_type: list[str]
    evolution_mode: list[Literal["darwin", "baldwin", "lamarck", "lb"]]
    crossover_rate: list[float]
    mutation_type: list[str]
    lamarckian_probability: Optional[list[Annotated[float, msgspec.Meta(gt=0, lt=1)]]] = None


class OutputHPTuningConfig(msgspec.Struct, kw_only=True):
    output_path: Optional[str] = None
    visualize: bool
    visualize_reps: int
    plot_display: str


class RunHPTuningConfig(msgspec.Struct, kw_only=True):
    data: Optional[DataHPTuningConfig] = None
    graph: Optional[GraphHPTuningConfig] = None

    ga: GAValuesHPTuningConfig
    output: OutputHPTuningConfig
    synthetic_data: bool
    nb_reps_per_graph: int
    problem_name: str 

# ---------------------------------------------------------------------------
# RunSolverConfig structs
# ---------------------------------------------------------------------------

class DataSolverConfig(msgspec.Struct, kw_only=True):
    dataset_name: list[str]
    dataset_split: str
    sample: float = None

class GraphSolverConfig(msgspec.Struct, kw_only=True):
    dataset_size: int
    num_nodes: int
    p: float


class GAValuesSolverConfig(msgspec.Struct, kw_only=True):
    pop_size: int
    nb_offsprings: int
    init_type: str
    evolution_mode: Literal["darwin", "baldwin", "lamarck", "lb"]
    crossover_rate: float
    mutation_type: str
    lamarckian_probability: Optional[Annotated[float, msgspec.Meta(gt=0, lt=1)]] = None


class OutputSolverConfig(msgspec.Struct, kw_only=True):
    output_path: Optional[str] = None
    visualize: bool
    visualize_reps: int
    plot_display: str


class RunSolverConfig(msgspec.Struct, kw_only=True):
    data: Optional[DataSolverConfig] = None
    graph: Optional[GraphSolverConfig] = None

    ga: GAValuesSolverConfig
    output: OutputSolverConfig
    synthetic_data: bool
    problem_name: str 


# ---------------------------------------------------------------------------
# RunBestParamsConfig structs
# ---------------------------------------------------------------------------

class RunBestParamsConfig(msgspec.Struct, kw_only=True):
    results_path: str
    problem_name: str
    aggregation_method: Literal["graph_agg"]
    nb_reps_per_graph: int
    supervised: bool
    data: DataHPTuningConfig
    output: OutputHPTuningConfig
    resume_path: Optional[str] = None
    combis_path: Optional[str] = None
    track_metrics: bool = True


# ---------------------------------------------------------------------------
# DownloadConfig
# ---------------------------------------------------------------------------

class DownloadConfig(msgspec.Struct, kw_only=True):
    dataset_name: str
    delete_raw: bool
    splits: list


# ---------------------------------------------------------------------------
# AnalyzeConfig
# ---------------------------------------------------------------------------

class AnalyzeConfig(msgspec.Struct, kw_only=True):
    path_result: str
    problem: str
    file_format: str
    nb_best_param_combi: int
    mix_graph_type: bool
    sig_fig: int
    supervised: bool


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------

def load_toml(path: str | Path, struct_type: type[msgspec.Struct]) -> msgspec.Struct:
    """Read a TOML file and decode it into a msgspec Struct."""
    try:
        raw = Path(path).read_bytes()
        data = tomllib.loads(raw.decode())
        return msgspec.convert(data, struct_type)
    except(msgspec.ValidationError, msgspec.DecodeError) as e:
        print(f"Error in config file '{path}':\n{e}")
        sys.exit(1) # Stop the program immediately

# ---------------------------------------------------------------------------
# Struct -> legacy dict converters
# ---------------------------------------------------------------------------

def _resolve_path(p: Optional[str], default: Path) -> str:
    """Resolve a path string: relative paths are resolved against PROJECT_ROOT."""
    if not p:
        return str(default)
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def run_config_to_dicts(cfg: RunSolverConfig | RunHPTuningConfig) -> dict:
    """Convert a RunConfig into the legacy kwargs dict for run_hyperparameter_tuning_parallel()."""

    if cfg.synthetic_data:
        params_graph = {
            "dataset_size": cfg.graph.dataset_size,
            "num_nodes": cfg.graph.num_nodes,
            "p": cfg.graph.p,
        }
        params_data = None
        
    else:
        params_data = {
            "path": str(DATA_DIR / "numpy_data"),
            "dataset_name": cfg.data.dataset_name,
            "dataset_split": cfg.data.dataset_split,
            "sample": cfg.data.sample,
        }
        params_graph = None


    params_ga = {
        "pop_size": cfg.ga.pop_size,
        "nb_offsprings": cfg.ga.nb_offsprings,
        "init_type": cfg.ga.init_type,
        "evolution_mode": cfg.ga.evolution_mode,
        "crossover_rate": cfg.ga.crossover_rate,
        "mutation_type": cfg.ga.mutation_type,
    }
    if cfg.ga.lamarckian_probability is not None:
        params_ga["lamarckian_probability"] = cfg.ga.lamarckian_probability
    
    params_outputs = {
        "output_path": _resolve_path(cfg.output.output_path, RESULTS_DIR),
        "visualize": [cfg.output.visualize, cfg.output.visualize_reps],
        "plot_display": cfg.output.plot_display,
    }
    
    base_kwargs = {
        "params_outputs": params_outputs,
        "params_ga": params_ga,
        "synthetic_data": cfg.synthetic_data,
        "params_graph": params_graph,
        "params_data": params_data,
        "problem_name": cfg.problem_name,
    }

    if isinstance(cfg, RunHPTuningConfig):
        base_kwargs["nb_reps_per_graph"] = cfg.nb_reps_per_graph
        return base_kwargs
    
    return base_kwargs



def download_config_to_args(cfg: DownloadConfig) -> dict:
    """Convert a DownloadConfig into kwargs for run_benchmark_conversion_pipeline()."""
    return {
        "path_data_benchmark": str(DATA_DIR / "benchmark_data"),
        "root_numpy_benchmark": str(DATA_DIR / "numpy_data"),
        "dataset_name": cfg.dataset_name,
        "delete_raw": cfg.delete_raw,
        "splits": cfg.splits,
    }


def best_params_config_to_dicts(cfg: RunBestParamsConfig) -> dict:
    """Convert a RunBestParamsConfig into kwargs for run_best_params()."""
    return {
        "results_path": _resolve_path(cfg.results_path, RESULTS_DIR),
        "problem_name": cfg.problem_name,
        "aggregation_method": cfg.aggregation_method,
        "nb_reps_per_graph": cfg.nb_reps_per_graph,
        "supervised": cfg.supervised,
        "params_data": {
            "path": str(DATA_DIR / "numpy_data"),
            "dataset_name": cfg.data.dataset_name,
            "dataset_split": cfg.data.dataset_split,
            "sample": cfg.data.sample,
        },
        "params_outputs": {
            "output_path": _resolve_path(cfg.output.output_path, RESULTS_DIR),
            "visualize": [cfg.output.visualize, cfg.output.visualize_reps],
            "plot_display": cfg.output.plot_display,
        },
        "resume_path": _resolve_path(cfg.resume_path, RESULTS_DIR) if cfg.resume_path else None,
        "combis_path": _resolve_path(cfg.combis_path, RESULTS_DIR) if cfg.combis_path else None,
        "track_metrics": cfg.track_metrics,
    }


def analyze_config_to_dict(cfg: AnalyzeConfig) -> dict:
    """Convert an AnalyzeConfig into the legacy config dict for run_analysis()."""
    return {
        "path_result": _resolve_path(cfg.path_result, RESULTS_DIR),
        "file_format": cfg.file_format,
        "nb_best_param_combi": cfg.nb_best_param_combi,
        "mix_graph_type": cfg.mix_graph_type,
        "sig_fig": cfg.sig_fig,
        "supervised": cfg.supervised,
    }
