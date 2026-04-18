"""Wrapper entry point: python -m polerina.scripts.run_solver"""

from polerina.config import RunSolverConfig, load_toml, run_config_to_dicts
from polerina.runners.solver import run_solver

DEFAULT_CONFIG = "config/run_solver.toml"


def main():
    cfg = load_toml(DEFAULT_CONFIG, RunSolverConfig)
    kwargs = run_config_to_dicts(cfg)
    run_solver(**kwargs)


if __name__ == "__main__":
    main()
