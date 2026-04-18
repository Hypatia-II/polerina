"""Wrapper entry point: python -m polerina.scripts.run_hp_tuning"""

from polerina.config import RunHPTuningConfig, load_toml, run_config_to_dicts
from polerina.runners.hp_tuning import run_hyperparameter_tuning

DEFAULT_CONFIG = "config/run_hp_tuning.toml"


def main():
    cfg = load_toml(DEFAULT_CONFIG, RunHPTuningConfig)
    kwargs = run_config_to_dicts(cfg)
    run_hyperparameter_tuning(**kwargs)



if __name__ == "__main__":
    main()
