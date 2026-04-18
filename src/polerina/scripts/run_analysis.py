"""Wrapper entry point: python -m polerina.scripts.run_analysis"""

from polerina.analysis.aggregate_results_utils import run_analysis
from polerina.config import AnalyzeConfig, analyze_config_to_dict, load_toml

DEFAULT_CONFIG = "config/analyze.toml"


def main():
    cfg = load_toml(DEFAULT_CONFIG, AnalyzeConfig)
    config = analyze_config_to_dict(cfg)
    run_analysis(config)


if __name__ == "__main__":
    main()
