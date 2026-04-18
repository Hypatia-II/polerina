"""Wrapper entry point: python -m polerina.scripts.download_data"""

from polerina.config import DownloadConfig, download_config_to_args, load_toml
from polerina.data_handler.data_prep import run_benchmark_conversion_pipeline

DEFAULT_CONFIG = "config/download.toml"


def main():
    cfg = load_toml(DEFAULT_CONFIG, DownloadConfig)
    kwargs = download_config_to_args(cfg)
    run_benchmark_conversion_pipeline(**kwargs)


if __name__ == "__main__":
    main()
