import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ["POLERINA_DATA_DIR"]) if "POLERINA_DATA_DIR" in os.environ else PROJECT_ROOT / "data"
RESULTS_DIR = Path(os.environ["POLERINA_RESULTS_DIR"]) if "POLERINA_RESULTS_DIR" in os.environ else PROJECT_ROOT / "results"
