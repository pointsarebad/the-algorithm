import os
from pathlib import Path

DATA_DIR: Path
if (file_path := os.getenv("DATA_DIR")) is None:
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
else:
    DATA_DIR = Path(file_path)

ELO_FILE_PATH: Path = DATA_DIR / "elo.csv"
ISO_FILE_PATH: Path = DATA_DIR / "iso.csv"
RESULTS_FILE_PATH: Path = DATA_DIR / "results.csv"