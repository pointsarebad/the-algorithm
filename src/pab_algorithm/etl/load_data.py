import pandas as pd

from pab_algorithm.config import ELO_FILE_PATH, ISO_FILE_PATH, RESULTS_FILE_PATH


def get_results_df() -> pd.DataFrame:
    return pd.read_csv(RESULTS_FILE_PATH)


def get_elo_df() -> pd.DataFrame:
    return pd.read_csv(ELO_FILE_PATH, index_col=0)


def get_iso_df() -> pd.DataFrame:
    return pd.read_csv(ISO_FILE_PATH)
