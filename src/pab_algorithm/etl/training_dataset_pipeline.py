import pandas as pd

from pab_algorithm.etl.shared import (
    add_scoring_power,
    combine_results_and_iso,
    remove_extras_and_add_year,
    remove_iso_columns,
    set_elos,
)


def create_dataset(df_powers: pd.DataFrame) -> pd.DataFrame:
    df = df_powers[["power", "is_friendly", "scored"]]
    df["elo_diff"] = df_powers.apply(lambda row: row["home"] - row["away"], axis=1)

    return df


def run_dataset_pipeline(
    df_results: pd.DataFrame,
    df_iso: pd.DataFrame,
    df_elo: pd.DataFrame,
    min_year: int | str | None = None,
    max_goals: int | None = None,
) -> pd.DataFrame:
    df_filtered_results = remove_extras_and_add_year(df_results)
    df_filtered_iso = remove_iso_columns(df_iso)

    df_combined = combine_results_and_iso(df_filtered_results, df_filtered_iso)
    del df_filtered_results, df_filtered_iso

    df_set_elos = set_elos(df_combined, df_elo, min_year=min_year, max_goals=max_goals)
    del df_combined

    df_powers = add_scoring_power(df_set_elos)
    del df_set_elos

    return create_dataset(df_powers)
