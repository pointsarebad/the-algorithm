import pandas as pd

from pab_algorithm.etl.shared import (
    add_scoring_power,
    combine_results_and_iso,
    remove_extras_and_add_year,
    remove_iso_columns,
)


def set_elos(
    df_combined: pd.DataFrame,
    df_elo: pd.DataFrame,
    min_year: int | str | None = None,
    max_goals: int | None = None,
) -> pd.DataFrame:
    if min_year is None:
        years = set(df_elo.columns)
    else:
        years = {c for c in df_elo.columns if c >= str(min_year)}

    countries = set(df_elo.index)

    df_filtered = df_combined[
        df_combined["year"].isin(years)
        & df_combined["home"].isin(countries)
        & df_combined["away"].isin(countries)
    ]

    df_filtered.loc[:, "home_elo"] = df_filtered.apply(
        lambda row: df_elo.loc[row["home"], row["year"]], axis=1
    )
    df_filtered.loc[:, "away_elo"] = df_filtered.apply(
        lambda row: df_elo.loc[row["away"], row["year"]], axis=1
    )

    df_filtered = df_filtered.dropna()

    df_home = pd.DataFrame({
        "code": df_filtered["home"],
        "home": df_filtered["home_elo"].astype(int),
        "away": df_filtered["away_elo"].astype(int),
        "is_friendly": df_filtered["is_friendly"].astype(bool),
        "scored": df_filtered["home_score"].astype(int),
    })

    df_away = pd.DataFrame({
        "code": df_filtered["away"],
        "home": df_filtered["away_elo"].astype(int),
        "away": df_filtered["home_elo"].astype(int),
        "is_friendly": df_filtered["is_friendly"].astype(bool),
        "scored": df_filtered["away_score"].astype(int),
    })

    df_set_elos = pd.concat([df_home, df_away], ignore_index=True)

    if max_goals and max_goals > 0:
        df_set_elos = df_set_elos[df_set_elos["scored"] <= max_goals]

    return df_set_elos


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
