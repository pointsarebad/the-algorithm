import pandas as pd

from pab_algorithm.etl.shared import (
    combine_results_and_iso,
    remove_extras_and_add_year,
    remove_iso_columns,
    set_elos,
)


def run_country_lookup_pipeline(
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

    df_powers = (
        df_set_elos[["code", "scored"]]
        .groupby("code")
        .mean()
        .reset_index()
        .rename(columns={"scored": "power"})
    )
    del df_set_elos

    df_powers["elo"] = [df_elo.loc[country, "2024"] for country in df_powers["code"]]

    # Adding these back in here as to not disrupt the flow of the dataset pipeline
    country_lookup = df_iso.set_index("Alpha-2 code")["English short name lower case"]
    df_powers["name"] = [country_lookup.loc[code] for code in df_powers["code"]]

    return df_powers
