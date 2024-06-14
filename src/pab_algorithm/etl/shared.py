import pandas as pd


def remove_extras_and_add_year(df_results: pd.DataFrame) -> pd.DataFrame:
    """Gets the relevant columns from the raw results table, filters the date and
    creates an is_friendly flag."""

    df = df_results[
        ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
    ].dropna(axis=0)

    df["year"] = df["date"].apply(lambda x: x[:4])
    df["is_friendly"] = df["tournament"].apply(
        lambda x: True if x.lower() == "friendly" else False
    )

    return df.drop(columns=["date", "tournament"])


def remove_iso_columns(df_iso: pd.DataFrame) -> pd.DataFrame:
    """Gets the name of countries and 2-digit ISO code from raw ISO data table."""

    return df_iso[["English short name lower case", "Alpha-2 code"]].rename(
        columns={
            "English short name lower case": "country",
            "Alpha-2 code": "code",
        }
    )


def combine_results_and_iso(
    df_results: pd.DataFrame, df_iso: pd.DataFrame
) -> pd.DataFrame:
    """Joins the reduced results and iso tables to map the ISO codes onto the home
    and away teams."""

    return (
        pd.merge(
            left=pd.merge(
                left=df_results,
                right=df_iso,
                left_on="home_team",
                right_on="country",
                how="inner",
            )
            .drop(columns=["country", "home_team"])
            .rename(columns={"code": "home"}),
            right=df_iso,
            left_on="away_team",
            right_on="country",
            how="inner",
        )
        .drop(columns=["country", "away_team"])
        .rename(columns={"code": "away"})
    )


def add_scoring_power(df_set_elos: pd.DataFrame) -> pd.DataFrame:
    """Adds scoring power to results and iso table, with the elo ratings attached."""

    # Scoring power is measured to be the mean number of goals scored by each team for all matches from the starting year defined in the pipeline
    df_power = (
        df_set_elos[["code", "scored"]]
        .groupby("code")
        .mean("scored")
        .reset_index()
        .rename(columns={"scored": "power"})
    )

    return pd.merge(
        left=df_set_elos,
        right=df_power,
        how="inner",
        left_on="code",
        right_on="code",
    )
