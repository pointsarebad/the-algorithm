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
        .mean()
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

    df_home = pd.DataFrame(
        {
            "code": df_filtered["home"],
            "home": df_filtered["home_elo"].astype(int),
            "away": df_filtered["away_elo"].astype(int),
            "is_friendly": df_filtered["is_friendly"].astype(bool),
            "scored": df_filtered["home_score"].astype(int),
        }
    )

    df_away = pd.DataFrame(
        {
            "code": df_filtered["away"],
            "home": df_filtered["away_elo"].astype(int),
            "away": df_filtered["home_elo"].astype(int),
            "is_friendly": df_filtered["is_friendly"].astype(bool),
            "scored": df_filtered["away_score"].astype(int),
        }
    )

    df_set_elos = pd.concat([df_home, df_away], ignore_index=True)

    if max_goals and max_goals > 0:
        df_set_elos = df_set_elos[df_set_elos["scored"] <= max_goals]

    return df_set_elos
