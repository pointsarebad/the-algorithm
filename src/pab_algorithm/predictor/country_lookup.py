from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from pab_algorithm.config import lookup_settings
from pab_algorithm.etl import load_data
from pab_algorithm.etl.country_lookup_pipeline import run_country_lookup_pipeline
from pab_algorithm.predictor.team import Team


class CountryLookup:
    def __init__(self, countries: Iterable[Team]) -> None:
        self._countries: dict[str, Team] = {c.code: c for c in countries}

        self._code_lookup: dict[str, str] = {
            c.name.lower(): c.code for c in self._countries.values()
        }

    def __getitem__(self, key: Any) -> Team:
        if not isinstance(key, str):
            raise IndexError(
                "Must search by country by country name or 2-digit ISO code"
            )

        if len(key) == 2:
            return self._countries[key]
        return self._countries[self._code_lookup[key.lower()]]

    def __iter__(self) -> Iterable[Team]:
        yield from self._countries.values()

    @classmethod
    def load_default_lookup(cls) -> CountryLookup:
        dataset: pd.DataFrame = run_country_lookup_pipeline(
            df_results=load_data.get_results_df(),
            df_iso=load_data.get_iso_df(),
            df_elo=load_data.get_elo_df(),
            min_year=lookup_settings.min_year,
            max_goals=lookup_settings.max_goals,
        )

        return cls(
            countries=(
                Team(
                    code=row.loc["code"],
                    name=row.loc["name"],
                    elo=row.loc["elo"],
                    power=row.loc["power"],
                )
                for _, row in dataset.iterrows()
            )
        )
