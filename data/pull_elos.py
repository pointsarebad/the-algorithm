import os
from collections import defaultdict
from logging import Logger, getLogger
from pathlib import Path

import pandas as pd
import requests

log: Logger = getLogger(__name__)


def create_url(year: int = 1910) -> str:
    return f"https://eloratings.net/{year}.tsv"


def get_elos_for_year(year: int = 1910) -> dict[str, str]:
    url = create_url(year=year)
    resp = requests.get(url=url)

    resp.raise_for_status()
    log.info("Got data for %d", year)

    raw_text: str = resp.text
    rows: list[list[str]] =  [row.split("\t") for row in raw_text.split("\n")]
    return {row[2]: row[3] for row in rows}


def get_full_elo(start: int = 1910, end: int = 2024) -> dict[str, dict[str, str]]:
    elos: defaultdict[int, dict[str, int]] = defaultdict(dict)

    for yr in range(start, end + 1):
        elo = get_elos_for_year(year=yr)
        for team, rating in elo.items():
            elos[yr].update({team: rating})
            
    return elos


def get_all_elos(file_path: Path) -> None:
    all_elos = get_full_elo()
    df: pd.DataFrame = pd.DataFrame(data=all_elos)

    df.to_csv(file_path)


if __name__ == "__main__":
    file_path: str | None = os.getenv("ELO_OUTPUT_PATH")
    
    output_path: Path = Path(file_path) if file_path else Path(__file__).parent / "elo.csv"
    get_all_elos(file_path=output_path)