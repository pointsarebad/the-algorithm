import requests
from collections import defaultdict
import pandas as pd

def create_url(year: int = 1910):
    return f"https://eloratings.net/{year}.tsv"

def get_elos_for_year(year: int = 1910):
    url = create_url(year=year)
    resp = requests.get(url=url)

    resp.raise_for_status()
    print(f"Got data for year {year}")

    raw_text = resp.text
    rows =  [row.split("\t") for row in raw_text.split("\n")]
    return {row[2]: row[3] for row in rows}

def get_full_elo(start=1910, end=2024):
    elos: defaultdict[int, dict[str, int]] = defaultdict(dict)

    for yr in range(start, end + 1):
        elo = get_elos_for_year(year=yr)
        for team, rating in elo.items():
            elos[yr].update({team: rating})
            
    return elos

all_elos = get_full_elo()
df = pd.DataFrame(data=all_elos)

df.to_csv("test.csv")