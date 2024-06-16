from dataclasses import dataclass


@dataclass
class Team:
    code: str
    name: str
    power: float
    elo: int
