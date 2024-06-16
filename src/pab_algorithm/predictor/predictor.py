import numpy as np
import numpy.typing as npt
from scipy.stats import mode

from pab_algorithm.data_types import ModelType, ScoringPowers
from pab_algorithm.predictor.country_lookup import CountryLookup
from pab_algorithm.predictor.model_store import AbstractModelStore, ModelStoreFactory
from pab_algorithm.predictor.team import Team


class ScorePredictor:
    def __init__(
        self,
        model_type: ModelType | None = None,
        default_samples: int = 5,
        lookup: CountryLookup | None = None,
    ) -> None:
        self.model: AbstractModelStore = ModelStoreFactory.load_model_store(
            model_type=model_type
        )

        self.default_samples = default_samples

        self.lookup = (
            lookup if lookup is not None else CountryLookup.load_default_lookup()
        )

    @staticmethod
    def get_samples(lam: ScoringPowers, n_samples: int) -> npt.NDArray[np.int32]:
        return np.random.poisson(lam=lam, size=(n_samples, 2)).astype(np.int32)

    @staticmethod
    def sample(lam: ScoringPowers, n_samples: int) -> tuple[int, int]:
        samples = ScorePredictor.get_samples(lam=lam, n_samples=n_samples)
        return tuple(mode(samples, axis=0).mode)

    def get_teams(self, home: str, away: str) -> tuple[Team, Team]:
        return self.lookup[home], self.lookup[away]

    def predict(
        self, home: str, away: str, n_samples: int | None = None
    ) -> tuple[int, int]:
        home_team, away_team = self.get_teams(home=home, away=away)

        if n_samples is None or n_samples <= 0:
            n_samples = self.default_samples

        scoring_powers = self.model.get_powers(home=home_team, away=away_team)

        return ScorePredictor.sample(lam=scoring_powers, n_samples=n_samples)

    def get_win_probs(
        self,
        home: str,
        away: str,
        n_samples: int | None = None,
    ) -> dict[str, float]:
        home_team, away_team = self.get_teams(home=home, away=away)

        if n_samples is None or n_samples <= 0:
            n_samples = self.default_samples

        scoring_powers = self.model.get_powers(home=home_team, away=away_team)
        samples = ScorePredictor.get_samples(lam=scoring_powers, n_samples=n_samples)

        return {
            home_team.name: (samples[:, 0] > samples[:, 1]).sum() / n_samples,
            away_team.name: (samples[:, 0] < samples[:, 1]).sum() / n_samples,
        }

    def display_score(
        self,
        home: str,
        away: str,
        n_samples: int | None = None,
    ) -> str:
        home_team, away_team = self.get_teams(home=home, away=away)
        goals = self.predict(
            home=home,
            away=away,
            n_samples=n_samples,
        )
        return f"{home_team.name} {goals[0]} - {goals[1]} {away_team.name}"
