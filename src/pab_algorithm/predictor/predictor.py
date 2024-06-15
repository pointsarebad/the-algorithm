import numpy as np
from scipy.stats import mode

from pab_algorithm.data_types import ModelType
from pab_algorithm.predictor.model_store import AbstractModelStore, ModelStoreFactory
from pab_algorithm.predictor.team import Team


class ScorePredictor:
    def __init__(
        self,
        model_type: ModelType | None = None,
        default_samples: int = 5,
    ) -> None:
        self.default_samples = default_samples

        self.model: AbstractModelStore = ModelStoreFactory.load_model_store(
            model_type=model_type
        )

    @staticmethod
    def sample(lambda_: float, n_samples: int) -> int:
        samples = np.random.poisson(lambda_, n_samples)
        return mode(samples, axis=None).mode

    def predict(
        self,
        home: Team,
        away: Team,
        n_samples: int | None = None,
    ) -> tuple[int, int]:
        if n_samples is None or n_samples <= 0:
            n_samples = self.default_samples

        home_power, away_power = self.model.get_powers(home=home, away=away)

        return (
            ScorePredictor.sample(home_power, n_samples=n_samples),
            ScorePredictor.sample(away_power, n_samples=n_samples),
        )

    def display_score(
        self,
        home: Team,
        away: Team,
        n_samples: int | None = None,
    ) -> str:
        goals = self.predict(
            home=home,
            away=away,
            n_samples=n_samples,
        )
        return f"{home.name} {goals[0]} - {goals[1]} {away.name}"
