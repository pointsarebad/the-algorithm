import numpy as np
from lightgbm.basic import Booster as LgbBooster
from scipy.stats import mode
from sklearn.linear_model._base import LinearRegression as SklearnRegressor

from pab_algorithm.data_types import ModelType
from pab_algorithm.predictor.model_store import ModelStore
from pab_algorithm.predictor.team import Team


class ScorePredictor:
    def __init__(
        self,
        default_samples: int = 5,
    ) -> None:

        self.default_samples = default_samples

        self.models: ModelStore = ModelStore.load_model_store()

    @staticmethod
    def sample(lambda_: float, n_samples: int) -> int:
        samples = np.random.poisson(lambda_, n_samples)
        return mode(samples, axis=None).mode

    def predict(
        self,
        home: Team,
        away: Team,
        model_type: ModelType | None = None,
        n_samples: int | None = None,
    ) -> tuple[int, int]:
        if n_samples is None or n_samples <= 0:
            n_samples = self.default_samples

        match model_type:
            case "linear":
                home_power, away_power = self.models.get_powers_linear(
                    home=home, away=away
                )
            case "gbm":
                home_power, away_power = self.models.get_powers_gbm(
                    home=home, away=away
                )
            case _:
                home_power, away_power = home.power, away.power

        return (
            ScorePredictor.sample(home_power, n_samples=n_samples),
            ScorePredictor.sample(away_power, n_samples=n_samples),
        )

    def display_score(
        self,
        home: Team,
        away: Team,
        model_type: ModelType | None = None,
        n_samples: int | None = None,
        show_power: bool = False,
    ) -> str:
        goals = self.predict(
            home=home,
            away=away,
            model_type=model_type,
            n_samples=n_samples,
            show_power=show_power,
        )
        return f"{home.name} {goals[0]} - {goals[1]} {away.name}"
