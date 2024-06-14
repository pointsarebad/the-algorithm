from typing import Literal, TypeAlias

import numpy as np
from lightgbm.basic import Booster as LgbBooster
from numpy import ndarray
from pab_algorithm.models.team import Team
from scipy.stats import mode
from sklearn.linear_model._base import LinearRegression as SklearnRegressor

ModelType: TypeAlias = Literal["linear", "gbm"]
LinearFeatures: TypeAlias = ndarray[np.int_]
GbmFeatures: TypeAlias = ndarray[np.int_, np.bool_]
ScoringPowers: TypeAlias = tuple[float, float]


class ModelStore:
    def __init__(
        self,
        linear: SklearnRegressor | None = None,
        booster: LgbBooster | None = None,
    ) -> None:
        self.linear = linear
        self.booster = booster

    def _get_linear_inputs(self, home: Team, away: Team) -> LinearFeatures:
        elo_diff: int = home.elo - away.elo
        return np.array(((elo_diff,), (-elo_diff,)))

    def _get_gbm_inputs(self, home: Team, away: Team) -> GbmFeatures:
        elo_diff: int = home.elo - away.elo
        return np.array(((elo_diff, False), (-elo_diff, False)))

    def _adjust_powers(
        self, home: Team, away: Team, powers: ndarray[np.float_]
    ) -> ScoringPowers:
        return max(home.power + powers[0], 0), max(away.power + powers[1], 0)

    def get_powers_linear(self, home: Team, away: Team) -> ScoringPowers:
        if self.linear is None:
            raise NotImplementedError

        inputs = self._get_linear_inputs(home=home, away=away)
        powers: ndarray[np.float_] = self.linear.predict(inputs)
        return self._adjust_powers(home=home, away=away, powers=powers)

    def get_powers_gbm(self, home: Team, away: Team) -> ScoringPowers:
        if self.booster is None:
            raise NotImplementedError

        inputs = self._get_gbm_inputs(home=home, away=away)
        powers: ndarray[np.float_] = self.booster.predict(inputs)
        return self._adjust_powers(home=home, away=away, powers=powers)


class ScorePredictor:
    def __init__(
        self,
        linear: SklearnRegressor | None = None,
        booster: LgbBooster | None = None,
        default_samples: int = 5,
    ) -> None:
        if linear is None and booster is None:
            raise ValueError("At least one of the regressor and booster must be set")

        self.default_samples = default_samples

        self.models: ModelStore = ModelStore(
            linear=linear,
            booster=booster,
        )

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
