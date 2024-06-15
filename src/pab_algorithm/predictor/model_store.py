from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from lightgbm.basic import Booster as LgbBooster
from numpy import ndarray
from sklearn.linear_model._base import LinearRegression as SklearnRegressor

from pab_algorithm.config import boost_params, linear_params, train_settings
from pab_algorithm.data_types import GbmFeatures, LinearFeatures, ScoringPowers
from pab_algorithm.modelling.train_lgb import train_lgb_from_default
from pab_algorithm.modelling.train_linear import train_linear_from_default
from pab_algorithm.predictor.team import Team


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

    @classmethod
    def load_model_store(cls) -> ModelStore:
        linear_model: SklearnRegressor
        gbm_model: LgbBooster

        linear_path: Path = (
            train_settings.output_model_store_dir / linear_params.output_model_name
        )
        if not os.path.isfile(path=linear_path):
            linear_model = train_linear_from_default()
        else:
            with linear_path.open("rb") as f:
                linear_model = pickle.load(f)

        gbm_path: Path = (
            train_settings.output_model_store_dir / boost_params.output_model_name
        )
        if not os.path.isfile(path=gbm_path):
            gbm_model = train_lgb_from_default()
        else:
            gbm_model = LgbBooster(model_file=gbm_path)

        return ModelStore(
            linear=linear_model,
            booster=gbm_model,
        )
