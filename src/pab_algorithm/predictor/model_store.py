from __future__ import annotations

import os
import pickle
from abc import ABC
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from lightgbm.basic import Booster as LgbBooster
from numpy import ndarray
from sklearn.linear_model._base import LinearRegression as SklearnRegressor

from pab_algorithm.config import boost_params, linear_params, train_settings
from pab_algorithm.data_types import (
    GbmFeatures,
    LinearFeatures,
    ModelType,
    ScoringPowers,
)
from pab_algorithm.modelling.train_lgb import train_lgb_from_default
from pab_algorithm.modelling.train_linear import train_linear_from_default
from pab_algorithm.predictor.team import Team


class AbstractModelStore(ABC):
    def __init__(self, model) -> None:
        self.model = model

    def _get_inputs(self, home: Team, away: Team) -> npt.NDArray[np.float64]:
        return NotImplemented

    def _adjust_powers(
        self, home: Team, away: Team, powers: ScoringPowers
    ) -> ScoringPowers:
        return np.array((
            max(home.power + powers[0], 0),
            max(away.power + powers[1], 0),
        ))

    def get_powers(self, home: Team, away: Team) -> ScoringPowers:
        return NotImplemented

    @classmethod
    def load_model_store(cls) -> AbstractModelStore:
        return NotImplemented


class ModelStoreFactory:
    @staticmethod
    def load_model_store(model_type: ModelType | None = None) -> AbstractModelStore:
        match model_type:
            case "linear":
                return LinearModelStore.load_model_store()
            case "gbm":
                return GbmModelStore.load_model_store()

        return BasicModelStore.load_model_store()


class LinearModelStore(AbstractModelStore):
    def __init__(self, model: SklearnRegressor) -> None:
        self.model = model

    def _get_inputs(self, home: Team, away: Team) -> LinearFeatures:
        elo_diff: int = home.elo - away.elo
        return np.array(((elo_diff,), (-elo_diff,)), dtype=np.float64)

    def get_powers(self, home: Team, away: Team) -> ScoringPowers:
        inputs = self._get_inputs(home=home, away=away)
        powers = cast(ScoringPowers, self.model.predict(inputs))
        return self._adjust_powers(home=home, away=away, powers=powers)

    @classmethod
    def load_model_store(cls) -> LinearModelStore:
        linear_path: Path = (
            train_settings.output_model_store_dir / linear_params.output_model_name
        )

        linear_model: SklearnRegressor
        if not os.path.isfile(path=linear_path):
            linear_model = train_linear_from_default()
        else:
            with linear_path.open("rb") as f:
                linear_model = pickle.load(f)

        return cls(model=linear_model)


class GbmModelStore(AbstractModelStore):
    def __init__(self, model: LgbBooster) -> None:
        self.model = model

    def _get_inputs(self, home: Team, away: Team) -> GbmFeatures:
        elo_diff: int = home.elo - away.elo
        return np.array(((elo_diff, False), (-elo_diff, False)), dtype=np.float64)

    def get_powers(self, home: Team, away: Team) -> ScoringPowers:
        inputs = self._get_inputs(home=home, away=away)
        powers = cast(ScoringPowers, self.model.predict(inputs))
        return self._adjust_powers(home=home, away=away, powers=powers)

    @classmethod
    def load_model_store(cls) -> GbmModelStore:
        gbm_path: Path = (
            train_settings.output_model_store_dir / boost_params.output_model_name
        )

        gbm_model: LgbBooster
        if not os.path.isfile(path=gbm_path):
            gbm_model = train_lgb_from_default()
        else:
            gbm_model = LgbBooster(model_file=gbm_path)

        return cls(model=gbm_model)


class BasicModelStore(AbstractModelStore):
    def __init__(self, model) -> None:
        self.model = model

        self.powers: ScoringPowers = np.array((0, 0), dtype=np.float64)

    def get_powers(self, home: Team, away: Team) -> ScoringPowers:
        return self._adjust_powers(home=home, away=away, powers=self.powers)

    @classmethod
    def load_model_store(cls) -> BasicModelStore:
        return cls(model=None)
