from typing import Literal, TypeAlias

import numpy as np

ModelType: TypeAlias = Literal["linear", "gbm"]

LinearFeatures: TypeAlias = np.ndarray[np.int_]
GbmFeatures: TypeAlias = np.ndarray[np.int_, np.bool_]

ScoringPowers: TypeAlias = tuple[float, float]
