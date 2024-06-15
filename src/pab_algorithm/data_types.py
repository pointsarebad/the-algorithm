from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

ModelType: TypeAlias = Literal["linear", "gbm"]

LinearDatasets: TypeAlias = tuple[npt.NDArray[np.float64], ...]

NumLinearFeatures: TypeAlias = Literal[1]
LinearFeatures: TypeAlias = np.ndarray[NumLinearFeatures, np.dtype[np.float64]]

NumGbmFeatures: TypeAlias = Literal[2]
GbmFeatures: TypeAlias = np.ndarray[NumGbmFeatures, np.dtype[np.float64]]

ScoringPowers: TypeAlias = np.ndarray[Literal[2], np.dtype[np.float64]]
