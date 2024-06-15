import pickle

import numpy.typing as npt
import pandas as pd
from numpy import float64
from pab_algorithm.config import TrainingSettings, linear_params, train_settings
from pab_algorithm.data_types import LinearDatasets
from pab_algorithm.etl import load_data
from pab_algorithm.etl.training_dataset_pipeline import run_dataset_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def create_linear_datasets(settings: TrainingSettings) -> LinearDatasets:
    dataset: pd.DataFrame = run_dataset_pipeline(
        df_results=load_data.get_results_df(),
        df_iso=load_data.get_iso_df(),
        df_elo=load_data.get_elo_df(),
        min_year=settings.min_year,
        max_goals=settings.max_goals,
    )

    X: npt.NDArray[float64] = dataset[["elo_diff"]].to_numpy(dtype=float64)
    y: npt.NDArray[float64] = (dataset["scored"] - dataset["power"]).to_numpy(
        dtype=float64
    )

    return train_test_split(
        X,
        y,
        train_size=settings.train_size,
        random_state=settings.random_state,
    )


def run_training(
    X_train: npt.NDArray[float64],
    X_val: npt.NDArray[float64],
    y_train: npt.NDArray[float64],
    y_val: npt.NDArray[float64],
) -> LinearRegression:
    linear_model: LinearRegression = LinearRegression()
    linear_model.fit(X_train, y_train)
    return linear_model


def train_linear_from_default() -> LinearRegression:
    X_train, X_val, y_train, y_val = create_linear_datasets(settings=train_settings)
    model = run_training(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    if train_settings.save_model:
        with (
            train_settings.output_model_store_dir / linear_params.output_model_name
        ).open("wb") as f:
            pickle.dump(model, f)

    return model
