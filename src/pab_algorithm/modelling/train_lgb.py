import numpy.typing as npt
import pandas as pd
from lightgbm import Booster, Dataset, train
from numpy import float64
from pab_algorithm.config import (
    BoosterParams,
    TrainingSettings,
    boost_params,
    train_settings,
)
from pab_algorithm.etl import load_data
from pab_algorithm.etl.training_dataset_pipeline import run_dataset_pipeline
from sklearn.model_selection import train_test_split


def create_lgb_datasets(settings: TrainingSettings) -> tuple[Dataset, Dataset]:
    dataset: pd.DataFrame = run_dataset_pipeline(
        df_results=load_data.get_results_df(),
        df_iso=load_data.get_iso_df(),
        df_elo=load_data.get_elo_df(),
        min_year=settings.min_year,
        max_goals=settings.max_goals,
    )

    X: npt.NDArray[float64] = dataset[["elo_diff", "is_friendly"]].to_numpy(
        dtype=float64
    )
    y: npt.NDArray[float64] = (dataset["scored"] - dataset["power"]).to_numpy(
        dtype=float64
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=settings.train_size,
        random_state=settings.random_state,
    )

    train_data: Dataset = Dataset(X_train, label=y_train)
    val_data: Dataset = Dataset(X_val, label=y_val, reference=train_data)

    return train_data, val_data


def run_training(
    train_data: Dataset,
    val_data: Dataset,
    params: BoosterParams,
) -> Booster:
    return train(
        params=params.model_dump(),
        train_set=train_data,
        num_boost_round=params.num_boost_rounds,
        valid_sets=[train_data, val_data],
        valid_names=["train", "validation"],
    )


def train_lgb_from_default() -> Booster:
    train_data, val_data = create_lgb_datasets(settings=train_settings)
    model = run_training(train_data=train_data, val_data=val_data, params=boost_params)

    if train_settings.save_model:
        model.save_model(
            filename=train_settings.output_model_store_dir
            / boost_params.output_model_name,
            num_iteration=model.best_iteration,
        )

    return model
