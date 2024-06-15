import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_DIR: Path
if (file_path := os.getenv("DATA_DIR", "data")) is None:
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
else:
    DATA_DIR = Path(file_path)

del file_path

ELO_FILE_PATH: Path = DATA_DIR / "elo.csv"
ISO_FILE_PATH: Path = DATA_DIR / "iso.csv"
RESULTS_FILE_PATH: Path = DATA_DIR / "results.csv"


class LookupSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="country_lookup_")

    min_year: int | None = None
    max_goals: int | None = None


class TrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="model_training_")

    random_state: int = 444
    train_size: float = 0.85

    min_year: int | None = None
    max_goals: int | None = None

    save_model: bool = True
    output_model_store_dir: Path = Path(__file__).parent / "model_store"


class BoosterParams(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="booster_params_",
    )

    objective: str = "regression"
    metric: str = "mse"
    boosting_type: str = "gbdt"
    num_leaves: int = 31
    learning_rate: float = 0.01
    feature_fraction: float = 0.9

    num_boost_rounds: int = Field(default=1000, exclude=True)

    output_model_name: str = Field(default="gbm_model.txt", exclude=True)


class LinearParams(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="linear_params_",
    )

    output_model_name: str = Field(default="linear_model.pkl", exclude=True)


train_settings = TrainingSettings()
lookup_settings = LookupSettings()

boost_params = BoosterParams()
linear_params = LinearParams()
