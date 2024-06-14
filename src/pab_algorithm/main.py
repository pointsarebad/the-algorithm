from pab_algorithm.etl.training_dataset_pipeline import create_dataset
import pandas as pd
from pab_algorithm.config import ISO_FILE_PATH

print(
    pd.read_csv(ISO_FILE_PATH).iloc[0]
)