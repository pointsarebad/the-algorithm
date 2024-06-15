from pab_algorithm.predictor.country_lookup import CountryLookup
from pab_algorithm.predictor.predictor import ScorePredictor


predictor = ScorePredictor(
    model_type="gbm",
    default_samples=1000,
)

lookup = CountryLookup.load_default_lookup()

home = lookup["germany"]
away = lookup["england"]

print(predictor.display_score(home, away))
