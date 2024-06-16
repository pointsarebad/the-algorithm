from pab_algorithm.predictor.country_lookup import CountryLookup
from pab_algorithm.predictor.predictor import ScorePredictor

predictor = ScorePredictor(model_type="linear", default_samples=1_000, use_cache=True)

lookup = CountryLookup.load_default_lookup()

home = "italy"
away = "spain"

print(predictor.get_win_probs(home, away))
print(predictor.display_score(away, home))
