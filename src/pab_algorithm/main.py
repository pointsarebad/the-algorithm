from pab_algorithm.predictor.model_store import ModelStore
from pab_algorithm.predictor.team import Team

home = Team(elo=1721.0, power=1.744186046511628, name="Poland", code="PO")
away = Team(elo=1968.0, power=2.227272727272727, name="Netherlands", code="NL")

store = ModelStore.load_model_store()

print(store.get_powers_gbm(home=home, away=away))
print(store.get_powers_linear(home=home, away=away))
