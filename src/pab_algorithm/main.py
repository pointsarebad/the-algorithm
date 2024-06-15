from pab_algorithm.predictor.model_store import ModelStoreFactory
from pab_algorithm.predictor.team import Team

home = Team(elo=1721.0, power=1.744186046511628, name="Poland", code="PO")
away = Team(elo=1968.0, power=2.227272727272727, name="Netherlands", code="NL")

store = ModelStoreFactory.load_model_store("gbm")
print(store.get_powers(home=home, away=away))
